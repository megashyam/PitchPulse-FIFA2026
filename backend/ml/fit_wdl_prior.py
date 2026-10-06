"""
Fit the Elo gap -> W/D/L ordered logit used by ml/prior_builder.elo_to_wdl.

    x      production Elo gap before each match (prequential replay of the
           ml.elo_ratings update), plus h for the home side unless neutral
    model  P(home) = sigmoid(beta*x - theta), P(away) = sigmoid(-beta*x - theta),
           P(draw) = 1 - P(home) - P(away)
    data   martj42 internationals from 2000-01-01, before ELO_ASOF

Test: train 2000-2021, test 2022 to ELO_ASOF, vs the old hand-set draw curve.
Then refit on everything and write ml/wdl_prior_coef.json.

Usage:
    python -m ml.fit_wdl_prior [--results results.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from statsmodels.base.model import GenericLikelihoodModel

from ml import elo_ratings as er
from ml.metrics import log_loss
from ml.prior_builder import MIN_PROB, WDL_COEF_PATH

START = "2000-01-01"
SCALE = 400.0  # Elo points per fit unit
ELO_ASOF = "2026-06-11"
TEST_FROM = "2022-01-01"
BUCKETS = [(0, 100), (100, 300), (300, 500), (500, np.inf)]


def prequential(rows: list[dict]) -> list[tuple]:
    """(date, rating gap before the match, neutral, label, home goals, away goals);
    label 0 home, 1 draw, 2 away."""
    r: dict[str, float] = {}
    out = []
    for m in rows:
        if m["date"] >= ELO_ASOF:
            break
        rh, ra = r.get(m["home"], er.INITIAL), r.get(m["away"], er.INITIAL)
        if m["date"] >= START:
            y = 0 if m["hs"] > m["as"] else 1 if m["hs"] == m["as"] else 2
            out.append((m["date"], rh - ra, m["neutral"], y, m["hs"], m["as"]))
        # same update as er.run_elo
        dr = rh - ra + (0.0 if m["neutral"] else er.HOME_ADV)
        w = 1.0 if m["hs"] > m["as"] else 0.5 if m["hs"] == m["as"] else 0.0
        d = er.k_factor(m["tournament"]) * er.goal_mult(m["hs"] - m["as"]) * (w - er.expected(dr))
        r[m["home"]], r[m["away"]] = rh + d, ra - d
    return out


def _floor(P: np.ndarray) -> np.ndarray:
    P = np.maximum(P, MIN_PROB)
    return P / P.sum(-1, keepdims=True)


def ordlogit(x: np.ndarray, beta: float, theta: float) -> np.ndarray:
    ph = 1 / (1 + np.exp(-(beta * x - theta)))
    pa = 1 / (1 + np.exp(beta * x + theta))
    return np.stack([ph, 1 - ph - pa, pa], -1)


def hand_set(x: np.ndarray) -> np.ndarray:
    """previous production curve"""
    e = 1 / (1 + 10 ** (-x / 400))
    d = np.clip(0.25 * np.exp(-np.abs(x) / 450) + 0.05, 0.03, 0.30)
    return np.stack([(1 - d) * e, d, (1 - d) * (1 - e)], -1)


class _OrdLogit(GenericLikelihoodModel):
    """params in O(1) units (400*beta, theta, home_adv/400). One theta for both
    cutpoints, so swapping teams swaps W/L at neutral venues."""

    def __init__(self, y, s, home):
        super().__init__(y, None, extra_params_names=["beta", "theta", "home_adv"])
        self.s, self.home = s, home
        self.df_model, self.df_resid = 0.0, len(y) - 3.0  # all three are "extra" params

    def nloglikeobs(self, params):
        b, t, h = params
        x = self.s + h * self.home
        P = ordlogit(x, b, t)
        y = self.endog.astype(int)
        return -np.log(np.maximum(P[np.arange(len(y)), y], 1e-12))


def fit_result(dr: np.ndarray, home: np.ndarray, y: np.ndarray):
    return _OrdLogit(y, dr / SCALE, home).fit(start_params=[0.005 * SCALE, 0.6, 100.0 / SCALE], method="bfgs",
                   maxiter=1000, gtol=1e-8, disp=0)


def _rescale(v: np.ndarray) -> tuple[float, float, float]:
    """O(1) units -> (beta, theta, home_adv)"""
    return float(v[0] / SCALE), float(v[1]), float(v[2] * SCALE)


def fit(dr: np.ndarray, home: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    return _rescale(fit_result(dr, home, y).params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="local martj42 results.csv (default: download)")
    a = ap.parse_args()
    text = Path(a.results).read_text(encoding="utf-8") if a.results else None
    data = prequential(er.load_results(text))
    date = np.array([d[0] for d in data])
    dr = np.array([d[1] for d in data])
    home = np.array([0.0 if d[2] else 1.0 for d in data])
    y = np.array([d[3] for d in data])
    tr, te = date < TEST_FROM, date >= TEST_FROM
    print(f"train n={tr.sum()}  test n={te.sum()}")

    b, t, h = fit(dr[tr], home[tr], y[tr])
    print(f"train fit: beta={b:.6f} theta={t:.4f} home_adv={h:.1f}")
    P_old = _floor(hand_set(dr[te] + er.HOME_ADV * home[te]))
    P_new = _floor(ordlogit(dr[te] + h * home[te], b, t))
    P_base = np.tile(np.bincount(y[tr], minlength=3) / tr.sum(), (te.sum(), 1))
    ll = {"ordered_logit": log_loss(P_new, y[te]), "hand_set": log_loss(P_old, y[te]),
          "base_rate": log_loss(P_base, y[te])}
    print("test log loss: " + "  ".join(f"{k} {v:.4f}" for k, v in ll.items()))

    # buckets on the gap the Elo update itself uses
    gap = np.abs(dr[te] + er.HOME_ADV * home[te])
    print("test draw rate by gap: bucket  n  hand-set  ordered-logit  actual")
    for lo, hi in BUCKETS:
        m = (gap >= lo) & (gap < hi)
        print(f"  {lo:>3}-{hi:<4} {m.sum():5d}  {P_old[m, 1].mean():.3f}  {P_new[m, 1].mean():.3f}  "
              f"{(y[te][m] == 1).mean():.3f}")

    res = fit_result(dr, home, y)
    b, t, h = _rescale(res.params)
    sb, st, sh = _rescale(res.bse)
    coef = {
        "beta": round(b, 8),
        "theta": round(t, 6),
        "home_adv": round(h, 2),
        "se": {"beta": round(sb, 8), "theta": round(st, 6), "home_adv": round(sh, 2)},
        "trained_on": f"martj42 internationals {START} to {max(date)}",
        "n_matches": int(len(y)),
        "test": {
            "train": f"{START} to {max(date[tr])}",
            "test": f"{TEST_FROM} to {max(date[te])}",
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "log_loss": {k: round(float(v), 4) for k, v in ll.items()},
        },
    }
    WDL_COEF_PATH.write_text(json.dumps(coef, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(coef, indent=2))


if __name__ == "__main__":
    main()
