"""
Fit the Dixon-Coles rho used by ml/in_play for priors without a totals market.

    prior  ordered-logit W/D/L on the prequential Elo gap (ml.fit_wdl_prior)
    model  per match, 90' rates fitted to that W/D/L at rho
           (ml.in_play.rates_from_wdl_batch), Dixon-Coles pmf of the score
    rho    maximises the mean log-likelihood of the real final scores
    data   martj42 internationals from 2000-01-01, before ELO_ASOF

The 1X2 prior fixes W and D, so rho decides how many goals the fitted rates
imply: rho = 0 can only produce the draw rate by lowering total goals.

Test: prior and rho fitted on 2000-2021, scored on 2022 to ELO_ASOF against
rho = 0. Then rho is refitted on everything with the production prior
(ml/wdl_prior_coef.json) and written to ml/dixon_coles_coef.json.

Usage:
    python -m ml.fit_dixon_coles [--results results.csv]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ml import elo_ratings as er
from ml.fit_wdl_prior import START, TEST_FROM, _floor, fit, ordlogit, prequential
from ml.in_play import MAX_GOALS, RHO_PATH, _p_over, _score_matrix, rates_from_wdl_batch
from ml.prior_builder import WDL_COEF_PATH

RHO_RANGE = (-0.25, 0.10)
RHO_TOL = 1e-4


def score_ll(P: np.ndarray, hs: np.ndarray, as_: np.ndarray, rho: float) -> np.ndarray:
    """Per-match log-likelihood of the final score."""
    r = rates_from_wdl_batch(P, rho)
    J = _score_matrix(r[:, 0], r[:, 1], rho)
    return np.log(np.maximum(J[np.arange(len(P)), hs, as_], 1e-300))


def fit_rho(P: np.ndarray, hs: np.ndarray, as_: np.ndarray) -> float:
    """Golden-section search for the rho maximising the mean score log-likelihood."""
    g = (math.sqrt(5) - 1) / 2
    a, b = RHO_RANGE
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = (score_ll(P, hs, as_, x).mean() for x in (c, d))
    while b - a > RHO_TOL:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = score_ll(P, hs, as_, c).mean()
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = score_ll(P, hs, as_, d).mean()
    return (a + b) / 2


def goals_check(P, hs, as_, rho) -> str:
    r = rates_from_wdl_batch(P, rho)
    over = _p_over(r[:, 0], r[:, 1], 2.5, rho).mean()
    return (f"mean total {r.sum(1).mean():.3f} (actual {(hs + as_).mean():.3f})  "
            f"P(over 2.5) {over:.3f} (actual {(hs + as_ > 2.5).mean():.3f})")


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
    hs = np.minimum([d[4] for d in data], MAX_GOALS)
    as_ = np.minimum([d[5] for d in data], MAX_GOALS)
    tr, te = date < TEST_FROM, date >= TEST_FROM
    print(f"train n={tr.sum()}  test n={te.sum()}")

    b, t, h = fit(dr[tr], home[tr], y[tr])
    P = _floor(ordlogit(dr + h * home, b, t))
    rho = fit_rho(P[tr], hs[tr], as_[tr])
    print(f"train fit: rho={rho:.4f}")
    for r in (0.0, rho):
        print(f"test rho={r:+.4f}: score log loss {-score_ll(P[te], hs[te], as_[te], r).mean():.4f}  "
              + goals_check(P[te], hs[te], as_[te], r))

    c = json.loads(WDL_COEF_PATH.read_text(encoding="utf-8"))
    P = _floor(ordlogit(dr + c["home_adv"] * home, c["beta"], c["theta"]))
    rho = fit_rho(P, hs, as_)
    coef = {
        "rho": round(rho, 4),
        "trained_on": f"martj42 internationals {START} to {max(date)}",
        "n_matches": int(len(y)),
    }
    RHO_PATH.write_text(json.dumps(coef, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(coef, indent=2))


if __name__ == "__main__":
    main()
