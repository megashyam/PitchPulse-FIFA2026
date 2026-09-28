"""
Momentum model trainer.

Fits P(team scores in the next 5 minutes) from recent shot quality, score
and red cards, using the same feature function as the live model
(ml/momentum_features.py).

Data: ESPN play-by-play summaries cached by ml/fit_shot_xg.py
(--cache/<league>/<event>.json); one sample per team per minute of play.
    train     80% of club matches (split by match, seeded)
    held-out  the other 20%
    external  --wc-dirs (e.g. WC 2022, WC 2026 summaries), never trained on

Reports log loss and Brier vs the base rate and a score-and-clock-only
model, plus expected calibration error, to ml/momentum_report.json;
coefficients go to ml/momentum_coef.json.

Usage:
    python -m ml.momentum_trainer --cache DIR --wc-dirs DIR1,DIR2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from feeds.espn import _events
from ml import shot_xg
from ml.fit_shot_xg import _fit_logistic, _logloss
from ml.momentum_features import (
    FEATURES,
    HORIZON,
    match_time,
    period_of,
    team_features,
)
from ml.team_names import canonical

HERE = Path(__file__).parent
COEF_PATH = HERE / "momentum_coef.json"
REPORT_PATH = HERE / "momentum_report.json"
BASELINE_FEATURES = ["score_diff", "red_diff", "minute_norm"]
_PERIOD_START = {1: (1, 0), 2: (46, 15), 3: (91, 40), 4: (106, 45)}
_PERIOD_END_MIN = {1: 45, 2: 90, 3: 105, 4: 120}


def match_samples(summary: dict) -> list[tuple[dict, int]]:
    """[(features, scored_next_5)] for both teams at every minute of play."""
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    ids, by_name = {}, {}
    for c in comp.get("competitors", []):
        side = 1 if c.get("homeAway") == "home" else 2
        name = canonical(c["team"]["displayName"])
        ids[str(c["team"]["id"])] = (side, name)
        by_name[name] = side
    if len(ids) != 2:
        return []

    shots = []
    for s in shot_xg.shots_from_commentary(summary.get("commentary", [])):
        side = ids.get(s.team_id, (0, ""))[0] or by_name.get(canonical(s.team_name), 0)
        if side:
            shots.append({"side": side, "minute": s.minute, "extra": s.extra, "xg": s.xg})

    goals, reds = [], []
    for e in _events(summary.get("keyEvents", []), ids):
        if e.type in ("goal", "penalty_goal"):
            goals.append((e.elapsed, e.extra, e.team_id))
        elif e.type == "own_goal":
            goals.append((e.elapsed, e.extra, 3 - e.team_id))
        elif e.type == "red":
            reds.append((e.elapsed, e.extra, e.team_id))

    # Period extents from every commentary timestamp.
    ends: dict[int, float] = {}
    for c in summary.get("commentary", []):
        m, x = shot_xg.parse_clock(((c.get("time") or {}).get("displayValue")) or "")
        if m:
            p = period_of(m)
            ends[p] = max(ends.get(p, 0.0), match_time(m, x))
    goal_t = [(match_time(g[0], g[1]), g[2]) for g in goals]

    out = []
    for p, end in ends.items():
        start_min, off = _PERIOD_START[p]
        t = float(start_min + off)
        while t <= end:
            minute_now = min(int(t - off), _PERIOD_END_MIN[p])
            for side in (1, 2):
                f = team_features(shots, goals, reds, t, side, minute_now)
                y = int(any(side == s and t < gt <= t + HORIZON for gt, s in goal_t))
                out.append((f, y))
            t += 1.0
    return out


def _load(dirs: list[Path]) -> list[list[tuple[dict, int]]]:
    out = []
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(s, dict):
                m = match_samples(s)
                if m:
                    out.append(m)
    return out


def _xy(samples, keys) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[f[k] for k in keys] for f, _ in samples])
    y = np.array([y for _, y in samples], dtype=float)
    return X, y


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        pm, ym = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(pm - ym)
        table.append({"mean_pred": round(pm, 4), "observed": round(ym, 4), "n": int(m.sum())})
    return round(float(ece), 4), table


def _evaluate(samples, w, bw, base_rate) -> dict:
    X, y = _xy(samples, FEATURES)
    Xb, _ = _xy(samples, BASELINE_FEATURES)
    p = 1 / (1 + np.exp(-(w[0] + X @ w[1])))
    pb = 1 / (1 + np.exp(-(bw[0] + Xb @ bw[1])))
    ece, table = _ece(p, y)
    return {
        "n": int(len(y)),
        "goal_rate": round(float(y.mean()), 4),
        "logloss": round(_logloss(p, y), 4),
        "logloss_score_clock_only": round(_logloss(pb, y), 4),
        "logloss_base_rate": round(_logloss(np.full_like(y, base_rate), y), 4),
        "brier": round(float(((p - y) ** 2).mean()), 5),
        "brier_base_rate": round(float(((base_rate - y) ** 2).mean()), 5),
        "ece": ece,
        "reliability": table,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--leagues", default="eng.1,esp.1,ger.1,ita.1,fra.1")
    ap.add_argument("--wc-dirs", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    club = _load([Path(a.cache) / lg for lg in a.leagues.split(",")])
    rng = random.Random(a.seed)
    rng.shuffle(club)
    cut = int(0.8 * len(club))
    train = [s for m in club[:cut] for s in m]
    held = [s for m in club[cut:] for s in m]

    X, y = _xy(train, FEATURES)
    w = _fit_logistic(X, y)
    Xb, _ = _xy(train, BASELINE_FEATURES)
    bw = _fit_logistic(Xb, y)
    base_rate = float(y.mean())

    COEF_PATH.write_text(
        json.dumps(
            {
                "intercept": w[0],
                "coef": dict(zip(FEATURES, map(float, w[1]))),
                "features": FEATURES,
                "target": "team scores within the next 5 minutes of match time",
                "trained_on": f"ESPN {a.leagues} ({cut} matches, {len(y)} samples)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "train": {"matches": cut, "samples": int(len(y)), "goal_rate": round(base_rate, 4)},
        "held_out_club": _evaluate(held, w, bw, base_rate),
    }
    for d in filter(None, a.wc_dirs.split(",")):
        ms = _load([Path(d)])
        report[f"external:{Path(d).name}"] = {
            "matches": len(ms),
            **_evaluate([s for m in ms for s in m], w, bw, base_rate),
        }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "reliability"} for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
