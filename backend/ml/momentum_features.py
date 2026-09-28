"""
Momentum model features.

One feature function shared by training (ml/momentum_trainer.py, replaying
ESPN play-by-play) and serving (ml/momentum_model.py), so there is no
train/serve skew.

Inputs:
    shots  [{"side": 1|2, "minute", "extra", "xg"}]      (feeds.espn shots)
    goals  [(minute, extra, side_credited)]
    reds   [(minute, extra, side_sent_off)]

Match time is continuous across periods so stoppage minutes never collide:
a fixed offset per period (1H 0, 2H +15, ET1 +40, ET2 +45).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

WINDOW = 15.0  # minutes of recent play
HORIZON = 5.0  # target: goal in the next 5 minutes
_OFFSET = {1: 0, 2: 15, 3: 40, 4: 45}

FEATURES = [
    "xg15_for",
    "xg15_against",
    "shots15_for",
    "shots15_against",
    "xg_for",
    "xg_against",
    "score_diff",
    "red_diff",
    "minute_norm",
]


def period_of(minute: int) -> int:
    return 1 if minute <= 45 else 2 if minute <= 90 else 3 if minute <= 105 else 4


def match_time(minute: int, extra: Optional[int] = None) -> float:
    return minute + (extra or 0) + _OFFSET[period_of(minute)]


def team_features(
    shots: Sequence[dict],
    goals: Iterable[tuple],
    reds: Iterable[tuple],
    t_now: float,
    side: int,
    minute_now: int,
) -> dict[str, float]:
    x15f = x15a = s15f = s15a = xf = xa = 0.0
    for s in shots:
        t = match_time(s["minute"], s.get("extra"))
        if t > t_now:
            continue
        mine = s["side"] == side
        xg = float(s["xg"])
        if mine:
            xf += xg
        else:
            xa += xg
        if t > t_now - WINDOW:
            if mine:
                x15f += xg
                s15f += 1
            else:
                x15a += xg
                s15a += 1
    gd = sum(
        (1 if g[2] == side else -1) for g in goals if match_time(g[0], g[1]) <= t_now
    )
    rd = sum(
        (1 if r[2] == side else -1) for r in reds if match_time(r[0], r[1]) <= t_now
    )
    return {
        "xg15_for": x15f,
        "xg15_against": x15a,
        "shots15_for": s15f,
        "shots15_against": s15a,
        "xg_for": xf,
        "xg_against": xa,
        "score_diff": max(-3, min(3, gd)) / 3.0,
        "red_diff": float(max(-2, min(2, rd))),
        "minute_norm": min(minute_now, 120) / 90.0,
    }


def predict(coef: dict, feats: dict[str, float]) -> float:
    z = coef["intercept"] + sum(coef["coef"][k] * feats[k] for k in FEATURES)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
