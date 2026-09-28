"""
Live momentum model.

P(each team scores in the next 5 minutes) from a logistic model over recent
shot quality, score and red cards; each side's share of the two
probabilities is its "momentum".

Stateless: features come from the match's shot list and events via
ml/momentum_features.team_features, the same function the trainer uses.
goal_prob_5min is the model's calibrated output (ml/momentum_report.json).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from api.schemas.event_types import LIVE_STATUSES
from api.schemas.schema import MatchState
from ml.momentum_features import WINDOW, match_time, predict, team_features

log = logging.getLogger(__name__)

COEF_PATH = Path(__file__).parent / "momentum_coef.json"
coef = json.loads(COEF_PATH.read_text(encoding="utf-8"))


def _goals_and_reds(state: MatchState) -> tuple[list, list]:
    goals, reds = [], []
    for e in state.events:
        if e.type in ("goal", "penalty_goal"):
            goals.append((e.elapsed, e.extra, e.team_id))
        elif e.type == "own_goal":
            goals.append((e.elapsed, e.extra, 3 - e.team_id))
        elif e.type in ("red", "yellow_red") and e.source == "espn":
            reds.append((e.elapsed, e.extra, e.team_id))
    return goals, reds


def update(state: MatchState, shots: list[dict]) -> Optional[dict]:
    """Momentum for a live match, else None; `shots` is match:{id}:shots."""
    if state.status_short not in LIVE_STATUSES:
        return None
    minute = state.elapsed or 0
    t_now = match_time(minute, state.elapsed_extra)
    goals, reds = _goals_and_reds(state)

    out = {}
    for side, key in ((1, "home"), (2, "away")):
        f = team_features(shots, goals, reds, t_now, side, minute)
        out[key] = {"p": predict(coef, f), "f": f}
    total = out["home"]["p"] + out["away"]["p"]

    def block(key: str) -> dict:
        f = out[key]["f"]
        return {
            "momentum_score": round(out[key]["p"] / total, 4),
            "goal_prob_5min": round(out[key]["p"], 4),
            "xg_15min": round(f["xg15_for"], 3),
            "shots_15min": int(f["shots15_for"]),
            "xg_total": round(f["xg_for"], 3),
        }

    return {
        "fixture_id": state.fixture_id,
        "home_name": state.home_name,
        "away_name": state.away_name,
        "elapsed": minute,
        "window_min": WINDOW,
        "stats_source": state.stats_source,
        "home": block("home"),
        "away": block("away"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
