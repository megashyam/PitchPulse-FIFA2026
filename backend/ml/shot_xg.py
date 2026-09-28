"""
Shot-level xG for ESPN/Opta commentary plays.

Each ESPN shot play carries pitch coordinates plus a commentary line naming
body part and assist type. Two coordinate conventions:
    - 0-100 Opta units attacking toward x=100 (WC 2026)
    - 0-1 fractions with x = distance from the goal line (2022-2025 feeds);
      OLD_X_SCALE metres per unit, matched to WC 2026 shot-location deciles

Features are distance/angle to goal plus those flags; coefficients come from
ml/fit_shot_xg.py and live in ml/shot_xg_coef.json.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PITCH_X, PITCH_Y, GOAL_W = 105.0, 68.0, 7.32
OLD_X_SCALE = 56.5
COEF_PATH = Path(__file__).with_name("shot_xg_coef.json")

SHOT_PLAY_TYPES = {
    "Shot Off Target",
    "Shot On Target",
    "Shot Blocked",
    "Shot Hit Woodwork",
    "Goal",
    "Goal - Header",
    "Goal - Free-kick",
    "Goal - Volley",
    "Penalty - Scored",
    "Penalty - Missed",
    "Penalty - Saved",
}
GOAL_PLAY_TYPES = {"Goal", "Goal - Header", "Goal - Free-kick", "Goal - Volley", "Penalty - Scored"}

FEATURES = [
    "log_dist",
    "angle",
    "header",
    "header_x_log_dist",
    "cross",
    "through_ball",
    "corner",
    "set_piece",
    "fast_break",
    "direct_fk",
]

_MINUTE_RE = re.compile(r"(\d+)'(?:\+(\d+)')?")


@dataclass
class Shot:
    minute: int
    extra: Optional[int]
    period: int
    team_id: str
    team_name: str
    player: Optional[str]
    x: float
    y: float
    header: bool
    penalty: bool
    direct_fk: bool
    cross: bool
    through_ball: bool
    corner: bool
    set_piece: bool
    fast_break: bool
    outcome: str  # goal | saved | missed | blocked | woodwork
    xg: float = 0.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def parse_clock(display: str) -> tuple[int, Optional[int]]:
    """"45'+4'" → (45, 4); "67'" → (67, None)."""
    m = _MINUTE_RE.match(display or "")
    if not m:
        return 0, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def _outcome(ptype: str) -> str:
    if ptype in GOAL_PLAY_TYPES:
        return "goal"
    if ptype in ("Shot On Target", "Penalty - Saved"):
        return "saved"
    if ptype == "Shot Blocked":
        return "blocked"
    if ptype == "Shot Hit Woodwork":
        return "woodwork"
    return "missed"


def shot_from_play(play: dict) -> Optional[Shot]:
    """ESPN commentary `play` → Shot; None unless a non-shootout shot with coordinates."""
    ptype = ((play.get("type") or {}).get("text")) or ""
    if ptype not in SHOT_PLAY_TYPES:
        return None
    period = int((play.get("period") or {}).get("number") or 0)
    if period >= 5 or play.get("shootout"):
        return None
    x, y = play.get("fieldPositionX"), play.get("fieldPositionY")
    if x is None or y is None:
        return None
    text = (play.get("text") or "").lower()
    minute, extra = parse_clock((play.get("clock") or {}).get("displayValue", ""))
    team = play.get("team") or {}
    parts = play.get("participants") or []
    player = ((parts[0].get("athlete") or {}).get("displayName")) if parts else None
    return Shot(
        minute=minute,
        extra=extra,
        period=period,
        team_id=str(team.get("id") or ""),
        team_name=team.get("displayName") or "",
        player=player,
        x=float(x),
        y=float(y),
        header="header" in text,
        penalty=ptype.startswith("Penalty"),
        direct_fk="direct free kick" in text or ptype == "Goal - Free-kick",
        cross="with a cross" in text,
        through_ball="through ball" in text,
        corner="following a corner" in text,
        set_piece="set piece" in text,
        fast_break="fast break" in text,
        outcome=_outcome(ptype),
    )


def to_metres(x: float, y: float) -> tuple[float, float]:
    """→ (metres from goal line, metres from centre line of the goal)."""
    if x <= 1.0 and y <= 1.0:
        return x * OLD_X_SCALE, abs(0.5 - y) * PITCH_Y
    return (100.0 - x) / 100.0 * PITCH_X, abs(50.0 - y) / 100.0 * PITCH_Y


def features(s: Shot) -> dict[str, float]:
    dx, dy = to_metres(s.x, s.y)
    dx = max(0.5, dx)
    dist = math.hypot(dx, dy)
    # Angle subtended by the goal mouth.
    angle = math.atan2(GOAL_W * dx, dx * dx + dy * dy - (GOAL_W / 2) ** 2)
    if angle < 0:
        angle += math.pi
    log_dist = math.log(dist)
    return {
        "log_dist": log_dist,
        "angle": angle,
        "header": float(s.header),
        "header_x_log_dist": float(s.header) * log_dist,
        "cross": float(s.cross),
        "through_ball": float(s.through_ball),
        "corner": float(s.corner),
        "set_piece": float(s.set_piece),
        "fast_break": float(s.fast_break),
        "direct_fk": float(s.direct_fk),
    }


_COEF: Optional[dict] = None


def _coef() -> dict:
    global _COEF
    if _COEF is None:
        _COEF = json.loads(COEF_PATH.read_text(encoding="utf-8"))
    return _COEF


def xg(s: Shot) -> float:
    c = _coef()
    if s.penalty:
        return float(c["penalty_xg"])
    f = features(s)
    z = c["intercept"] + sum(c["coef"][k] * f[k] for k in FEATURES)
    return 1.0 / (1.0 + math.exp(-z))


def shots_from_commentary(commentary: list[dict]) -> list[Shot]:
    out: list[Shot] = []
    for c in commentary:
        s = shot_from_play(c.get("play") or {})
        if s is not None:
            s.xg = round(xg(s), 4)
            out.append(s)
    return out
