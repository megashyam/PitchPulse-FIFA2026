"""
StatsBomb open-data constants and event parsing helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43            # FIFA World Cup
SEASON_IDS = [106, 3]          # WC 2022, WC 2018
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "statsbomb"


def load(path: str, client: Optional[httpx.Client] = None) -> list | dict:
    """GET {SB_BASE}/{path} with an on-disk cache. Blocking."""
    f = CACHE_DIR / path
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    own = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        r = client.get(f"{SB_BASE}/{path}")
        r.raise_for_status()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(r.text, encoding="utf-8")
        return r.json()
    finally:
        if own:
            client.close()


def goals_in_regulation(events: list[dict], home: str, away: str) -> tuple[int, int]:
    """Score after 90 minutes (periods 1-2), own goals ("Own Goal For") included."""
    score = {home: 0, away: 0}
    for e in events:
        if e.get("period", 1) > 2:
            continue
        etype = (e.get("type") or {}).get("name", "")
        team = (e.get("team") or {}).get("name", "")
        is_goal = etype == "Own Goal For" or (
            etype == "Shot" and ((e.get("shot") or {}).get("outcome") or {}).get("name") == "Goal"
        )
        if is_goal and team in score:
            score[team] += 1
    return score[home], score[away]

# Shot outcomes that count as "on target". Compared case-insensitively —
# StatsBomb data contains both "Saved to Post" and "Saved To Post" variants.
ON_TARGET_SHOT_OUTCOMES = {
    "goal",
    "saved",
    "saved to post",
    "saved twice",
}

GK_SAVE_OUTCOMES = {
    "touched out",
    "success",
    "in play safe",
    "collected twice",
    "success in play",
    "success out",
}


def shot_is_on_target(outcome_name: str) -> bool:
    return (outcome_name or "").strip().lower() in ON_TARGET_SHOT_OUTCOMES


def gk_is_save(outcome_name: str) -> bool:
    return (outcome_name or "").strip().lower() in GK_SAVE_OUTCOMES


def card_from_event(ev: dict) -> Optional[str]:
    """'red' | 'yellow' | None for a StatsBomb event.

    Reads Foul Committed and Bad Behaviour cards; a second yellow is red.
    """
    etype = (ev.get("type") or {}).get("name", "")
    if etype == "Foul Committed":
        card = (ev.get("foul_committed", {}).get("card") or {}).get("name")
    elif etype == "Bad Behaviour":
        card = (ev.get("bad_behaviour", {}).get("card") or {}).get("name")
    else:
        return None
    if not card:
        return None
    if "Red" in card or "Second Yellow" in card:
        return "red"
    if "Yellow" in card:
        return "yellow"
    return None


def sort_events(events: list[dict]) -> list[dict]:
    """Events in chronological order, without period 5 (penalty shootout)."""
    return sorted(
        (e for e in events if e.get("period", 1) < 5),
        key=lambda e: (e.get("period", 1), e.get("minute", 0), e.get("index", 0)),
    )
