"""
Fixture-aware topic tracking for the Narrative Hub.

One SUNION(matches:active, matches:completed) plus one MGET per tick
produces both:
    get_tracked_topics()  topic strings for the detector: live, upcoming
                          and recently finished team names, plus "WC2026"
    get_topic_meta()      topic → {fixture_id, status_short, is_home}

Completed fixtures stay tracked through the retention window.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from api.schemas.event_types import COMPLETED_STATUSES

COMPLETED_RETENTION_HOURS = 48.0  # keep tracking a finished match's teams
# this long after full-time — post-match reaction chatter is real signal


async def get_topic_meta(redis_client) -> Dict[str, dict]:
    """{topic: {"fixture_id", "status_short", "is_home"}} for tracked teams.

    A team in several fixtures maps to the most recently updated one.
    """
    from api.schemas.schema import MatchState

    fixture_ids = await redis_client.sunion("matches:active", "matches:completed")
    if not fixture_ids:
        return {}

    keys = [f"match:{fid}:state" for fid in fixture_ids]
    raw_values = await redis_client.mget(keys)

    meta: Dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    for raw in raw_values:
        if not raw:
            continue
        try:
            state = MatchState.model_validate_json(raw)
        except Exception:
            continue

        is_completed = state.status_short in COMPLETED_STATUSES
        if is_completed:
            try:
                age_hours = (
                    now - state.updated_at.replace(tzinfo=timezone.utc)
                ).total_seconds() / 3600
            except Exception:
                age_hours = 0
            if age_hours > COMPLETED_RETENTION_HOURS:
                continue

        for team_name, is_home in ((state.home_name, True), (state.away_name, False)):
            existing = meta.get(team_name)
            if existing is None or state.status_short not in COMPLETED_STATUSES:
                meta[team_name] = {
                    "fixture_id": state.fixture_id,
                    "status_short": state.status_short,
                    "is_home": is_home,
                }

    return meta


async def get_tracked_topics(redis_client) -> List[str]:
    """Flat topic list for the detector — team names + global tournament topic."""
    meta = await get_topic_meta(redis_client)
    topics = set(meta.keys())
    topics.add("WC2026")
    return sorted(topics)


def build_match_context(state) -> str:
    """Short match context string that grounds arc synthesis."""
    parts = [f"{state.home_name} vs {state.away_name}"]

    if state.status_short in COMPLETED_STATUSES:
        parts.append(f"Full time {state.home_score}-{state.away_score}")
    elif state.status_short == "NS":
        parts.append("kickoff not yet started")
    else:
        parts.append(f"{state.elapsed or 0}' — {state.home_score}-{state.away_score}")

    goal_events = [
        e for e in state.events if e.type in ("goal", "own_goal", "penalty_goal")
    ]
    if goal_events:
        scorer_strs = []
        for e in goal_events[:5]:
            name = e.player_name or "unknown scorer"
            scorer_strs.append(f"{e.elapsed}' {name} ({e.team_name})")
        parts.append("Goals: " + "; ".join(scorer_strs))

    return ". ".join(parts) + "."
