"""
agents/narrative_topics.py
============================
Fixture-aware topic tracking for the Narrative Hub.

Provides dynamic topic discovery by combining active and recently completed
fixtures into a unified tracking set. Team names are used as detector topics,
while metadata maps each topic back to fixture context for downstream narrative
generation.

Key capabilities:
    - Single Redis SUNION + MGET pipeline for efficient fixture retrieval.
    - Tracks live, upcoming, and recently completed matches.
    - Retains completed fixtures for post-match narrative signals.
    - Provides fixture metadata including status and home/away context.
    - Generates compact match context strings for LLM grounding.

Consumed by narrative detection and generation services to connect real-time
match state with emerging narrative signals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from api.schemas.event_types import COMPLETED_STATUSES

COMPLETED_RETENTION_HOURS = 48.0


async def get_topic_meta(redis_client) -> Dict[str, dict]:
    """
    Retrieve metadata for all currently tracked narrative topics.

    Combines active and recently completed fixtures from Redis, loads match
    states in a single batch operation, filters expired completed matches, and
    maps team names to their associated fixture context.

    Args:
        redis_client: Async Redis client used for fixture state retrieval.

    Returns:
        Dict[str, dict]: Mapping of team/topic names to fixture metadata:
            {
                "fixture_id": int,
                "status_short": str,
                "is_home": bool
            }

    Notes:
        If a team appears in multiple fixtures, the most relevant fixture state
        is retained based on update ordering and completion status.
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
    """
    Generate the current set of narrative detection topics.

    Retrieves active fixture metadata and converts tracked teams into a flat
    topic list for downstream anomaly detection. Always includes the global
    tournament-level topic.

    Args:
        redis_client: Async Redis client used for fixture lookup.

    Returns:
        List[str]: Sorted list of tracked team names plus the WC2026 topic.
    """
    meta = await get_topic_meta(redis_client)
    topics = set(meta.keys())
    topics.add("WC2026")
    return sorted(topics)


def build_match_context(state) -> str:
    """
    Convert a match state object into an LLM-readable context summary.

    Creates a compact natural-language representation containing teams,
    match status, score information, and recent goal events. The generated
    context is used to ground narrative generation agents with live fixture
    information.

    Args:
        state: Validated match state containing teams, scores, status, and events.

    Returns:
        str: Human-readable match context suitable for LLM prompting.

    Example:
        "Argentina vs France. 67' — 2-1. Goals: 23' Messi (Argentina)."
    """
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
