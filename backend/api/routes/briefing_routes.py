"""
Pre-match briefing endpoints.

    GET      /briefing          latest briefing
    GET      /briefing/feed     full history for this match
    GET/POST /briefing/trigger  generate one if match status changed since the
                                last generation (token-gated)
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from agents import briefing_agent
from api.routes._security import require_trigger_token
from api.schemas.event_types import COMPLETED_STATUSES
from api.schemas.schema import MatchState

router = APIRouter()
log = logging.getLogger(__name__)

TTL_LIVE = 86_400
TTL_COMPLETED = 2_592_000
FEED_CAP = 9  # up to 10 briefings per match — kickoff/HT/FT plus room


def _ttl_for(status_short: str) -> int:
    return TTL_COMPLETED if status_short in COMPLETED_STATUSES else TTL_LIVE


@router.get("/{fixture_id}/briefing")
async def get_briefing(fixture_id: str, request: Request):
    """Latest briefing."""
    r = request.app.state.redis
    raw = await r.lindex(f"match:{fixture_id}:briefing:feed", 0)
    if not raw:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No briefing for fixture {fixture_id}. "
                "Generate one: GET /matches/{fixture_id}/briefing/trigger"
            ),
        )
    return json.loads(raw)


@router.get("/{fixture_id}/briefing/feed")
async def get_briefing_feed(fixture_id: str, request: Request):
    """Full briefing history for this match, newest first."""
    r = request.app.state.redis
    raw = await r.lrange(f"match:{fixture_id}:briefing:feed", 0, FEED_CAP)
    entries = [json.loads(e) for e in raw]
    return {"fixture_id": int(fixture_id), "count": len(entries), "briefings": entries}


@router.api_route(
    "/{fixture_id}/briefing/trigger",
    methods=["GET", "POST"],
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_briefing(fixture_id: str, request: Request):
    """Generate a briefing only if match status changed since the last one."""
    r = request.app.state.redis
    state_raw = await r.get(f"match:{fixture_id}:state")

    if not state_raw:
        raise HTTPException(
            status_code=404,
            detail=f"No match state for fixture {fixture_id}. Is the producer running?",
        )

    state = MatchState.model_validate_json(state_raw)
    current_status = state.status_short

    last_status_key = f"match:{fixture_id}:briefing:last_status"
    last_status = await r.get(last_status_key)

    if last_status is not None and last_status == current_status:
        existing_raw = await r.lindex(f"match:{fixture_id}:briefing:feed", 0)
        if existing_raw:
            existing = json.loads(existing_raw)
            existing["status"] = "unchanged"
            return existing

    text, model_label = await briefing_agent.generate(
        home_name=state.home_name,
        away_name=state.away_name,
        fixture_id=state.fixture_id,
    )

    result = {
        "fixture_id": int(fixture_id),
        "home_name": state.home_name,
        "away_name": state.away_name,
        "match_status": current_status,
        "briefing": text,
        "model": model_label,
        "source": "kickoff facts (ESPN, Elo, head-to-head)"
        + ("" if model_label == "template" else " + Groq"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "generated",
    }

    ttl = _ttl_for(current_status)
    result_json = json.dumps(result)

    pipe = r.pipeline(transaction=True)
    pipe.lpush(f"match:{fixture_id}:briefing:feed", result_json)
    pipe.ltrim(f"match:{fixture_id}:briefing:feed", 0, FEED_CAP)
    pipe.expire(f"match:{fixture_id}:briefing:feed", ttl)
    pipe.setex(last_status_key, ttl, current_status)
    await pipe.execute()

    log.info(
        f"[{fixture_id}] New briefing generated for status={current_status} "
        f"via {model_label}"
    )
    return result
