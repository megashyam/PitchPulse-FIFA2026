"""
Momentum endpoints.

    GET/POST /matches/{fixture_id}/momentum/trigger  debug: force one cycle (token-gated)
    GET      /matches/{fixture_id}/momentum          latest snapshot
    GET      /matches/{fixture_id}/momentum/stream   SSE (pub/sub-backed)
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from api.routes._security import require_trigger_token
from api.routes._sse import pubsub_sse
from api.schemas.event_types import LIVE_STATUSES
from api.schemas.schema import MatchState
from ml import momentum_model

router = APIRouter()
log = logging.getLogger(__name__)


@router.api_route(
    "/{fixture_id}/momentum/trigger",
    methods=["GET", "POST"],
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_momentum(fixture_id: str, request: Request):
    """Run one momentum update cycle (token-gated)."""
    r = request.app.state.redis
    raw = await r.get(f"match:{fixture_id}:state")

    if not raw:
        raise HTTPException(
            status_code=404,
            detail=f"No match state found for fixture {fixture_id}. "
            "Is the match producer running?",
        )

    try:
        state = MatchState.model_validate_json(raw)
    except Exception as exc:
        return {"status": "error", "stage": "parse_match_state", "error": str(exc)}

    try:
        shots_raw = await r.get(f"match:{fixture_id}:shots")
        result = momentum_model.update(state, json.loads(shots_raw) if shots_raw else [])
    except Exception as exc:
        return {"status": "error", "stage": "momentum_model.update", "error": str(exc)}

    if result is None:
        return {
            "status": "skipped",
            "reason": f"status_short={state.status_short!r} is not in the live set "
            "{'1H','HT','2H','ET','P'}",
            "elapsed": state.elapsed,
            "status_short": state.status_short,
        }

    try:
        await r.setex(f"match:{fixture_id}:momentum", 3600, json.dumps(result))
        await r.publish("momentum_update", json.dumps({"fixture_id": state.fixture_id}))
    except Exception as exc:
        return {"status": "error", "stage": "redis_write", "error": str(exc)}

    return {"status": "written", "result": result}


@router.get("/{fixture_id}/momentum")
async def get_momentum(fixture_id: str, request: Request):
    """Latest momentum snapshot for this fixture.

    Non-live matches return 200 {"status": "not_started"}; a live match
    without data yet (before the first worker tick) returns 404.
    """
    r = request.app.state.redis
    raw = await r.get(f"match:{fixture_id}:momentum")

    if not raw:
        state_raw = await r.get(f"match:{fixture_id}:state")
        if state_raw:
            try:
                state = MatchState.model_validate_json(state_raw)
                if state.status_short not in LIVE_STATUSES:
                    return {
                        "fixture_id": int(fixture_id),
                        "status": "not_started",
                        "match_status": state.status_short,
                        "message": "Momentum tracking only runs while a match is live.",
                    }
            except Exception:
                pass
        raise HTTPException(
            status_code=404,
            detail=(
                f"No momentum data for fixture {fixture_id}. "
                "Check that the match is live and the momentum worker is running."
            ),
        )
    return json.loads(raw)


@router.get("/{fixture_id}/momentum/stream")
async def momentum_stream(fixture_id: str, request: Request):
    """SSE stream, pub/sub-backed."""
    r = request.app.state.redis

    async def generator():
        async for event in pubsub_sse(
            redis_client=r,
            channel="momentum_update",
            key=f"match:{fixture_id}:momentum",
            event_name="momentum_update",
            is_disconnected=request.is_disconnected,
            match_fixture_id=fixture_id,
        ):
            yield event

    return EventSourceResponse(generator(), ping=15)
