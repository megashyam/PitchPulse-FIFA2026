"""
Match state stream.

    GET /matches/{fixture_id}/stream   SSE (pub/sub-backed)
"""

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from api.routes._sse import pubsub_sse

router = APIRouter()


@router.get("/{fixture_id}/stream")
async def match_stream(fixture_id: str, request: Request):
    r = request.app.state.redis

    async def generator():
        async for event in pubsub_sse(
            redis_client=r,
            channel="match_update",
            key=f"match:{fixture_id}:state",
            event_name="match_update",
            is_disconnected=request.is_disconnected,
            match_fixture_id=fixture_id,
        ):
            yield event

    return EventSourceResponse(generator())
