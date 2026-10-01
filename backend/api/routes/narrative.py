"""
Narrative Hub endpoints.

    GET      /narrative/spikes             last N spikes from the Redis feed
    GET      /narrative/spikes/{spike_id}  single spike by ID
    GET      /narrative/trending           tournament-wide trending topics
    GET      /narrative/stream             SSE (pub/sub-backed)
    GET      /narrative/arc/{spike_id}     fetch or generate the arc for a spike
    GET      /narrative/topic/{topic}/arc  arc for a row in the trending snapshot
    GET/POST /narrative/trigger            debug: force one detector tick (token-gated)
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from agents import narrative_arc_agent
from agents.narrative_spike_detector import get_detector
from api.routes._security import require_trigger_token
from api.routes._sse import pubsub_sse

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/spikes")
async def get_spikes(request: Request, limit: int = 20):
    """Return the last `limit` narrative spikes (newest first)."""
    r = request.app.state.redis
    raw_list = await r.lrange("narrative:spikes:feed", 0, min(limit - 1, 49))
    if not raw_list:
        return {
            "spikes": [],
            "count": 0,
            "message": (
                "No spikes yet. The narrative worker runs every 60s and detects "
                "anomalies after ~30 ticks. Use GET /narrative/trigger to force "
                "a tick immediately."
            ),
        }
    spikes = [json.loads(s) for s in raw_list]
    return {
        "spikes": spikes,
        "count": len(spikes),
        "updated_at": spikes[0].get("timestamp") if spikes else None,
    }


@router.get("/spikes/{spike_id}")
async def get_spike(spike_id: str, request: Request):
    r = request.app.state.redis
    raw = await r.get(f"narrative:spike:{spike_id}")
    if not raw:
        raise HTTPException(404, f"Spike {spike_id} not found")
    return json.loads(raw)


@router.get("/trending")
async def get_trending(request: Request, limit: int = 12):
    """All topics ranked by buzz, returned under `spikes` like the feed."""
    r = request.app.state.redis
    raw = await r.get("narrative:trending:latest")
    items = json.loads(raw) if raw else []
    return {
        "spikes": items[: min(limit, len(items))],
        "count": len(items),
        "updated_at": items[0].get("timestamp") if items else None,
    }


@router.get("/stream")
async def narrative_stream(request: Request):
    """SSE, pub/sub-backed; heartbeats come from pubsub_sse's fallback poll."""
    r = request.app.state.redis

    async def generator():
        async for event in pubsub_sse(
            redis_client=r,
            channel="narrative_spike",
            key="narrative:stream:latest",
            event_name="narrative_spike",
            is_disconnected=request.is_disconnected,
            fallback_poll_s=30.0,
        ):
            yield event

    return EventSourceResponse(generator(), ping=15)


@router.get("/arc/{spike_id}")
async def get_or_generate_arc(spike_id: str, request: Request):
    """Return the arc for a spike, generating it now if not already cached."""
    r = request.app.state.redis
    raw = await r.get(f"narrative:spike:{spike_id}")
    if not raw:
        raise HTTPException(404, f"Spike {spike_id} not found")

    spike_dict = json.loads(raw)

    if spike_dict.get("arc"):
        return {"spike_id": spike_id, "arc": spike_dict["arc"], "cached": True}

    from agents.narrative_spike_detector import NarrativeSpike

    try:
        spike = NarrativeSpike(
            spike_id=spike_dict["spike_id"],
            topic=spike_dict["topic"],
            tick=spike_dict["tick"],
            severity=spike_dict["severity"],
            sources=spike_dict["sources"],
            source_names=spike_dict.get("source_names", []),
            summary=spike_dict["summary"],
            timestamp=spike_dict.get("timestamp", 0),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to reconstruct spike: {e}")

    loop = asyncio.get_running_loop()
    arc = await narrative_arc_agent.synthesise(spike, loop)

    spike_dict["arc"] = arc
    await r.setex(f"narrative:spike:{spike_id}", 86_400, json.dumps(spike_dict))

    return {"spike_id": spike_id, "arc": arc, "cached": False}


@router.get("/topic/{topic}/arc")
async def get_topic_arc(topic: str, request: Request):
    """Arc for a trending row the worker didn't pre-fill (below ARC_TOP_N)."""
    from api.workers.narrative_worker import arc_for_row

    r = request.app.state.redis
    raw = await r.get("narrative:trending:latest")
    row = next((x for x in json.loads(raw) if x.get("topic") == topic), None) if raw else None
    if row is None:
        raise HTTPException(404, f"Topic {topic} not in trending snapshot")
    if row.get("arc"):
        return {"topic": topic, "arc": row["arc"], "cached": True}
    arc = await arc_for_row(r, row, asyncio.get_running_loop())
    return {"topic": topic, "arc": arc, "cached": False}


@router.api_route(
    "/trigger",
    methods=["GET", "POST"],
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_narrative(request: Request):
    """Force one detector tick (token-gated)."""
    r = request.app.state.redis
    loop = asyncio.get_running_loop()
    detector = get_detector()

    spikes = await detector.tick()
    if not spikes:
        tick = detector._tick_count
        remaining = detector.warmup_remaining()
        if remaining > 0:
            return {
                "status": "warming_up",
                "tick": tick,
                "message": (
                    f"Spike scorer needs {remaining} more ticks of live baseline "
                    "before scoring."
                ),
            }
        return {
            "status": "no_spikes",
            "tick": tick,
            "message": "No surges this tick — all topics within baseline.",
        }

    results = []
    for spike in spikes:
        arc = await narrative_arc_agent.synthesise(spike, loop)
        spike.arc = arc
        spike_dict = spike.to_dict()
        spike_json = json.dumps(spike_dict)
        await r.setex(f"narrative:spike:{spike.spike_id}", 86_400, spike_json)
        await r.lpush("narrative:spikes:feed", spike_json)
        await r.ltrim("narrative:spikes:feed", 0, 49)
        await r.expire("narrative:spikes:feed", 86_400)
        await r.setex("narrative:stream:latest", 3_600, spike_json)
        await r.publish("narrative_spike", spike_json)
        results.append(spike_dict)
        log.info(f"Trigger: stored spike {spike.spike_id}")

    return {"status": "written", "spikes_detected": len(results), "spikes": results}
