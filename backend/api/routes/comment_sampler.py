"""
Narrative comment sample storage and retrieval.

    GET /narrative/{topic}/comments   recent sample posts, newest first

Samples are kept per topic with a 10-minute TTL, capped at 20 via LTRIM.
"""

from __future__ import annotations
import json
import time
import logging

from fastapi import APIRouter, Request

router = APIRouter()
log = logging.getLogger(__name__)

SAMPLE_TTL = 600  # 10 minutes — keep it feeling "live"
SAMPLE_CAP = 19  # up to 20 samples per topic


async def store_comment_samples(redis_client, topic: str, samples: list[dict]) -> None:
    """Store sample posts for a topic.

    samples: [{"text", "source", "author", "permalink", "timestamp"}]
    """
    key = f"narrative:{topic}:comments"
    for s in samples:
        s.setdefault("timestamp", time.time())
        await redis_client.lpush(key, json.dumps(s))
    await redis_client.ltrim(key, 0, SAMPLE_CAP)
    await redis_client.expire(key, SAMPLE_TTL)


@router.get("/{topic}/comments")
async def get_comment_samples(topic: str, request: Request):
    """Recent sample posts for a topic, newest first."""
    r = request.app.state.redis
    raw = await r.lrange(f"narrative:{topic}:comments", 0, SAMPLE_CAP)
    samples = [json.loads(s) for s in raw]
    return {"topic": topic, "count": len(samples), "samples": samples}
