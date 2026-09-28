"""
Narrative comment sample endpoints.

    GET /narrative/{topic}/comments            cached comment samples
    GET /narrative/{topic}/comments/seed-demo  dev helper: seed demo data

store_comment_samples dedupes by permalink (or a text hash) and writes
LPUSH + LTRIM + EXPIRE in one pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from fastapi import APIRouter, Depends, Request

from api.routes._security import require_trigger_token

router = APIRouter()
log = logging.getLogger(__name__)

SAMPLE_TTL = 600
SAMPLE_CAP = 19


def _dedupe_key(s: dict) -> str:
    if s.get("permalink"):
        return s["permalink"]
    return hashlib.sha1(s.get("text", "").encode()).hexdigest()


async def store_comment_samples(redis_client, topic: str, samples: list[dict]) -> None:
    key = f"narrative:{topic}:comments"

    existing_raw = await redis_client.lrange(key, 0, SAMPLE_CAP)
    seen = {_dedupe_key(json.loads(e)) for e in existing_raw}

    fresh = [s for s in samples if _dedupe_key(s) not in seen]
    if not fresh:
        return

    pipe = redis_client.pipeline(transaction=True)
    for s in fresh:
        s.setdefault("timestamp", time.time())
        pipe.lpush(key, json.dumps(s))
    pipe.ltrim(key, 0, SAMPLE_CAP)
    pipe.expire(key, SAMPLE_TTL)
    await pipe.execute()


@router.get("/{topic}/comments")
async def get_comment_samples(topic: str, request: Request):
    r = request.app.state.redis
    raw = await r.lrange(f"narrative:{topic}:comments", 0, SAMPLE_CAP)
    samples = [json.loads(s) for s in raw]
    return {"topic": topic, "count": len(samples), "samples": samples}


@router.get(
    "/{topic}/comments/seed-demo",
    dependencies=[Depends(require_trigger_token)],
)
async def seed_demo_comments(topic: str, request: Request):
    """Seed fixed demo samples flagged demo=True (dev only, token-gated)."""
    r = request.app.state.redis
    demo_samples = [
        {
            "text": f"{topic} looking sharp in the warmup, midfield press is intense today",
            "source": "mastodon",
            "author": "demo-1",
            "permalink": None,
            "demo": True,
        },
        {
            "text": f"anyone else think {topic} changes formation at half if this stays 0-0",
            "source": "mastodon",
            "author": "demo-2",
            "permalink": None,
            "demo": True,
        },
        {
            "text": f"{topic} fans are LOUD right now, whole section on their feet",
            "source": "bluesky",
            "author": "demo-3",
            "permalink": None,
            "demo": True,
        },
        {
            "text": f"stat check: {topic} have 68% possession but only 2 shots on target so far",
            "source": "mastodon",
            "author": "demo-4",
            "permalink": None,
            "demo": True,
        },
        {
            "text": f"{topic} keeper made a huge save there, momentum shift incoming",
            "source": "bluesky",
            "author": "demo-5",
            "permalink": None,
            "demo": True,
        },
    ]
    await store_comment_samples(r, topic, demo_samples)
    return {"status": "seeded", "topic": topic, "count": len(demo_samples)}
