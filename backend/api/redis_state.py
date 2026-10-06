"""
Shared Redis helpers for workers and routes: match-state loading, feed TTLs
and capped feed writes.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, Optional

import redis.asyncio as aioredis
from fastapi import HTTPException

from schemas.event_types import COMPLETED_STATUSES
from schemas.schema import MatchState

from config import TTL_COMPLETED, TTL_LIVE

log = logging.getLogger(__name__)




async def load_state(r: aioredis.Redis, fid) -> Optional[MatchState]:
    """match:{fid}:state, or None if missing or unparsable."""
    raw = await r.get(f"match:{fid}:state")
    if not raw:
        return None
    try:
        return MatchState.model_validate_json(raw)
    except Exception as exc:
        log.warning(f"[{fid}] MatchState parse error: {exc}")
        return None


async def require_state(r: aioredis.Redis, fid) -> MatchState:
    """load_state for routes: 404 when there is no usable state."""
    state = await load_state(r, fid)
    if state is None:
        raise HTTPException(404, f"No match state for fixture {fid}. Is the producer running?")
    return state


def ttl_for(status_short: str, live: int = TTL_LIVE, completed: int = TTL_COMPLETED) -> int:
    return completed if status_short in COMPLETED_STATUSES else live


def push_feed(pipe, key: str, entries: Iterable[dict], cap: int, ttl: int) -> None:
    """LPUSH entries in order (so the list reads newest-first), keep cap+1, set TTL."""
    for e in entries:
        pipe.lpush(key, json.dumps(e))
    pipe.ltrim(key, 0, cap)
    pipe.expire(key, ttl)


async def completed_backlog(
    r: aioredis.Redis,
    active: set,
    completed: set,
    key_fmt: str,
    limit: Optional[int] = None,
    done: Optional[set] = None,
) -> set:
    """Completed fixtures (not active) with no `key_fmt.format(fid)` key, in id
    order. `done` remembers fixtures already checked this process; `limit`
    caps how many are returned per call."""
    out: set = set()
    for fid in sorted(set(completed) - set(active) - (done or set())):
        if limit is not None and len(out) >= limit:
            break
        if done is not None:
            done.add(fid)
        if not await r.exists(key_fmt.format(fid)):
            out.add(fid)
    return out
