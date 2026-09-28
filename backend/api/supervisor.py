"""
Leader election and worker supervision.

Only the instance holding the Redis leader lock runs the producer and the
workers; other instances serve HTTP from Redis. If the leader dies, its lock
expires after LEADER_TTL_S and another instance takes over.

Each worker runs under supervise(): if it exits or raises it is restarted
with exponential backoff. Workers that need match data wait for it
indefinitely instead of giving up after a timeout. health() reports
per-worker liveness from monitoring.metrics.LAST_TICK.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import redis.asyncio as aioredis

from monitoring.metrics import LAST_TICK, WORKER_RESTARTS

log = logging.getLogger(__name__)

LEADER_KEY = "wc2026:leader"
LEADER_TTL_S = float(os.getenv("LEADER_TTL_S", "15"))
RENEW_EVERY_S = LEADER_TTL_S / 3
MIN_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
HEALTHY_RUN_S = 300.0  # a run this long resets the backoff
STALE_GRACE_S = 300.0  # slack on top of 3 intervals before a worker is stale

WorkerFn = Callable[[aioredis.Redis], Awaitable[None]]

# Compare-and-set on the lock value, so an instance can only extend or
# release a lock it still owns.
_RENEW = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
)
_RELEASE = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)


@dataclass
class WorkerStatus:
    interval_s: float
    phase: str = "starting"  # starting | waiting_for_data | running | restarting
    started_at: Optional[float] = None
    restarts: int = 0
    last_error: Optional[str] = None


_workers: dict[str, WorkerStatus] = {}
_role = {"role": "follower", "since": time.monotonic()}


# ── Leader lock ────────────────────────────────────────────────────────────


class LeaderLock:
    def __init__(self, r: aioredis.Redis, key: str = LEADER_KEY, ttl_s: float = LEADER_TTL_S):
        self.r = r
        self.key = key
        self.ttl_ms = int(ttl_s * 1000)
        self.token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    async def acquire(self) -> bool:
        return bool(await self.r.set(self.key, self.token, nx=True, px=self.ttl_ms))

    async def renew(self) -> bool:
        return bool(await self.r.eval(_RENEW, 1, self.key, self.token, self.ttl_ms))

    async def release(self) -> None:
        await self.r.eval(_RELEASE, 1, self.key, self.token)


async def _cancel(tasks: list[asyncio.Task]) -> None:
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_as_leader(
    r: aioredis.Redis,
    spawn: Callable[[], list[asyncio.Task]],
    lock: Optional[LeaderLock] = None,
    renew_every_s: float = RENEW_EVERY_S,
) -> None:
    """Run spawn()'s tasks while holding the leader lock.

    Losing the lock, or failing to renew it for a full TTL, cancels them and
    this instance retries as a follower.
    """
    lock = lock or LeaderLock(r)
    ttl_s = lock.ttl_ms / 1000
    tasks: list[asyncio.Task] = []
    renewed_at = 0.0
    leading = False
    try:
        while True:
            try:
                if not tasks:
                    if await lock.acquire():
                        log.info(f"Leader lock acquired ({lock.token}) — starting workers")
                        _role.update(role="leader", since=time.monotonic())
                        renewed_at = time.monotonic()
                        leading = True
                        tasks = spawn()
                elif await lock.renew():
                    renewed_at = time.monotonic()
                else:
                    log.warning("Leader lock lost — stopping workers")
                    await _cancel(tasks)
                    tasks = []
                    leading = False
                    _role.update(role="follower", since=time.monotonic())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(f"Leader lock check failed: {exc}")
                if tasks and time.monotonic() - renewed_at > ttl_s:
                    # Another instance may already hold the lock.
                    log.warning("Leader lock unconfirmed for a full TTL — stopping workers")
                    await _cancel(tasks)
                    tasks = []
                    leading = False
                    _role.update(role="follower", since=time.monotonic())
            await asyncio.sleep(renew_every_s)
    finally:
        await _cancel(tasks)
        if leading:
            # If this fails the lock still expires after one TTL.
            try:
                await lock.release()
            except Exception as exc:
                log.warning(f"Leader lock release failed: {exc!r}")
            _role.update(role="follower", since=time.monotonic())


# ── Supervision ────────────────────────────────────────────────────────────


async def wait_for_data(r: aioredis.Redis, name: str, warn_every_s: float = 180.0) -> None:
    """Block until the producer has written at least one fixture."""
    start = last_warn = time.monotonic()
    delay = 2.0
    while not await r.smembers("matches:active"):
        if time.monotonic() - last_warn > warn_every_s:
            last_warn = time.monotonic()
            log.warning(f"{name}: still waiting for match data ({last_warn - start:.0f}s)")
        await asyncio.sleep(delay)
        delay = min(delay * 1.3, 10.0)


async def supervise(
    name: str,
    fn: WorkerFn,
    r: aioredis.Redis,
    interval_s: float,
    *,
    needs_data: bool = False,
    min_backoff_s: float = MIN_BACKOFF_S,
    max_backoff_s: float = MAX_BACKOFF_S,
) -> None:
    """Run fn(r) forever with exponential-backoff restarts; cancellation propagates."""
    st = _workers[name] = WorkerStatus(interval_s=interval_s)
    backoff = min_backoff_s
    while True:
        st.started_at = time.monotonic()
        try:
            if needs_data:
                st.phase = "waiting_for_data"
                await wait_for_data(r, name)
            st.phase = "running"
            await fn(r)
            st.last_error = "returned"
            log.error(f"Worker {name} returned unexpectedly")
        except asyncio.CancelledError:
            st.phase = "stopped"
            raise
        except Exception as exc:
            st.last_error = f"{type(exc).__name__}: {exc}"[:200]
            log.error(f"Worker {name} crashed: {exc}", exc_info=True)
        if time.monotonic() - st.started_at > HEALTHY_RUN_S:
            backoff = min_backoff_s
        st.phase = "restarting"
        st.restarts += 1
        WORKER_RESTARTS.labels(name).inc()
        log.warning(f"Restarting {name} in {backoff:.0f}s (restart #{st.restarts})")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, max_backoff_s)


def health() -> dict:
    """Per-worker liveness.

    Stale when the last completed tick is older than 3 intervals plus
    STALE_GRACE_S.
    """
    now = time.monotonic()
    workers = {}
    ok = True
    for name, st in _workers.items():
        last = LAST_TICK.get(name)
        ref = last if last is not None else st.started_at
        age = None if ref is None else now - ref
        stale = (
            st.phase == "running"
            and age is not None
            and age > 3 * st.interval_s + STALE_GRACE_S
        )
        alive = st.phase in ("running", "waiting_for_data") and not stale
        ok = ok and alive
        workers[name] = {
            "phase": st.phase,
            "alive": alive,
            "last_tick_age_s": None if last is None else round(now - last, 1),
            "restarts": st.restarts,
            "last_error": st.last_error,
        }
    role = _role["role"]
    if role == "leader" and not workers:
        ok = False
    return {
        "role": role,
        "role_age_s": round(now - _role["since"], 1),
        "status": "ok" if ok else "degraded",
        "workers": workers if role == "leader" else {},
    }


def reset() -> None:
    """Test helper."""
    _workers.clear()
    _role.update(role="follower", since=time.monotonic())
