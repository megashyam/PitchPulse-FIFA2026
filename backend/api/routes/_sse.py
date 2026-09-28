"""
Shared pub/sub-backed SSE generator.

Workers publish change signals on "momentum_update", "intel_update",
"counterfactual_update", "narrative_spike" and "prediction_update".
PubSubHub holds one Redis pub/sub connection per process and fans each
message out to per-client queues; a channel stays subscribed while at least
one client listens.

The backing key is re-read only when a message arrives, with a slow poll
(`fallback_poll_s`) as a safety net for missed messages.
"""

import asyncio
import contextlib
import json
import logging
from typing import AsyncIterator, Awaitable, Callable, Optional

import redis.asyncio as aioredis

from monitoring.metrics import SSE_CONNECTIONS

log = logging.getLogger(__name__)

QUEUE_MAX = 64  # per client; messages are change signals, so dropping old ones is safe


class PubSubHub:
    def __init__(self, r: aioredis.Redis) -> None:
        self.r = r
        self.subs: dict[str, set[asyncio.Queue]] = {}
        self.pubsub = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        self.closing = False

    async def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        async with self.lock:
            if self.pubsub is None:
                self.pubsub = self.r.pubsub()
            if channel not in self.subs:
                self.subs[channel] = set()
                await self.pubsub.subscribe(channel)
            self.subs[channel].add(q)
            if self.task is None or self.task.done():
                self.task = asyncio.create_task(self._read(), name="sse_pubsub_hub")
        return q

    async def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        async with self.lock:
            qs = self.subs.get(channel)
            if qs is None:
                return
            qs.discard(q)
            if not qs:
                del self.subs[channel]
                try:
                    await self.pubsub.unsubscribe(channel)
                except Exception as exc:
                    log.debug(f"hub unsubscribe {channel} failed: {exc}")

    def _dispatch(self, channel: str, data: str) -> None:
        for q in list(self.subs.get(channel, ())):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(data)

    async def _read(self) -> None:
        # Loop on a flag, not only on cancellation: redis-py's get_message
        # can swallow a CancelledError delivered mid-read.
        while not self.closing:
            if not self.subs:
                await asyncio.sleep(0.5)
                continue
            try:
                msg = await self.pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # redis-py reconnects and resubscribes on the next read.
                log.warning(f"SSE hub read error: {exc}")
                await asyncio.sleep(1.0)
                continue
            if msg is not None and msg.get("type") == "message":
                self._dispatch(msg["channel"], msg["data"])

    async def close(self) -> None:
        self.closing = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.wait({self.task}, timeout=2.0)
        if self.pubsub is not None:
            with contextlib.suppress(Exception):
                await self.pubsub.aclose()
        self.subs.clear()


_hub: Optional[PubSubHub] = None


def get_hub(r: aioredis.Redis) -> PubSubHub:
    global _hub
    if _hub is None or _hub.r is not r:
        _hub = PubSubHub(r)
    return _hub


async def close_hub() -> None:
    global _hub
    if _hub is not None:
        await _hub.close()
        _hub = None


@contextlib.asynccontextmanager
async def subscription(r: aioredis.Redis, channel: str) -> AsyncIterator[asyncio.Queue]:
    hub = get_hub(r)
    q = await hub.subscribe(channel)
    SSE_CONNECTIONS.labels(channel).inc()
    try:
        yield q
    finally:
        SSE_CONNECTIONS.labels(channel).dec()
        await hub.unsubscribe(channel, q)


async def next_message(q: asyncio.Queue, timeout: float) -> Optional[str]:
    try:
        return await asyncio.wait_for(q.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


async def pubsub_sse(
    *,
    redis_client: aioredis.Redis,
    channel: str,
    key: str,
    event_name: str,
    is_disconnected: Callable[[], Awaitable[bool]],
    match_fixture_id: Optional[str] = None,
    emit_ok: Optional[Callable[[str], Awaitable[bool]]] = None,
    fallback_poll_s: float = 10.0,
    ping_every: int = 15,
):
    """Yield sse_starlette event dicts for `key`, driven by `channel`.

    Sends the current value immediately, then re-reads `key` on each message
    and emits when it changed. With `match_fixture_id`, messages for other
    fixtures are skipped without a Redis read. `emit_ok(raw)` can veto an
    emission. Polls every `fallback_poll_s` in case a message is dropped.
    """
    last_raw: Optional[str] = None
    tick = 0

    async with subscription(redis_client, channel) as q:
        raw = await redis_client.get(key)
        if raw is not None:
            if emit_ok is None or await emit_ok(raw):
                last_raw = raw
                yield {"event": event_name, "data": raw}
        else:
            yield {
                "event": "waiting",
                "data": json.dumps({"message": f"waiting for {event_name} data"}),
            }

        while True:
            if await is_disconnected():
                break

            msg = await next_message(q, fallback_poll_s)

            if msg is not None and match_fixture_id is not None:
                try:
                    payload = json.loads(msg)
                    if str(payload.get("fixture_id")) != str(match_fixture_id):
                        continue
                except Exception:
                    pass  # malformed payload — fall through to a real check

            raw = await redis_client.get(key)
            if raw is not None and raw != last_raw:
                if emit_ok is None or await emit_ok(raw):
                    last_raw = raw
                    yield {"event": event_name, "data": raw}

            tick += 1
            if tick % max(1, int(ping_every // max(fallback_poll_s, 1))) == 0:
                yield {"event": "heartbeat", "data": "{}"}
