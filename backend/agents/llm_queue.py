"""
Priority admission for LLM calls.

Every Ollama and Groq request waits on a PriorityGate: a semaphore that
admits waiters by (priority, arrival order), so a live goal's narration
jumps ahead of queued backfill work. Running calls are never pre-empted.

Priority is carried in a context variable, set once around a block or per
asyncio task:

    with llm_priority(Priority.BACKGROUND):
        await counterfactual_agent.update_all(...)

Tasks created inside the block inherit it; set_llm_priority() sets it for
the rest of the current task.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import time
from contextvars import ContextVar
from enum import IntEnum
from typing import Iterator, Optional

from monitoring.metrics import LLM_QUEUE_DEPTH, LLM_WAIT


class Priority(IntEnum):
    LIVE = 0  # narration for a match in progress
    USER = 1  # on-demand requests, briefings, fresh narrative spikes
    BACKGROUND = 2  # completed-match backfill, trending arcs


_priority: ContextVar[Priority] = ContextVar("llm_priority", default=Priority.USER)


def current_priority() -> Priority:
    return _priority.get()


def set_llm_priority(p: Priority) -> None:
    _priority.set(p)


@contextlib.contextmanager
def llm_priority(p: Priority) -> Iterator[None]:
    token = _priority.set(p)
    try:
        yield
    finally:
        _priority.reset(token)


class PriorityGate:
    def __init__(self, name: str, capacity: int) -> None:
        self.name = name
        self.capacity = max(1, capacity)
        self.active = 0
        self._heap: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()

    @property
    def waiting(self) -> int:
        return sum(1 for *_, f in self._heap if not f.done())

    @contextlib.asynccontextmanager
    async def slot(self, priority: Optional[Priority] = None):
        p = current_priority() if priority is None else priority
        label = p.name.lower()
        t0 = time.monotonic()
        if self.active < self.capacity and not self.waiting:
            self.active += 1
        else:
            fut = asyncio.get_running_loop().create_future()
            heapq.heappush(self._heap, (int(p), next(self._seq), fut))
            LLM_QUEUE_DEPTH.labels(self.name, label).inc()
            try:
                await fut
            except asyncio.CancelledError:
                # Granted a slot but cancelled before resuming: pass it on.
                if fut.done() and not fut.cancelled():
                    self._release()
                raise
            finally:
                LLM_QUEUE_DEPTH.labels(self.name, label).dec()
        LLM_WAIT.labels(self.name, label).observe(time.monotonic() - t0)
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        # Hand the slot straight to the best live waiter; `active` is unchanged.
        while self._heap:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                fut.set_result(None)
                return
        self.active -= 1


class Pacer:
    """Minimum spacing between request starts (Groq's per-minute limit)."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._next = 0.0

    async def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = time.monotonic()
        start = max(now, self._next)
        self._next = start + self.min_interval_s
        if start > now:
            await asyncio.sleep(start - now)
