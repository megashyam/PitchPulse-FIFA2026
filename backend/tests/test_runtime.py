"""
Runtime tests.

Covers the LLM priority gate, worker supervision, leader election, the
predict sim lock and the shared SSE pub/sub hub. Uses fakeredis; no services
needed.
"""

import asyncio
import time

import fakeredis
import pytest

from agents.llm_queue import Pacer, Priority, PriorityGate, llm_priority
from api import supervisor
from api.routes import _sse
from api.routes.predict import LOCK_KEY, claim_sim
from monitoring import metrics


def _run(coro):
    return asyncio.run(coro)


# ── Priority gate ──────────────────────────────────────────────────────────


def test_gate_admits_by_priority_then_arrival():
    async def main():
        gate = PriorityGate("test", 1)
        order = []
        hold = asyncio.Event()

        async def call(tag, p):
            async with gate.slot(p):
                order.append(tag)
                if tag == "first":
                    await hold.wait()

        first = asyncio.create_task(call("first", Priority.BACKGROUND))
        await asyncio.sleep(0)
        waiters = [
            asyncio.create_task(call(tag, p))
            for tag, p in [
                ("bg1", Priority.BACKGROUND),
                ("bg2", Priority.BACKGROUND),
                ("live", Priority.LIVE),
                ("user", Priority.USER),
            ]
        ]
        await asyncio.sleep(0)
        assert gate.waiting == 4
        hold.set()
        await asyncio.gather(first, *waiters)
        return order, gate.active

    order, active = _run(main())
    assert order == ["first", "live", "user", "bg1", "bg2"]
    assert active == 0


def test_gate_cancelled_waiter_does_not_leak_slot():
    async def main():
        gate = PriorityGate("test", 1)
        hold = asyncio.Event()

        async def holder():
            async with gate.slot(Priority.USER):
                await hold.wait()

        async def waiter():
            async with gate.slot(Priority.LIVE):
                pass

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        w = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        w.cancel()
        await asyncio.gather(w, return_exceptions=True)
        hold.set()
        await h
        async with gate.slot(Priority.USER):  # must not deadlock
            pass
        return gate.active

    assert _run(main()) == 0


def test_gate_uses_context_priority():
    async def main():
        gate = PriorityGate("test", 1)
        seen = []
        hold = asyncio.Event()

        async def holder():
            async with gate.slot():
                await hold.wait()

        async def call(tag):
            async with gate.slot():
                seen.append(tag)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        with llm_priority(Priority.BACKGROUND):
            bg = asyncio.create_task(call("bg"))
        await asyncio.sleep(0)
        with llm_priority(Priority.LIVE):
            live = asyncio.create_task(call("live"))
        await asyncio.sleep(0)
        hold.set()
        await asyncio.gather(h, bg, live)
        return seen

    assert _run(main()) == ["live", "bg"]


def test_pacer_spaces_requests():
    async def main():
        p = Pacer(0.05)
        t0 = time.monotonic()
        for _ in range(3):
            await p.wait()
        return time.monotonic() - t0

    assert _run(main()) >= 0.09


# ── Supervision ────────────────────────────────────────────────────────────


def test_supervisor_restarts_crashed_worker():
    supervisor.reset()
    calls = {"n": 0}

    async def flaky(r):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("boom")
        while True:
            metrics.tick_done("flaky")
            await asyncio.sleep(0.01)

    async def main():
        t = asyncio.create_task(
            supervisor.supervise("flaky", flaky, None, 1.0, min_backoff_s=0.01)
        )
        await asyncio.sleep(0.2)
        supervisor._role["role"] = "leader"
        h = supervisor.health()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        return h

    h = _run(main())
    w = h["workers"]["flaky"]
    assert calls["n"] == 3
    assert w["restarts"] == 2
    assert w["alive"] and w["phase"] == "running"
    assert "boom" in w["last_error"]
    assert h["status"] == "ok"
    supervisor.reset()


def test_health_flags_stale_worker():
    supervisor.reset()
    supervisor._workers["slow"] = supervisor.WorkerStatus(interval_s=30.0, phase="running")
    metrics.LAST_TICK["slow"] = time.monotonic() - (90 + supervisor.STALE_GRACE_S + 5)
    supervisor._role["role"] = "leader"
    h = supervisor.health()
    assert h["status"] == "degraded"
    assert not h["workers"]["slow"]["alive"]
    supervisor.reset()


# ── Leader election ────────────────────────────────────────────────────────


def test_leader_lock_is_exclusive():
    async def main():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        a, b = supervisor.LeaderLock(r, ttl_s=5), supervisor.LeaderLock(r, ttl_s=5)
        got = (await a.acquire(), await b.acquire())
        renew = (await a.renew(), await b.renew())
        await b.release()  # not the owner: no-op
        still_a = await r.get(supervisor.LEADER_KEY) == a.token
        await a.release()
        return got, renew, still_a, await b.acquire()

    got, renew, still_a, b_after = _run(main())
    assert got == (True, False)
    assert renew == (True, False)
    assert still_a
    assert b_after


def test_only_one_instance_runs_workers_and_failover():
    supervisor.reset()

    async def main():
        server = fakeredis.FakeServer()
        spawned = {"a": 0, "b": 0}

        def spawner(tag):
            def spawn():
                spawned[tag] += 1
                return [asyncio.create_task(asyncio.sleep(3600))]
            return spawn

        def leader(tag):
            r = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
            lock = supervisor.LeaderLock(r, ttl_s=0.3)
            return asyncio.create_task(
                supervisor.run_as_leader(r, spawner(tag), lock=lock, renew_every_s=0.05)
            )

        a = leader("a")
        await asyncio.sleep(0.02)
        b = leader("b")
        await asyncio.sleep(0.3)
        before = dict(spawned)
        a.cancel()  # releases the lock, or it expires after one TTL
        await asyncio.gather(a, return_exceptions=True)
        await asyncio.sleep(0.6)
        b.cancel()
        await asyncio.gather(b, return_exceptions=True)
        return before, spawned

    before, after = _run(main())
    assert before == {"a": 1, "b": 0}
    assert after == {"a": 1, "b": 1}
    supervisor.reset()


# ── Predict sim lock ───────────────────────────────────────────────────────


def test_claim_sim_is_atomic():
    async def main():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        ids = await asyncio.gather(*[claim_sim(r) for _ in range(10)])
        return ids, await r.get(LOCK_KEY)

    ids, lock = _run(main())
    winners = [i for i in ids if i is not None]
    assert len(winners) == 1
    assert lock == winners[0]


# ── SSE hub ────────────────────────────────────────────────────────────────


def test_hub_fans_out_on_one_connection():
    async def main():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        async with _sse.subscription(r, "chan") as q1, _sse.subscription(r, "chan") as q2:
            hub = _sse.get_hub(r)
            pubsubs = {id(hub.pubsub)}
            await asyncio.sleep(0.05)
            await r.publish("chan", "hello")
            got = (await _sse.next_message(q1, 2.0), await _sse.next_message(q2, 2.0))
        remaining = dict(hub.subs)
        await _sse.close_hub()
        return got, pubsubs, remaining

    got, pubsubs, remaining = _run(main())
    assert got == ("hello", "hello")
    assert len(pubsubs) == 1
    assert remaining == {}


# ── Backfill throttling ────────────────────────────────────────────────────


def test_finished_match_leaves_active_set_by_kickoff():
    from datetime import datetime, timedelta, timezone

    from api.workers.match_producer import _worker_visible

    async def main():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        now = datetime.now(timezone.utc)
        return (
            await _worker_visible(r, 1, "FT", now - timedelta(days=3)),
            await _worker_visible(r, 2, "FT", now - timedelta(hours=2)),
            await _worker_visible(r, 3, "1H", now - timedelta(days=3)),
        )

    assert _run(main()) == (False, True, True)


def test_intel_backfill_is_throttled(monkeypatch):
    from api.workers import intel_worker

    seen = []

    async def fake_update(r, fid):
        seen.append(fid)

    monkeypatch.setattr(intel_worker, "_update_fixture", fake_update)
    monkeypatch.setattr(intel_worker, "_backfilled", set())

    async def main():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        await r.sadd("matches:completed", *[str(i) for i in range(10)])
        await r.sadd("matches:active", "99")
        ticks = []
        for _ in range(3):
            seen.clear()
            await intel_worker._update_all(r)
            ticks.append(sorted(seen))
        return ticks

    ticks = _run(main())
    per_tick = intel_worker.BACKFILL_PER_TICK
    assert all(len(t) == 1 + per_tick and "99" in t for t in ticks)
    backfilled = [f for t in ticks for f in t if f != "99"]
    assert len(set(backfilled)) == 3 * per_tick  # no fixture repeats


def test_topic_arc_fills_rows_below_top_n():
    import json
    from types import SimpleNamespace

    from fastapi import HTTPException

    from api.routes import narrative

    row = {
        "spike_id": "trend-abc", "topic": "Brazil", "tick": 1, "severity": 0.4,
        "sources": {"wikipedia": 1.0}, "source_names": [], "summary": "s",
        "timestamp": 0,
    }

    async def main():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        await r.set("narrative:trending:latest", json.dumps([row]))
        req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=r)))
        hit = await narrative.get_topic_arc("Brazil", req)
        try:
            await narrative.get_topic_arc("Peru", req)
            missing = None
        except HTTPException as e:
            missing = e.status_code
        return hit, missing

    hit, missing = _run(main())
    assert hit["topic"] == "Brazil" and hit["arc"] and not hit["cached"]
    assert missing == 404
