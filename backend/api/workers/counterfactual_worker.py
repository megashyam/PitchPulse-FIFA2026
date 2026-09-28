"""
Counterfactual background worker.

Runs the counterfactual agent on active fixtures and backfills completed
ones (a capped number per tick).

Design:
    - Coverage is rebuilt from the persisted Redis feed before a fixture is
      first processed, so a restart never re-simulates analysed events.
    - Completed matches (FT/AET/PEN) backfill every trigger type; events
      that can't move the bracket get a template entry.
    - Per-fixture memory is released when a fixture leaves matches:active.
    - Feed rewrites are a MULTI/EXEC transaction.
    - Replay restarts are detected and reset agent state.
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis

from agents import counterfactual_agent
from agents.llm_queue import Priority, llm_priority
from api.tournament_state import results_before
from api.schemas.event_types import COMPLETED_STATUSES, TRIGGER_TYPES
from api.schemas.schema import MatchState
from monitoring.metrics import WORKER_ERRORS, WORKER_TICK_DURATION, tick_done

log = logging.getLogger(__name__)
INTERVAL = 30.0

PROCESSABLE = {"1H", "HT", "2H", "ET", "P", "FT", "AET", "PEN"}
TTL_LIVE = 3_600  # 1 hour — match in progress
TTL_COMPLETED = 2_592_000  # 30 days — match history
FEED_MAX = 19  # up to 20 entries (0-indexed ltrim)
BACKFILL_PER_TICK = 1  # completed fixtures without a feed, per tick

_prev_elapsed: dict[str, int] = {}
_seeded: set[str] = set()  # fixtures whose coverage was restored this process
_backfilled: set[str] = set()  # completed fixtures already attempted this process


def _ttl_for(status_short: str) -> int:
    return TTL_COMPLETED if status_short in COMPLETED_STATUSES else TTL_LIVE


async def run(redis_client: aioredis.Redis) -> None:
    log.info("Counterfactual worker started — every 30s (restart-safe coverage)")
    loop = asyncio.get_running_loop()
    while True:
        try:
            with WORKER_TICK_DURATION.labels("counterfactual").time():
                await _update_all(redis_client, loop)
            tick_done("counterfactual")
        except asyncio.CancelledError:
            log.info("Counterfactual worker cancelled")
            raise
        except Exception as exc:
            WORKER_ERRORS.labels("counterfactual").inc()
            log.error(f"CF worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _update_all(r: aioredis.Redis, loop) -> None:
    active = set(await r.smembers("matches:active"))
    completed_ids = await r.smembers("matches:completed")

    # Release memory for fixtures that left the active set.
    for stale in [fid for fid in list(_prev_elapsed) if fid not in active]:
        _prev_elapsed.pop(stale, None)
        _seeded.discard(stale)
        try:
            counterfactual_agent.clear_state(int(stale))
        except (TypeError, ValueError):
            pass

    # Backfill completed fixtures this worker never saw while active
    # (same as intel_worker).
    fixtures_to_process = set(active)
    backfill = 0
    # A match with no trigger events never gets a feed; _backfilled keeps it
    # from taking the slot every tick.
    for cid in sorted(set(completed_ids) - active - _backfilled):
        if backfill >= BACKFILL_PER_TICK:
            break
        _backfilled.add(cid)
        if not await r.exists(f"match:{cid}:counterfactual:feed"):
            fixtures_to_process.add(cid)
            backfill += 1

    if not fixtures_to_process:
        return

    for fid in fixtures_to_process:
        try:
            await _update_fixture(r, fid, loop)
        except Exception as exc:
            log.error(f"[{fid}] CF update error: {exc}", exc_info=True)


async def _seed_coverage_from_feed(
    r: aioredis.Redis, fid: str, state: MatchState
) -> None:
    """Rebuild the agent's covered set from the persisted feed (restart-safe)."""
    if fid in _seeded:
        return
    _seeded.add(fid)

    raw_entries = await r.lrange(f"match:{fid}:counterfactual:feed", 0, FEED_MAX)
    if not raw_entries:
        return

    sigs: set[str] = set()
    for raw in raw_entries:
        try:
            e = json.loads(raw)
            team_id = e.get("event_team_id")
            if team_id is None:  # entries without event_team_id
                team_id = 1 if e.get("event_team") == state.home_name else 2
            sigs.add(
                counterfactual_agent.event_sig(
                    e["minute"], e.get("extra"), e["event_type"], team_id
                )
            )
        except Exception:
            continue

    if sigs:
        counterfactual_agent.seed_covered(state.fixture_id, sigs)
        log.info(f"[{fid}] CF coverage restored from feed: {len(sigs)} event(s)")


async def _update_fixture(r: aioredis.Redis, fid: str, loop) -> None:
    state_raw = await r.get(f"match:{fid}:state")
    if not state_raw:
        return

    try:
        state = MatchState.model_validate_json(state_raw)
    except Exception as exc:
        log.warning(f"[{fid}] CF MatchState parse error: {exc}")
        return

    if state.status_short not in PROCESSABLE:
        return

    completed = state.status_short in COMPLETED_STATUSES

    # Completed matches with nothing sim-worthy: don't even wake the agent.
    if completed and not any(ev.type in TRIGGER_TYPES for ev in state.events):
        return

    await _seed_coverage_from_feed(r, fid, state)

    current_elapsed = state.elapsed or 0
    prev_elapsed = _prev_elapsed.get(fid, current_elapsed)

    # Replay restart detection (a no-op for live data, where elapsed only
    # increases).
    if prev_elapsed - current_elapsed > 10:
        log.info(
            f"[{fid}] Replay restart — elapsed dropped from {prev_elapsed}' "
            f"to {current_elapsed}'. Clearing stale CF keys."
        )
        await r.delete(f"match:{fid}:counterfactual:latest")
        await r.delete(f"match:{fid}:counterfactual:feed")
        counterfactual_agent.clear_state(state.fixture_id)
        _seeded.discard(fid)

    _prev_elapsed[fid] = current_elapsed
    ttl = _ttl_for(state.status_short)

    if completed:
        # update() only ever surfaces ONE (the most-recently-added,
        # regardless of minute) uncovered trigger per call and is
        # MIN_GAP-throttled — for a completed match that means it could get
        # stuck offering a low-priority yellow-card event forever while the
        # real goals never get their turn, and even in the best case only
        # one event backfills per ~30s tick before the fixture ages out of
        # matches:active. update_all() exists precisely for this: analyse
        # every uncovered trigger for the match in a single pass.
        pinned = await results_before(r, state.kickoff_time)
        with llm_priority(Priority.BACKGROUND):
            results = await counterfactual_agent.update_all(state, loop, results=pinned)
        if not results:
            return

        pipe = r.pipeline(transaction=True)
        for res in results:  # oldest-first in; each lpush -> newest-first out
            pipe.lpush(f"match:{fid}:counterfactual:feed", json.dumps(res))
        pipe.ltrim(f"match:{fid}:counterfactual:feed", 0, FEED_MAX)
        pipe.expire(f"match:{fid}:counterfactual:feed", ttl)
        latest = max(results, key=lambda x: x.get("minute", 0))
        pipe.setex(f"match:{fid}:counterfactual:latest", ttl, json.dumps(latest))
        await pipe.execute()

        for res in results:
            await r.publish(
                "counterfactual_update",
                json.dumps(
                    {
                        "fixture_id": res["fixture_id"],
                        "minute": res["minute"],
                        "event_type": res["event_type"],
                        "path_shift_pct": res["path_shift_pct"],
                    }
                ),
            )
        return

    pinned = await results_before(r, state.kickoff_time)
    with llm_priority(Priority.LIVE):
        result = await counterfactual_agent.update(state, loop, results=pinned)
    if result is None:
        return

    result_json = json.dumps(result)

    pipe = r.pipeline(transaction=True)
    pipe.setex(f"match:{fid}:counterfactual:latest", ttl, result_json)
    pipe.lpush(f"match:{fid}:counterfactual:feed", result_json)
    pipe.ltrim(f"match:{fid}:counterfactual:feed", 0, FEED_MAX)
    pipe.expire(f"match:{fid}:counterfactual:feed", ttl)
    await pipe.execute()

    await r.publish(
        "counterfactual_update",
        json.dumps(
            {
                "fixture_id": result["fixture_id"],
                "minute": result["minute"],
                "event_type": result["event_type"],
                "path_shift_pct": result["path_shift_pct"],
            }
        ),
    )
