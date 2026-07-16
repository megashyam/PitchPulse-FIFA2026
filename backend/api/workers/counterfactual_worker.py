"""
Counterfactual worker.

Background process that monitors match events and generates counterfactual
analysis when meaningful changes occur (goals, cards, or other triggers).

The worker uses persisted Redis match state and event history instead of
temporary in-memory state, ensuring deterministic replay after restarts and
preventing duplicate simulations of the same event.

Completed fixtures are processed less frequently because their match outcome
is final and only limited historical probability updates may still be needed.
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis

from agents import counterfactual_agent
from api.schemas.event_types import COMPLETED_STATUSES, TRIGGER_TYPES
from api.schemas.schema import MatchState

log = logging.getLogger(__name__)
INTERVAL = 30.0

PROCESSABLE = {"1H", "HT", "2H", "ET", "P", "FT", "AET", "PEN"}
TTL_LIVE = 3_600
TTL_COMPLETED = 2_592_000
FEED_MAX = 19

_prev_elapsed: dict[str, int] = {}
_seeded: set[str] = set()


def _ttl_for(status_short: str) -> int:
    return TTL_COMPLETED if status_short in COMPLETED_STATUSES else TTL_LIVE


async def run(redis_client: aioredis.Redis) -> None:
    """Continuously update counterfactual feeds for active fixtures."""
    log.info("Counterfactual worker started — every 30s (restart-safe coverage)")
    loop = asyncio.get_running_loop()
    while True:
        try:
            await _update_all(redis_client, loop)
        except asyncio.CancelledError:
            log.info("Counterfactual worker cancelled")
            return
        except Exception as exc:
            log.error(f"CF worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _update_all(r: aioredis.Redis, loop) -> None:
    active = set(await r.smembers("matches:active"))
    completed_ids = await r.smembers("matches:completed")
    fixture_ids = await r.smembers("matches:active")

    active = set(fixture_ids)
    for stale in [fid for fid in list(_prev_elapsed) if fid not in active]:
        _prev_elapsed.pop(stale, None)
        _seeded.discard(stale)
        try:
            counterfactual_agent.clear_state(int(stale))
        except (TypeError, ValueError):
            pass

    fixtures_to_process = set(active)
    for cid in completed_ids:
        if not await r.exists(f"match:{cid}:counterfactual:feed"):
            fixtures_to_process.add(cid)

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
            if team_id is None:
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
    """Update one fixture's counterfactual feed if it is still processable."""
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

    if completed and not any(ev.type in TRIGGER_TYPES for ev in state.events):
        return

    await _seed_coverage_from_feed(r, fid, state)

    current_elapsed = state.elapsed or 0
    prev_elapsed = _prev_elapsed.get(fid, current_elapsed)

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

        results = await counterfactual_agent.update_all(state, loop)
        if not results:
            return

        pipe = r.pipeline(transaction=True)
        for res in results:
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

    result = await counterfactual_agent.update(state, loop)
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
