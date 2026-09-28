"""
Tactical fingerprint background worker.

Every 120 seconds, refreshes the tactical fingerprint match for every
fixture in matches:active, so matches appear as soon as the TacticalProfiles
index is populated. Fingerprints drift slowly, hence the longer interval.

Redis writes:
    match:{fixture_id}:tactical   JSON fingerprint match result   TTL 600s
"""

import asyncio
import logging

import redis.asyncio as aioredis

from api.routes.tactical import compute_and_cache
from api.schemas.event_types import COMPLETED_STATUSES
from api.schemas.schema import MatchState
from monitoring.metrics import WORKER_ERRORS, WORKER_TICK_DURATION, tick_done

log = logging.getLogger(__name__)
INTERVAL = 120.0


async def run(redis_client: aioredis.Redis) -> None:
    log.info("Tactical worker started — refreshing every 120s")
    loop = asyncio.get_running_loop()
    while True:
        try:
            with WORKER_TICK_DURATION.labels("tactical").time():
                await _update_all(redis_client, loop)
            tick_done("tactical")
        except asyncio.CancelledError:
            log.info("Tactical worker cancelled")
            raise
        except Exception as exc:
            WORKER_ERRORS.labels("tactical").inc()
            log.error(f"Tactical worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _update_all(r: aioredis.Redis, loop) -> None:
    """Process matches:active, plus completed fixtures with no cached fingerprint."""
    active_ids = set(await r.smembers("matches:active"))
    completed_ids = await r.smembers("matches:completed")

    fixtures_to_process = set(active_ids)
    for cid in completed_ids:
        if not await r.exists(f"match:{cid}:tactical"):
            fixtures_to_process.add(cid)

    fixture_ids = list(fixtures_to_process)
    if not fixture_ids:
        return

    results = await asyncio.gather(
        *[_update_fixture(r, fid, loop) for fid in fixture_ids],
        return_exceptions=True,
    )
    for fid, res in zip(fixture_ids, results):
        if isinstance(res, Exception):
            log.error(f"[{fid}] tactical update raised: {res}", exc_info=res)


async def _update_fixture(r: aioredis.Redis, fid: str, loop) -> None:
    raw = await r.get(f"match:{fid}:state")
    if not raw:
        return

    try:
        state = MatchState.model_validate_json(raw)
    except Exception as exc:
        log.warning(f"[{fid}] tactical MatchState parse error: {exc}")
        return

    # NS matches have no live possession/pass data yet to build a descriptor
    # from — skip until kickoff.
    if state.status_short == "NS":
        return
    # Completed fixtures are computed once and cached for 30 days.
    if state.status_short in COMPLETED_STATUSES and await r.exists(f"match:{fid}:tactical"):
        return

    result = await compute_and_cache(r, fid, state, loop)
    if result.get("status") == "written":
        log.debug(f"[{fid}] tactical fingerprint refreshed")
