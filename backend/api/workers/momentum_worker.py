"""
Momentum background worker.

Every 30 seconds, runs the momentum model on every active MatchState,
writes the result and publishes a pub/sub notification for the SSE layer.
Per-fixture state is released when a fixture leaves matches:active.

Redis writes:
    match:{fixture_id}:momentum   JSON momentum snapshot   TTL 3600s

Pub/sub:
    channel momentum_update
    payload {fixture_id, home_momentum, away_momentum, goal probs, elapsed}
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis

from api.schemas.schema import MatchState
from ml import momentum_model
from monitoring.metrics import WORKER_ERRORS, WORKER_TICK_DURATION, tick_done

log = logging.getLogger(__name__)
INTERVAL = 30.0  # seconds between full update cycles


async def run(redis_client: aioredis.Redis) -> None:
    """Entry point. Called once from lifespan, runs for the process lifetime."""
    log.info("Momentum worker started — updating every 30s")
    while True:
        try:
            with WORKER_TICK_DURATION.labels("momentum").time():
                await _update_all(redis_client)
            tick_done("momentum")
        except asyncio.CancelledError:
            log.info("Momentum worker cancelled — shutting down")
            raise
        except Exception as exc:
            WORKER_ERRORS.labels("momentum").inc()
            log.error(f"Momentum worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _update_all(r: aioredis.Redis) -> None:
    """Process all fixtures currently in the matches:active set."""
    fixture_ids = list(await r.smembers("matches:active"))

    if not fixture_ids:
        return

    results = await asyncio.gather(
        *[_update_fixture(r, fid) for fid in fixture_ids],
        return_exceptions=True,
    )
    for fid, res in zip(fixture_ids, results):
        if isinstance(res, Exception):
            log.error(f"[{fid}] _update_fixture raised: {res}", exc_info=res)


async def _update_fixture(r: aioredis.Redis, fid_str: str) -> None:
    """Read MatchState → momentum_model.update() → write and publish."""
    raw = await r.get(f"match:{fid_str}:state")
    if not raw:
        return

    try:
        state = MatchState.model_validate_json(raw)
    except Exception as exc:
        log.warning(f"[{fid_str}] MatchState parse error: {exc}")
        return

    shots_raw = await r.get(f"match:{fid_str}:shots")
    result = momentum_model.update(state, json.loads(shots_raw) if shots_raw else [])
    if result is None:
        return

    result_json = json.dumps(result)
    await r.setex(f"match:{fid_str}:momentum", 3600, result_json)

    notif = json.dumps(
        {
            "fixture_id": result["fixture_id"],
            "home_momentum": result["home"]["momentum_score"],
            "away_momentum": result["away"]["momentum_score"],
            "home_goal_prob": result["home"]["goal_prob_5min"],
            "away_goal_prob": result["away"]["goal_prob_5min"],
            "elapsed": result["elapsed"],
        }
    )
    await r.publish("momentum_update", notif)

    log.debug(
        f"[{fid_str}] {result['home_name'][:10]:10s} "
        f"M={result['home']['momentum_score']:.2f} "
        f"G={result['home']['goal_prob_5min']:.3f} | "
        f"{result['away_name'][:10]:10s} "
        f"M={result['away']['momentum_score']:.2f} "
        f"G={result['away']['goal_prob_5min']:.3f}"
    )
