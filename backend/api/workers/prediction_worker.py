"""
=
Periodically re-triggers the tournament Monte Carlo simulation

"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis

from api.schemas.predict import SimStatus

log = logging.getLogger(__name__)

INTERVAL = 1800.0


async def run(redis_client: aioredis.Redis) -> None:
    log.info("Prediction worker started — re-simulating tournament every 30 min")
    while True:
        try:
            await _maybe_resimulate(redis_client)
        except asyncio.CancelledError:
            log.info("Prediction worker cancelled")
            return
        except Exception as exc:
            log.error(f"Prediction worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _maybe_resimulate(r: aioredis.Redis) -> None:

    from api.routes.predict import STATUS_KEY, STATUS_TTL, _run_and_store

    raw = await r.get(STATUS_KEY)
    if raw and json.loads(raw).get("status") == "running":
        log.info("Prediction worker: sim already running, skipping this tick")
        return

    sim_id = str(uuid.uuid4())[:8]
    await r.setex(
        STATUS_KEY,
        STATUS_TTL,
        SimStatus(
            status="running", sim_id=sim_id, started_at=datetime.now(timezone.utc)
        ).model_dump_json(),
    )
    log.info(f"Prediction worker: re-simulating (sim_id={sim_id})")
    await _run_and_store(r, sim_id, 50_000)
