"""
Tournament prediction background worker.

Periodically re-runs the tournament Monte Carlo simulation
(api/routes/predict.py). Each run pins every played result and conditions
live matches on their current state.
"""

import asyncio
import logging

import redis.asyncio as aioredis

from monitoring.metrics import WORKER_ERRORS, WORKER_TICK_DURATION, tick_done

log = logging.getLogger(__name__)

INTERVAL = 1800.0  # 30 min — a 50k-sim run is ~1.3s, cheap to repeat often


async def run(redis_client: aioredis.Redis) -> None:
    log.info("Prediction worker started — re-simulating tournament every 30 min")
    while True:
        try:
            with WORKER_TICK_DURATION.labels("prediction").time():
                await _maybe_resimulate(redis_client)
            tick_done("prediction")
        except asyncio.CancelledError:
            log.info("Prediction worker cancelled")
            raise
        except Exception as exc:
            WORKER_ERRORS.labels("prediction").inc()
            log.error(f"Prediction worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _maybe_resimulate(r: aioredis.Redis) -> None:
    # Lazy import keeps the worker -> route dependency one-directional.
    from api.routes.predict import _run_and_store, claim_sim

    sim_id = await claim_sim(r)
    if sim_id is None:
        log.info("Prediction worker: sim already running, skipping this tick")
        return
    log.info(f"Prediction worker: re-simulating (sim_id={sim_id})")
    await _run_and_store(r, sim_id, 50_000)
