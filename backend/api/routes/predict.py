"""
Tournament prediction endpoints.

    POST /predict/simulate     start a Monte Carlo simulation (non-blocking)
    GET  /predict/status       status while a sim runs
    GET  /predict/tournament   latest tournament prediction from Redis
    GET  /predict/team/{name}  single team prediction

Design:
    - A run is claimed with SET NX on predict:sim:lock; predict:sim:status
      is informational only.
    - /simulate is token-gated like the other trigger routes.
    - Every read goes to Redis (no in-process cache), so any instance serves
      the latest result.
    - Simulations run on ml.executors.SIM_EXECUTOR.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from api.routes._security import require_trigger_token
from api.routes._sse import pubsub_sse
from api.schemas.predict import (
    SimStatus,
    SimTriggerResponse,
    TeamPrediction,
    TournamentPrediction,
)
from ml.executors import SIM_EXECUTOR
from ml.odds_api_client import get_oddsapi_client
from api.tournament_state import sim_conditions
from ml.tournament_sim import SimResult, run_simulation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])

REDIS_KEY = "predict:tournament:latest"
REDIS_TTL = 86_400  # 24 hours
STATUS_KEY = "predict:sim:status"
STATUS_TTL = 900
LOCK_KEY = "predict:sim:lock"
LOCK_TTL = 900  # sims finish well within 15 min; a crashed run self-heals

_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """create_task that keeps a reference until the task finishes."""
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


async def _get_status(r) -> SimStatus:
    raw = await r.get(STATUS_KEY)
    if not raw:
        return SimStatus(status="idle")
    return SimStatus(**json.loads(raw))


async def _set_status(r, status: SimStatus) -> None:
    await r.setex(STATUS_KEY, STATUS_TTL, status.model_dump_json())


async def claim_sim(r) -> str | None:
    """Claim the sim slot; the new sim_id, or None if a sim is running."""
    sim_id = str(uuid.uuid4())[:8]
    if not await r.set(LOCK_KEY, sim_id, nx=True, ex=LOCK_TTL):
        return None
    await _set_status(
        r,
        SimStatus(
            status="running", sim_id=sim_id, started_at=datetime.now(timezone.utc)
        ),
    )
    return sim_id


@router.post(
    "/simulate",
    response_model=SimTriggerResponse,
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_simulation(request: Request, n_sims: int = 50_000):
    """Start a tournament simulation; returns a sim_id for /predict/status."""
    r = request.app.state.redis
    sim_id = await claim_sim(r)
    if sim_id is None:
        raise HTTPException(status_code=409, detail="Simulation already running")

    n_sims = max(1_000, min(100_000, n_sims))
    logger.info(f"Simulation triggered: sim_id={sim_id}, n_sims={n_sims}")
    _spawn(_run_and_store(r, sim_id, n_sims))

    return SimTriggerResponse(
        accepted=True,
        message=f"Simulation started ({n_sims:,} runs). Poll /predict/status.",
        sim_id=sim_id,
    )


@router.get("/status", response_model=SimStatus)
async def get_status(request: Request):
    """Current simulation status: idle | running | complete | error."""
    return await _get_status(request.app.state.redis)


@router.get("/stream")
async def predict_stream(request: Request):
    """SSE, pub/sub-backed, tournament-wide; fires when a new sim lands."""
    r = request.app.state.redis

    async def generator():
        async for event in pubsub_sse(
            redis_client=r,
            channel="prediction_update",
            key=REDIS_KEY,
            event_name="prediction_update",
            is_disconnected=request.is_disconnected,
        ):
            yield event

    return EventSourceResponse(generator(), ping=15)


@router.get("/tournament", response_model=TournamentPrediction)
async def get_tournament(request: Request):
    """Latest tournament prediction; starts a sim and returns 202 if none."""
    r = request.app.state.redis
    pred = await _latest(r)
    if pred is not None:
        return pred

    sim_id = await claim_sim(r)
    if sim_id is not None:
        _spawn(_run_and_store(r, sim_id, 50_000))
    return JSONResponse(
        status_code=202,
        content={
            "detail": "No simulation results yet. Simulation started — poll /predict/status."
        },
    )


async def _latest(r) -> Optional[TournamentPrediction]:
    raw = await r.get(REDIS_KEY)
    return TournamentPrediction(**json.loads(raw)) if raw else None


@router.get("/team/{name}", response_model=TeamPrediction)
async def get_team(name: str, request: Request):
    """Prediction for a single team by exact name."""
    pred = await _latest(request.app.state.redis)
    if pred is None:
        raise HTTPException(status_code=404, detail="No simulation results yet")
    team = next((t for t in pred.teams if t.name.lower() == name.lower()), None)
    if team is None:
        raise HTTPException(status_code=404, detail=f"Team '{name}' not found")
    return team


async def _run_and_store(redis, sim_id: str, n_sims: int) -> None:
    """Run the sim on SIM_EXECUTOR, store the result, release the lock."""
    try:
        odds_client = get_oddsapi_client()
        odds_table = await odds_client.get_all_odds()
        # Played matches pinned, live ones conditioned on their current state.
        results, live = await sim_conditions(redis)

        loop = asyncio.get_running_loop()
        result: SimResult = await loop.run_in_executor(
            SIM_EXECUTOR,
            lambda: run_simulation(
                odds_table=odds_table, n_sims=n_sims, results=results, live=live
            ),
        )

        pred = TournamentPrediction(
            sim_id=sim_id,
            n_sims=result.n_sims,
            elapsed_s=result.elapsed_s,
            run_at=datetime.now(timezone.utc),
            teams=[TeamPrediction.from_result(t) for t in result.teams],
            status="complete",
        )

        await redis.setex(REDIS_KEY, REDIS_TTL, pred.model_dump_json())
        await _set_status(redis, SimStatus(status="complete", sim_id=sim_id))
        logger.info(
            f"Simulation complete: sim_id={sim_id}, elapsed={result.elapsed_s}s"
        )

        await _push_sse_update(redis, sim_id, result.elapsed_s)

    except Exception as exc:
        logger.exception(f"Simulation failed: {exc}")
        await _set_status(
            redis, SimStatus(status="error", sim_id=sim_id, error=str(exc))
        )
    finally:
        # Only release our own claim.
        if await redis.get(LOCK_KEY) == sim_id:
            await redis.delete(LOCK_KEY)


async def _push_sse_update(redis, sim_id: str, elapsed_s: float) -> None:
    payload = json.dumps(
        {
            "type": "prediction_update",
            "sim_id": sim_id,
            "elapsed_s": elapsed_s,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    await redis.publish("prediction_update", payload)
