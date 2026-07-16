"""
FastAPI application entry point for the hybrid match intelligence platform.

This module owns application lifecycle management, API initialization, shared
infrastructure setup, and background worker orchestration.

The service combines:

    Live Match Data
          +
    Historical Football Intelligence
          +
    ML Feature Pipelines
          +
    LLM-Based Analysis Agents
          ↓
    Unified Match Intelligence API


Core responsibilities:
    - Initialize FastAPI application and API routes.
    - Establish shared Redis and Weaviate connections.
    - Launch and manage background intelligence workers.
    - Coordinate producer, ML, and agent pipelines.
    - Provide health monitoring endpoints.
    - Handle graceful application shutdown.


Runtime architecture:

    FastAPI Application
            |
            |
        Redis State Layer
            |
    -------------------------
    |       |       |       |
 Producer Momentum Intel Narrative
    |
 StatsBomb + Live Feed


Background workers:
    - hybrid producer
    - momentum model
    - match intelligence agent
    - counterfactual engine
    - tactical analysis
    - narrative detection
    - briefing generation

The application intentionally runs with a single process worker because
several components maintain process-local state. Horizontal scaling should
occur through independent service instances sharing Redis rather than through
multiple uvicorn worker processes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.weaviate_client import get_weaviate_client
from api.routes._security import TRIGGER_TOKEN
from api.routes.briefing_routes import router as briefing_router
from api.routes.counterfactual_routes import router as cf_router
from api.routes.group_table import router as group_table_router
from api.routes.intel import router as intel_router
from api.routes.lineups import router as lineups_router
from api.routes.match import router as match_router
from api.routes.match_stream import router as stream_router
from api.routes.momentum import router as momentum_router
from api.routes.narrative import router as narrative_router
from api.routes.narrative_comments import router as narrative_comments_router
from api.routes.predict import router as predict_router
from api.routes.tactical import router as tactical_router
from api.routes.team_form import router as team_form_router
from api.workers.briefing_worker import run as briefing_worker_run
from api.workers.counterfactual_worker import run as cf_worker_run
from api.workers.hybrid_producer import run as hybrid_producer_run
from api.workers.intel_worker import run as intel_worker_run
from api.workers.momentum_worker import run as momentum_worker_run
from api.workers.narrative_worker import run as narrative_worker_run
from api.workers.prediction_worker import run as prediction_worker_run
from api.workers.tactical_worker import run as tactical_worker_run
from ml.tactical_indexer import ensure_indexed as ensure_tactical_indexed

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PORT = int(os.getenv("PORT", "8000"))
WEB_CONCURRENCY = int(os.getenv("WEB_CONCURRENCY", "1"))
HEALTH_CACHE_S = 30.0

log = logging.getLogger(__name__)

log.info(
    "env check — ZAFRONIX_API_KEY=%s GROQ_API_KEY=%s API_SPORTS_KEY=%s TRIGGER_TOKEN=%s",
    "set" if os.getenv("ZAFRONIX_API_KEY") else "MISSING",
    "set" if os.getenv("GROQ_API_KEY") else "MISSING",
    (
        "set"
        if os.getenv("API_SPORTS_KEY")
        else "not set (optional — falls back to free tiers)"
    ),
    (
        "set"
        if os.getenv("TRIGGER_TOKEN")
        else "MISSING (all /trigger debug endpoints are open!)"
    ),
)

if WEB_CONCURRENCY > 1:
    raise RuntimeError(
        "main_hybrid.py holds in-process state (ScoreTracker, momentum EWMA, "
        "counterfactual coverage) and runs its own background producer/workers. "
        "Running with WEB_CONCURRENCY > 1 gives each worker process a disjoint "
        "copy of that state and multiple competing producers. Scale by running "
        "additional independent instances behind a shared Redis, not via "
        "uvicorn --workers. (audit H2)"
    )


async def _wait_for_data(r: aioredis.Redis, timeout: float = 180.0) -> bool:
    """
    Wait until initial match data becomes available.

    Used during startup to prevent derived-data workers from executing before
    the hybrid producer has populated canonical MatchState objects.

    Uses exponential backoff polling to reduce Redis load while allowing
    slow first-run initialization such as StatsBomb event loading.
    """
    deadline = time.monotonic() + timeout
    delay = 2.0
    while time.monotonic() < deadline:
        ids = await r.smembers("matches:active")
        if ids:
            log.info(f"Data ready — fixtures: {ids}")
            return True
        await asyncio.sleep(delay)
        delay = min(delay * 1.3, 10.0)
    log.warning("_wait_for_data: timed out")
    return False


async def _guarded(name: str, r: aioredis.Redis, fn) -> None:
    """
    Start a worker only after verifying that source match data is available.

    Prevents downstream ML and agent workers from running against empty state
    during application startup.

    This creates an ordered dependency chain:
        Producer → MatchState → Derived Intelligence Workers
    """
    if await _wait_for_data(r):
        await fn(r)
    else:
        log.warning(f"{name}: skipping — no data within timeout")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.

    Startup:
        - Create Redis connection.
        - Initialize Weaviate client.
        - Launch background workers.
        - Trigger tactical indexing.
        - Seed active fixtures.

    Shutdown:
        - Cancel background tasks.
        - Await worker termination.
        - Close external connections.

    The lifecycle manager provides centralized ownership of all long-running
    application processes.
    """
    app.state.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    wv = get_weaviate_client()
    r = app.state.redis

    hybrid_task = asyncio.create_task(hybrid_producer_run(r), name="hybrid_producer")
    momentum_task = asyncio.create_task(_guarded("momentum", r, momentum_worker_run))
    intel_task = asyncio.create_task(_guarded("intel", r, intel_worker_run))
    cf_task = asyncio.create_task(_guarded("counterfactual", r, cf_worker_run))
    briefing_task = asyncio.create_task(_guarded("briefing", r, briefing_worker_run))
    tactical_task = asyncio.create_task(_guarded("tactical", r, tactical_worker_run))

    narrative_task = asyncio.create_task(narrative_worker_run(r), name="narrative")
    prediction_task = asyncio.create_task(prediction_worker_run(r), name="prediction")

    asyncio.create_task(ensure_tactical_indexed(), name="tactical_auto_index")

    yield

    for t in (
        hybrid_task,
        momentum_task,
        intel_task,
        cf_task,
        briefing_task,
        narrative_task,
        tactical_task,
        prediction_task,
    ):
        t.cancel()
    await asyncio.gather(
        hybrid_task,
        momentum_task,
        intel_task,
        cf_task,
        briefing_task,
        narrative_task,
        tactical_task,
        prediction_task,
        return_exceptions=True,
    )
    wv.close()
    await app.state.redis.aclose()


app = FastAPI(
    title="WC2026 Match Intelligence — HYBRID",
    version="0.6.0-hybrid",
    description="Live WC 2026 scores (worldcup26.ir) + StatsBomb historical stats",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stream_router, prefix="/matches", tags=["stream"])
app.include_router(momentum_router, prefix="/matches", tags=["momentum"])
app.include_router(intel_router, prefix="/matches", tags=["intel"])
app.include_router(cf_router, prefix="/matches", tags=["counterfactual"])
app.include_router(briefing_router, prefix="/matches", tags=["briefing"])
app.include_router(tactical_router, prefix="/matches", tags=["tactical"])
app.include_router(match_router, prefix="/matches", tags=["matches"])
app.include_router(predict_router)
app.include_router(narrative_router, prefix="/narrative", tags=["narrative"])
app.include_router(
    narrative_comments_router, prefix="/narrative", tags=["narrative-comments"]
)
app.include_router(group_table_router, prefix="/matches", tags=["group-table"])
app.include_router(lineups_router, prefix="/matches", tags=["lineups"])
app.include_router(team_form_router, prefix="/matches", tags=["team-form"])


_health_cache: dict = {"at": 0.0, "weaviate": "unavailable", "collections": {}}


def _probe_weaviate() -> tuple[str, dict]:
    """
    Perform synchronous Weaviate readiness checks.

    Runs outside the async event loop because the client performs blocking
    operations.

    Returns:
        - service readiness status
        - indexed collection statistics

    The probe is isolated so slow vector database responses cannot block API
    request handling.
    """
    wv = get_weaviate_client()
    ready = wv.ready
    return ("ready" if ready else "unavailable", wv.counts() if ready else {})


@app.get("/health", tags=["meta"])
async def health():
    """
    Return application readiness and dependency status.

    Reports:
        - service availability
        - current operating mode
        - live fixture visibility
        - Weaviate readiness
        - indexed collection statistics

    Weaviate checks are cached briefly to avoid expensive vector database
    calls during frequent infrastructure health probes.
    """
    now = time.monotonic()
    if now - _health_cache["at"] > HEALTH_CACHE_S:
        status, collections = await asyncio.to_thread(_probe_weaviate)
        _health_cache["weaviate"] = status
        _health_cache["collections"] = collections
        _health_cache["at"] = now

    ids = await app.state.redis.sunion("matches:active", "matches:completed")
    return {
        "status": "ok",
        "mode": "hybrid",
        "data_source": "worldcup26.ir + StatsBomb open-data",
        "fixtures": list(ids),
        "weaviate": _health_cache["weaviate"],
        "collections": _health_cache["collections"],
    }
