"""
FastAPI entry point.

Real WC 2026 match data from ESPN's public API (scores, stats, events,
lineups, play-by-play), with the committed snapshot in data/wc2026 as the
permanent fallback. No paid API key is needed.

Usage:
    set PYTHONPATH=.
    set GROQ_API_KEY=gsk_...
    python -m uvicorn api.main_hybrid:app --host 0.0.0.0 --port 8000 --reload

Runtime:
    - Leader election (api/supervisor.py): only the instance holding the
      Redis leader lock runs the producer and workers, so any number of
      instances can share one Redis.
    - Crashed workers restart with backoff; /health reports their liveness.
    - /health caches Weaviate collection counts for 30s.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

# Must run before any other project module is imported: several modules read
# env vars at import time.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from agents.weaviate_client import get_weaviate_client
from kg.graph_builder import ensure_graph_built
from kg.neo4j_client import get_neo4j_client
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
from api.workers.match_producer import run as match_producer_run
from api.workers.intel_worker import run as intel_worker_run
from api.workers.momentum_worker import run as momentum_worker_run
from api.workers.narrative_worker import run as narrative_worker_run
from api.workers.prediction_worker import run as prediction_worker_run
from api.workers.tactical_worker import run as tactical_worker_run
from api import supervisor
from api.routes._sse import close_hub
from ml.tactical_indexer import ensure_indexed as ensure_tactical_indexed

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PORT = int(os.getenv("PORT", "8000"))
HEALTH_CACHE_S = 30.0

# Sole entrypoint — worker/route/agent modules assume logging is configured
# here rather than configuring it themselves (see match_producer.py's own
# __main__-gated basicConfig, never hit when imported).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Self-verifying startup check (pre-launch checklist item 0.1/0.5): confirms
# .env actually loaded into THIS process's environment, not just that the
# file has the right values on disk. Logged once at import time, well
# after load_dotenv() ran at the very top of this file.
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

# (fn, interval_s, needs match data) — interval feeds /health staleness.
WORKERS = {
    "match_producer": (match_producer_run, 30.0, False),
    "momentum": (momentum_worker_run, 30.0, True),
    "intel": (intel_worker_run, 30.0, True),
    "counterfactual": (cf_worker_run, 30.0, True),
    "briefing": (briefing_worker_run, 300.0, True),
    "tactical": (tactical_worker_run, 120.0, True),
    # narrative and prediction read no match state.
    "narrative": (narrative_worker_run, 60.0, False),
    "prediction": (prediction_worker_run, 1800.0, False),
}


def _spawn_workers(r: aioredis.Redis) -> list[asyncio.Task]:
    tasks = [
        asyncio.create_task(
            supervisor.supervise(name, fn, r, interval, needs_data=needs_data),
            name=name,
        )
        for name, (fn, interval, needs_data) in WORKERS.items()
    ]
    # One-shot: each fills an empty store and exits. Leader-only so two
    # instances never index at once.
    tasks.append(asyncio.create_task(ensure_tactical_indexed(), name="tactical_auto_index"))
    tasks.append(asyncio.create_task(ensure_graph_built(), name="kg_auto_build"))
    return tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    wv = get_weaviate_client()
    kg = get_neo4j_client()
    r = app.state.redis

    leader_task = asyncio.create_task(
        supervisor.run_as_leader(r, lambda: _spawn_workers(r)), name="leader"
    )

    yield

    leader_task.cancel()
    await asyncio.gather(leader_task, return_exceptions=True)
    await close_hub()
    wv.close()
    kg.close()
    await app.state.redis.aclose()


app = FastAPI(
    title="WC2026 Match Intelligence",
    version="0.7.0",
    description="Real WC 2026 match data (ESPN public API + committed snapshot)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP-request-level metrics (latency, in-progress, status codes) at
# GET /metrics. Worker-tick and SSE-connection metrics (monitoring/metrics.py)
# cover what this instrumentator has no visibility into.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

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
    """Blocking Weaviate readiness and counts; run via asyncio.to_thread."""
    wv = get_weaviate_client()
    ready = wv.ready
    return ("ready" if ready else "unavailable", wv.counts() if ready else {})


@app.get("/health", tags=["meta"])
async def health():
    now = time.monotonic()
    if now - _health_cache["at"] > HEALTH_CACHE_S:
        status, collections = await asyncio.to_thread(_probe_weaviate)
        _health_cache["weaviate"] = status
        _health_cache["collections"] = collections
        _health_cache["at"] = now

    ids = await app.state.redis.sunion("matches:active", "matches:completed")
    sup = supervisor.health()
    return {
        "status": sup["status"],
        "role": sup["role"],
        "workers": sup["workers"],
        "data_source": "ESPN public API + data/wc2026 snapshot",
        "fixtures": list(ids),
        "weaviate": _health_cache["weaviate"],
        "collections": _health_cache["collections"],
    }
