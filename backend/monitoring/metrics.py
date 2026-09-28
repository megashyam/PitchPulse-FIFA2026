"""
Shared Prometheus metrics.

Covers what prometheus-fastapi-instrumentator (HTTP metrics, wired in
api/main_hybrid.py) can't see: the recurring asyncio workers and live SSE
connection counts.
"""

import time

from prometheus_client import Counter, Gauge, Histogram

WORKER_TICKS = Counter(
    "wc2026_worker_ticks_total",
    "Number of completed worker loop iterations",
    ["worker"],
)

WORKER_TICK_DURATION = Histogram(
    "wc2026_worker_tick_duration_seconds",
    "Duration of one worker loop iteration",
    ["worker"],
)

WORKER_ERRORS = Counter(
    "wc2026_worker_errors_total",
    "Number of worker loop iterations that raised",
    ["worker"],
)

SSE_CONNECTIONS = Gauge(
    "wc2026_sse_connections",
    "Currently open SSE connections",
    ["channel"],
)

WORKER_RESTARTS = Counter(
    "wc2026_worker_restarts_total",
    "Number of times the supervisor restarted a worker",
    ["worker"],
)

WORKER_LAST_TICK = Gauge(
    "wc2026_worker_last_tick_timestamp_seconds",
    "Unix time of the last completed worker loop iteration",
    ["worker"],
)

LLM_QUEUE_DEPTH = Gauge(
    "wc2026_llm_queue_depth",
    "LLM calls waiting for a backend slot",
    ["backend", "priority"],
)

LLM_WAIT = Histogram(
    "wc2026_llm_queue_wait_seconds",
    "Time an LLM call waited for a backend slot",
    ["backend", "priority"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

INTEL_NARRATIONS = Counter(
    "wc2026_intel_narrations_total",
    "Match-intel narration outcomes from the LangGraph pipeline",
    ["kind", "outcome"],  # grounded_first | grounded_retry | template_*
)

# Monotonic time of each worker's last completed tick, read by /health.
LAST_TICK: dict[str, float] = {}


def tick_done(worker: str) -> None:
    WORKER_TICKS.labels(worker).inc()
    WORKER_LAST_TICK.labels(worker).set(time.time())
    LAST_TICK[worker] = time.monotonic()
