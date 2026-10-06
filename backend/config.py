"""Tunables in one place. Env-backed values are read at import; .env is
loaded before any project import (api/main_hybrid.py, tests/conftest.py).
Secrets read per call (Mastodon/Bluesky credentials) stay in
their modules. Fitted coefficients, literature constants and pitch geometry
stay with the code that uses them."""

import os

# infrastructure (env)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:3000")
TRIGGER_TOKEN = os.getenv("TRIGGER_TOKEN", "")

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "wc2026pass")

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
OLLAMA_CONCURRENCY = int(os.getenv("OLLAMA_CONCURRENCY", "1"))
# one window for every call: Ollama reloads the model (~8s) when num_ctx changes
OLLAMA_NUM_CTX = 1536

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_CACHE_TTL = float(os.getenv("ODDS_CACHE_TTL", "5400"))
ESPN_LEAGUE = os.getenv("ESPN_LEAGUE", "fifa.world")
ESPN_SEASON = os.getenv("ESPN_SEASON", "2026")
MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "mastodon.social")

# workers
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))  # producer, seconds
FT_ACTIVE_GRACE_S = int(os.getenv("FT_ACTIVE_GRACE_S", str(6 * 3600)))  # FT match stays active
INTEL_BACKFILL_PER_TICK = 2  # completed fixtures without an intel feed, per tick
CF_BACKFILL_PER_TICK = 1  # completed fixtures without a CF feed, per tick
# LLM arcs for the top N trending stories each tick; the on-demand arc
# endpoint (api/routes/narrative.py) covers the rest.
ARC_TOP_N = 6
STATE_EVERY = 5  # ticks

# Redis TTLs / feed caps (seconds; caps are list lengths)
TTL_LIVE = 3_600  # match in progress
TTL_COMPLETED = 30 * 86_400  # match history
INTEL_FEED_CAP = 30
CF_FEED_MAX = 19  # up to 20 entries (0-indexed ltrim)
BRIEFING_FEED_CAP = 9  # up to 10 briefings per match — kickoff/HT/FT plus room
BRIEFING_TTL = 86_400
ARC_CACHE_TTL = 1_800  # an unchanged surge keeps its arc for 30 min
SCORER_STATE_TTL = 3_600
TRENDING_TTL = 3_600
SPIKE_TTL = 86_400
SPIKE_FEED_MAX = 49  # up to 50 spikes (0-indexed ltrim)
STREAM_LATEST_TTL = 3_600

# intel agent
MIN_INTERVAL_S = 25.0  # wall-clock gap between colour narratives
PERIODIC_MINS = 5  # match minutes between periodic tactical narratives
NARRATE_MIN_SCORE = 0.20  # narratability needed without a forced tick
LLM_MIN_SCORE = 0.50  # above this the LLM writes; below, the template
XG_GAP_HIGH = 0.8  # xG-vs-goals gap worth +0.20 narratability
XG_GAP_LOW = 0.4  # gap worth +0.10
XG_GAP_NARRATE = 0.6  # gap that makes the colour entry an xg_divergence

# counterfactual
CF_SIMS = int(os.getenv("CF_SIMS", "20000"))  # paths per before/after simulation
DELTA_THRESHOLD = 0.003  # min |Δ champion prob| to list a team (and ≥ 2 paired SE)
CF_MIN_GAP_S = 45.0  # live path: one event per this many seconds
CF_MAX_EVENTS_PER_PASS = 20  # backfill path: events per tick
# prompt asks for one 60-80 word paragraph; small models ignore that, so this token cap
# (~80-90 words) enforces it
CF_NARRATIVE_MAX_TOKENS = 130

# narrative
BASELINE_TICKS = 180  # 3h of 60s ticks
MIN_BASELINE = 60  # live points before a source is scored
GAP = 5  # most recent points left out of the baseline
MIN_SCALE = 0.35  # log units
Z_CORROB = 3.0  # robust z for a source backed by another
Z_SOLO = 6.0  # robust z for a lone source
Z_EXIT = 1.5
EXIT_TICKS = 5
MAX_SPIKES_PER_TICK = 3
# A zero reading from a source whose baseline median is at least this rate
# (per hour) is treated as an outage: not scored, not stored.
OUTAGE_ZERO_RATE = 30.0
# Trends index (0-100) the earlier part of the hour must reach for a lift.
TRENDS_MIN_MEDIAN = 3.0
# Raw floor a surging source must also clear (per hour; lift for trends).
MIN_COUNT = {"mastodon": 6.0, "bluesky": 20.0, "trends": 1.5, "wikipedia": 3.0}
MASTODON_WINDOW_S = 30 * 60
MASTODON_PAGE = 40
BLUESKY_WINDOW_S = 15 * 60
BLUESKY_PAGE = 100
WIKI_WINDOW_S = 60 * 60
WIKI_PAGE = 50
WIKI_CACHE_S = 5 * 60
TRENDS_CACHE_S = 15 * 60
TRENDS_STALE_S = 30 * 60
TRENDS_PER_TICK = 2
TRENDS_BACKOFF_S = 30 * 60

# RAG
HYBRID_ALPHA = 0.75  # 75% vector + 25% BM25
INTEL_TOP_K = 5
ARC_TOP_K = 3
READY_TTL_S = 5.0  # cache for Weaviate / Neo4j readiness checks

# simulation
N_SIMS = 50_000
