"""
Shared fixtures for the integration test suite.

The tests exercise the real stack (Redis, Weaviate, the odds provider, the
LLM provider and StatsBomb open data) rather than mocks. Every external
dependency is a fixture that skips, never errors, when the service or
credential is absent:

    # everything
    set PYTHONPATH=.
    python -m pytest tests/ -v -m integration

    # only the services that are currently available
    python -m pytest tests/ -v -m integration -k "redis or statsbomb"

Environment (mirrors the application's own os.getenv usage):
    REDIS_URL              default redis://localhost:6379
    WEAVIATE_HOST/PORT     default localhost:8080
    ODDS_API_KEY           provider keys used by ml.odds_api_client
    GROQ_API_KEY           LLM generation smoke test
    OLLAMA_URL             optional local LLM

Markers: integration, redis, weaviate, odds, llm, statsbomb.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest
import pytest_asyncio

from ml.statsbomb import SB_BASE, COMPETITION_ID, SEASON_IDS

# --------------------------------------------------------------- markers / async


def pytest_configure(config):
    for m in ("integration", "redis", "weaviate", "odds", "llm", "statsbomb"):
        config.addinivalue_line("markers", f"{m}: online integration test")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --------------------------------------------------------------- Redis


@pytest_asyncio.fixture(scope="function")
async def redis_client():
    """Live Redis connection; skips if unreachable.

    Writes use the ``test:`` / high fixture-id namespace and are cleaned up.
    """
    import redis.asyncio as aioredis

    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    r = aioredis.from_url(url, decode_responses=True)
    try:
        await r.ping()
    except Exception as exc:  # noqa: BLE001
        await r.aclose()
        pytest.skip(f"Redis unreachable at {url}: {exc}")

    created_keys: list[str] = []
    r._test_keys = created_keys  # type: ignore[attr-defined]
    try:
        yield r
    finally:
        # scrub any test fixtures we registered
        for k in created_keys:
            await r.delete(k)
        await r.srem("matches:active", *[k.split(":")[1] for k in created_keys] or [""])
        await r.srem(
            "matches:completed", *[k.split(":")[1] for k in created_keys] or [""]
        )
        await r.aclose()


# --------------------------------------------------------------- Weaviate


@pytest.fixture(scope="session")
def weaviate():
    """Process-wide Weaviate client; skips if not ready or not indexed."""
    try:
        from agents.weaviate_client import get_weaviate_client
    except Exception as exc:  # noqa: BLE001  (weaviate client lib not installed)
        pytest.skip(f"Weaviate client library unavailable: {exc}")

    wv = get_weaviate_client()
    if not wv.ready:
        pytest.skip("Weaviate not ready — start the vector DB and index first")
    return wv


@pytest.fixture(scope="session")
def embedder():
    """Production encoder (all-MiniLM-L6-v2); skips if it can't be loaded."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Embedding model unavailable: {exc}")


# --------------------------------------------------------------- odds


@pytest_asyncio.fixture(scope="session")
async def live_odds():
    """Odds snapshot from the configured provider; skips if empty."""
    from ml.odds_api_client import get_oddsapi_client

    odds = await get_oddsapi_client().get_all_odds()
    if not odds:
        pytest.skip("Odds provider returned no markets (key/off-season/rate-limit)")
    return odds


# --------------------------------------------------------------- LLM


@pytest.fixture(scope="session")
def has_llm():
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OLLAMA_URL")):
        pytest.skip("No GROQ_API_KEY / OLLAMA_URL configured")
    return True


# --------------------------------------------------------------- StatsBomb real data


@pytest.fixture(scope="session")
def statsbomb_matches():
    """Real StatsBomb World Cup match list (public open data over HTTPS)."""
    matches: list[dict] = []
    with httpx.Client(timeout=30.0) as c:
        for sid in SEASON_IDS:
            try:
                r = c.get(f"{SB_BASE}/matches/{COMPETITION_ID}/{sid}.json")
                r.raise_for_status()
                matches.extend(r.json())
            except Exception:  # noqa: BLE001
                continue
    if not matches:
        pytest.skip("StatsBomb match list unreachable")
    return matches


@pytest.fixture(scope="session")
def statsbomb_events(statsbomb_matches):
    """Real event stream for one concrete WC match, plus its metadata."""
    m = next(
        (x for x in statsbomb_matches if x.get("home_score") is not None),
        statsbomb_matches[0],
    )
    with httpx.Client(timeout=30.0) as c:
        r = c.get(f"{SB_BASE}/events/{m['match_id']}.json")
        r.raise_for_status()
        events = r.json()
    return {
        "match": m,
        "home": m["home_team"]["home_team_name"],
        "away": m["away_team"]["away_team_name"],
        "home_score": int(m["home_score"]),
        "away_score": int(m["away_score"]),
        "events": events,
    }
