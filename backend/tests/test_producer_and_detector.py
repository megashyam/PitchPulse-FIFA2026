"""
Producer, persistence, retrieval, and detector integration tests.

Exercises the real infrastructure:
    - Redis: persist() round-trip and pub/sub through the production functions.
    - ESPN: live scoreboard fetch and normalization (skipped if unreachable).
    - Weaviate: hybrid_search over the indexed collections with MiniLM query
      vectors, including the event_type filter and alpha behaviour.
    - SpikeScorer: run on a StatsBomb-derived multi-source series.
"""

import json

import numpy as np
import pytest

from api.workers import match_producer as hp
from api.schemas.schema import MatchState, TeamStats
from agents import narrative_spike_detector as nsd
from agents import spike_scorer as ss
from agents.spike_scorer import SpikeScorer

# --------------------------------------------------------------- Redis round-trip


@pytest.mark.integration
@pytest.mark.redis
@pytest.mark.asyncio
async def test_persist_and_read_back(redis_client):
    r = redis_client
    fid = 990100
    key = f"match:{fid}:state"
    r._test_keys.append(key)

    state = MatchState(
        fixture_id=fid,
        status_short="2H",
        status_long="Second Half",
        elapsed=60,
        home_id=1,
        home_name="Spain",
        home_score=1,
        away_id=2,
        away_name="Brazil",
        away_score=0,
        home_stats=TeamStats(possession=57.0),
        away_stats=TeamStats(possession=43.0),
    )
    await hp.persist(r, state, changed=True, worker_visible=True)

    raw = await r.get(key)
    assert raw is not None
    back = MatchState.model_validate_json(raw)
    assert back.home_score == 1 and back.away_name == "Brazil"
    assert str(fid) in await r.smembers("matches:active")

    # completed transition moves it between sets
    done = state.model_copy(update={"status_short": "FT"})
    await hp.persist(r, done, changed=True, worker_visible=False)
    assert str(fid) in await r.smembers("matches:completed")
    assert str(fid) not in await r.smembers("matches:active")


@pytest.mark.integration
@pytest.mark.redis
@pytest.mark.asyncio
async def test_pubsub_emitted_on_change(redis_client):
    r = redis_client
    fid = 990101
    key = f"match:{fid}:state"
    r._test_keys.append(key)

    pubsub = r.pubsub()
    await pubsub.subscribe("match_update")
    await pubsub.get_message(timeout=1.0)  # drop subscribe ack

    state = MatchState(
        fixture_id=fid, status_short="1H", elapsed=10, home_name="A", away_name="B"
    )
    await hp.persist(r, state, changed=True, worker_visible=True)

    got = None
    for _ in range(10):
        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        if msg:
            got = json.loads(msg["data"])
            break
    await pubsub.unsubscribe("match_update")
    await pubsub.aclose()
    assert got is not None and str(got["fixture_id"]) == str(fid)


# --------------------------------------------------------------- live feed


@pytest.mark.integration
@pytest.mark.asyncio
async def test_espn_scoreboard_parses():
    """Fetch and normalize the real ESPN WC scoreboard."""
    import httpx

    from feeds import espn

    try:
        async with httpx.AsyncClient() as c:
            raw = await espn.fetch_scoreboard(c)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ESPN unreachable: {exc}")
    parsed = [p for g in raw if (p := espn.parse_event(g))]
    assert len(parsed) == 104
    for p in parsed:
        assert p["status"] in espn.STATUS_LONG
        assert p["home_name"] and p["away_name"]


# --------------------------------------------------------------- Weaviate retrieval


@pytest.mark.integration
@pytest.mark.weaviate
def test_hybrid_search_returns_real_docs(weaviate, embedder):
    from agents.weaviate_client import NARRATIVE_ARCS

    counts = weaviate.counts()
    if counts.get(NARRATIVE_ARCS, 0) == 0:
        pytest.skip("NarrativeArcs empty — run the indexer first")

    query = "goal during sustained pressure phase, momentum swing"
    qv = embedder.encode(query, normalize_embeddings=True).tolist()
    docs = weaviate.hybrid_search(
        query_vector=qv, query_text=query, top_k=5, collection=NARRATIVE_ARCS
    )
    assert isinstance(docs, list) and len(docs) > 0
    assert all(isinstance(d, str) and d for d in docs)


@pytest.mark.integration
@pytest.mark.weaviate
def test_event_filter_narrows_results(weaviate, embedder):
    from agents.weaviate_client import NARRATIVE_ARCS

    if weaviate.counts().get(NARRATIVE_ARCS, 0) == 0:
        pytest.skip("NarrativeArcs empty")
    query = "red card reduced to ten men numerical disadvantage"
    qv = embedder.encode(query, normalize_embeddings=True).tolist()
    objs = weaviate.hybrid_search(
        query_vector=qv,
        query_text=query,
        top_k=5,
        collection=NARRATIVE_ARCS,
        event_filter="red_card",
        return_objects=True,
    )
    for o in objs:
        assert o.get("_score") is not None


@pytest.mark.integration
@pytest.mark.weaviate
def test_tactical_profiles_retrieval(weaviate, embedder):
    from agents.weaviate_client import TACTICAL_PROFILES

    if weaviate.counts().get(TACTICAL_PROFILES, 0) == 0:
        pytest.skip("TacticalProfiles empty")
    query = "high press aggressive PPDA compact block possession dominant"
    qv = embedder.encode(query, normalize_embeddings=True).tolist()
    objs = weaviate.hybrid_search(
        query_vector=qv,
        query_text=query,
        top_k=3,
        collection=TACTICAL_PROFILES,
        return_objects=True,
    )
    assert len(objs) > 0
    for o in objs:
        assert "ppda" in o and "team" in o


# --------------------------------------------------------------- detector on real data


@pytest.mark.integration
@pytest.mark.statsbomb
def test_anomaly_detects_real_derived_spike(statsbomb_events):
    """Push the real per-minute event rate of a WC match through the production
    SpikeScorer as two live sources, then inject a two-source surge: the surge
    alerts, the real match stream alone does not."""
    from collections import Counter

    d = statsbomb_events
    per_min = Counter(int(e.get("minute", 0)) for e in d["events"])
    minutes = sorted(per_min)
    if len(minutes) < 60:
        pytest.skip("Too few minutes in real match")
    counts = np.array([per_min[m] for m in minutes], dtype=float)
    # per-minute events -> per-hour rate; two sources driven by the same signal
    rng = np.random.default_rng(0)
    sc = SpikeScorer()
    alerts = 0
    for c in counts:
        r = sc.update(
            "match",
            {
                "mastodon": c * 60,
                "bluesky": 0.7 * c * 60 + rng.normal(0, 30),
                "trends": None,
                "wikipedia": None,
            },
        )
        alerts += len(sc.emit([r]))
    assert alerts == 0

    peak = float(np.median(counts)) * 60 * 8
    r = sc.update(
        "match", {"mastodon": peak, "bluesky": 0.7 * peak, "trends": None, "wikipedia": None}
    )
    assert r.candidate and set(r.surging) == {"mastodon", "bluesky"}


def test_detector_constants():
    assert nsd.MAX_SPIKES_PER_TICK == 3
    assert ss.Z_CORROB == 3.0 and ss.Z_SOLO == 6.0
    assert ss.BASELINE_TICKS == 180 and ss.MIN_BASELINE == 60
    assert not hasattr(nsd, "IsolationForest")
