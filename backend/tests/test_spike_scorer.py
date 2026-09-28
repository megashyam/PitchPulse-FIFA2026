"""
Offline tests for SpikeScorer rules and live-source parsing.

Covers cross-source corroboration, activity floors, outage handling, warm-up,
episode logic, the per-tick alert cap, state persistence, the window-rate and
Trends-lift helpers, and source response parsing, with no network access.
"""

import asyncio
import json
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest

from agents import spike_scorer as ss
from agents import narrative_spike_detector as nsd
from agents.spike_scorer import SpikeScorer

BASE = {"mastodon": 6.0, "bluesky": 200.0, "trends": 1.0, "wikipedia": 4.0}


def _noisy(rng, src):
    if src == "trends":
        return float(np.exp(rng.normal(0, 0.1)))
    return float(rng.poisson(BASE[src]))


def _warm(sc, topic="T", n=120, seed=0, sources=ss.SOURCES):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        sc.update(topic, {s: (_noisy(rng, s) if s in sources else None) for s in ss.SOURCES})


def _tick(sc, topic="T", seed=1, **over):
    rng = np.random.default_rng(seed)
    vals = {s: _noisy(rng, s) for s in ss.SOURCES}
    vals.update(over)
    return sc.update(topic, vals)


def _at_z(sc, topic, src, z):
    """Raw value that lands at roughly `z` sigma for `src`."""
    base = np.fromiter(sc._state[topic].hist[src], float)[:-ss.GAP]
    med = np.median(base)
    scale = max(1.4826 * np.median(np.abs(base - med)), ss.MIN_SCALE)
    x = med + z * scale
    return float(np.exp(x)) if src == "trends" else float(np.expm1(x))


def test_two_source_surge_alerts():
    sc = SpikeScorer()
    _warm(sc)
    r = _tick(sc, bluesky=_at_z(sc, "T", "bluesky", 4), mastodon=_at_z(sc, "T", "mastodon", 4))
    assert r.candidate and set(r.surging) == {"bluesky", "mastodon"}
    assert 0.0 <= r.severity <= 1.0


def test_single_source_4sigma_does_not_alert_but_7sigma_does():
    sc = SpikeScorer()
    _warm(sc)
    r = _tick(sc, bluesky=_at_z(sc, "T", "bluesky", 4))
    assert not r.candidate
    r = _tick(sc, bluesky=_at_z(sc, "T", "bluesky", 7))
    assert r.candidate and r.solo and r.surging == ["bluesky"]


def test_low_count_surge_is_ignored():
    sc2 = SpikeScorer()
    for _ in range(100):
        sc2.update("W", {"wikipedia": 0.0, "mastodon": 0.0})
    r = sc2.update("W", {"wikipedia": 2.0, "mastodon": 2.0})
    assert r.z["mastodon"] >= ss.Z_CORROB and r.z["wikipedia"] >= ss.Z_CORROB
    assert not r.candidate and r.surging == []


def test_drop_to_zero_does_not_alert():
    sc = SpikeScorer()
    _warm(sc)
    r = sc.update("T", {s: 0.0 for s in ss.SOURCES})
    assert not r.candidate and r.surging == []
    assert all(z < 0 for z in r.z.values() if z is not None)
    assert r.z["bluesky"] is None  # busy source reading 0 = outage


def test_zero_on_busy_source_is_outage_not_stored():
    sc = SpikeScorer()
    _warm(sc)
    n = len(sc._state["T"].hist["bluesky"])
    for _ in range(150):
        r = sc.update("T", {"mastodon": 6.0, "bluesky": 0.0, "trends": 1.0, "wikipedia": 4.0})
        assert r.z["bluesky"] is None
    assert len(sc._state["T"].hist["bluesky"]) == n
    r = _tick(sc)  # recovery is not a surge
    assert not r.candidate


def test_none_values_never_scored_or_stored():
    sc = SpikeScorer()
    _warm(sc, sources=("mastodon", "bluesky"))
    assert len(sc._state["T"].hist["trends"]) == 0
    r = sc.update("T", {"mastodon": 6.0, "bluesky": 200.0, "trends": None, "wikipedia": None})
    assert r.z["trends"] is None and r.z["wikipedia"] is None
    assert len(sc._state["T"].hist["wikipedia"]) == 0


def test_no_alert_during_warmup():
    sc = SpikeScorer()
    _warm(sc, n=ss.MIN_BASELINE - 1)
    assert sc.warmup_remaining("T") == 1
    r = sc.update("T", {"mastodon": 500.0, "bluesky": 50000.0, "trends": 20.0, "wikipedia": 200.0})
    assert not r.candidate and r.warming


def test_sustained_surge_alerts_once_then_rearms():
    sc = SpikeScorer()
    _warm(sc)
    hi = {"mastodon": _at_z(sc, "T", "mastodon", 5), "bluesky": _at_z(sc, "T", "bluesky", 5)}
    fired = 0
    for i in range(20):
        r = _tick(sc, seed=100 + i, **hi)
        fired += len(sc.emit([r]))
    assert fired == 1
    for i in range(ss.EXIT_TICKS + 5):
        _tick(sc, seed=200 + i)
    assert not sc._state["T"].in_episode
    r = _tick(sc, seed=999, **hi)
    assert r.candidate and len(sc.emit([r])) == 1


def test_cap_defers_extra_topics_to_next_tick():
    sc = SpikeScorer()
    topics = [f"T{i}" for i in range(5)]
    for i, t in enumerate(topics):
        _warm(sc, t, seed=i)
    hi = {t: {"mastodon": _at_z(sc, t, "mastodon", 5), "bluesky": _at_z(sc, t, "bluesky", 5)} for t in topics}
    first = sc.emit([_tick(sc, t, seed=50, **hi[t]) for t in topics], cap=3)
    assert len(first) == 3
    second = sc.emit([_tick(sc, t, seed=51, **hi[t]) for t in topics], cap=3)
    assert {r.topic for r in second} == set(topics) - {r.topic for r in first}


def test_state_round_trip():
    sc = SpikeScorer()
    _warm(sc)
    sc.mark_alerted("T")
    blob = json.loads(json.dumps(sc.export_state()))
    sc2 = SpikeScorer()
    sc2.load_state(blob)
    assert sc2.export_state() == sc.export_state()
    a = _tick(sc, seed=7)
    b = _tick(sc2, seed=7)
    assert a.z == b.z


# ------------------------------------------------------------ signal helpers


def test_window_rate_counts_and_saturation():
    now = 10_000.0
    ts = [now - 60 * i for i in range(10)] + [now - 4000]
    assert ss.window_rate(ts, now, 1800, 40) == pytest.approx(10 / 1800 * 3600)
    full = [now - 10 * i for i in range(40)]  # 40 posts in 390s, page full
    assert ss.window_rate(full, now, 1800, 40) == pytest.approx(40 / 390 * 3600)


def test_trends_lift_is_scale_invariant():
    series = [20] * 55 + [60] * 5
    assert ss.trends_lift(series) == pytest.approx(3.0)
    assert ss.trends_lift([v * 1.7 for v in series]) == pytest.approx(3.0)
    assert ss.trends_lift([0] * 55 + [100] * 5) is None  # sparse baseline


# ------------------------------------------------------------ live sources


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _run(coro_fn, handler):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await coro_fn(c)

    return asyncio.run(go())


def test_wikipedia_params_and_count():
    seen = {}
    now = time.time()

    def handler(req):
        seen.update(dict(req.url.params))
        revs = [{"timestamp": _iso(now - 300 * i)} for i in range(6)]
        return httpx.Response(200, json={"query": {"pages": [{"title": "x", "revisions": revs}]}})

    src = nsd.WikipediaSource()
    val, prov = _run(lambda c: src.read(c, "USA"), handler)
    assert prov == "live" and val == pytest.approx(6.0)
    assert seen["titles"] == "United States men's national soccer team"
    assert seen["prop"] == "revisions" and "rvend" in seen and seen["redirects"] == "1"
    assert nsd.wiki_article("Brazil") == "Brazil national football team"
    assert nsd.wiki_article("WC2026") == "2026 FIFA World Cup"


def test_mastodon_window_and_saturation(monkeypatch):
    monkeypatch.setenv("MASTODON_ACCESS_TOKEN", "x")
    now = time.time()

    def page(stamps):
        return lambda req: httpx.Response(
            200,
            json={"statuses": [{"created_at": _iso(t), "content": "<p>hi</p>"} for t in stamps]},
        )

    src = nsd.MastodonSource()
    # unordered, 3 inside 30 min, 2 older
    stamps = [now - 100, now - 5000, now - 1000, now - 9000, now - 1700]
    val, prov = _run(lambda c: src.read(c, "Spain"), page(stamps))
    assert prov == "live" and val == pytest.approx(6.0)
    sat = [now - 15 * i for i in range(40)]
    val, _ = _run(lambda c: src.read(c, "Spain"), page(sat))
    assert val == pytest.approx(40 / 585 * 3600, rel=0.02)  # not the capped 80/hr


def test_bluesky_sort_and_since(monkeypatch):
    monkeypatch.setenv("BLUESKY_HANDLE", "h")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "p")
    now = time.time()
    seen = {}

    def handler(req):
        if req.url.path.endswith("createSession"):
            return httpx.Response(200, json={"accessJwt": "t"})
        seen.update(dict(req.url.params))
        posts = [{"indexedAt": _iso(now - 60 * i), "record": {"text": "x"}, "author": {"handle": "a"}} for i in range(5)]
        return httpx.Response(200, json={"posts": posts})

    src = nsd.BlueskySource()
    val, prov = _run(lambda c: src.read(c, "Japan"), handler)
    assert prov == "live" and val == pytest.approx(5 / 900 * 3600)
    assert seen["sort"] == "latest" and seen["limit"] == "100"
    since = datetime.fromisoformat(seen["since"].replace("Z", "+00:00")).timestamp()
    assert abs(since - (now - 900)) < 5


def test_trends_lift_from_dataframe():
    pd = pytest.importorskip("pandas")
    src = nsd.TrendsSource.__new__(nsd.TrendsSource)

    class Fake:
        def build_payload(self, *a, **k):
            pass

        def interest_over_time(self):
            vals = [10] * 55 + [40] * 5 + [99]
            return pd.DataFrame({"Spain": vals, "isPartial": [False] * 60 + [True]})

    src._pytrends = Fake()
    assert src._blocking_read("Spain") == pytest.approx(4.0)


def test_trends_plan_limits_fetches():
    src = nsd.TrendsSource.__new__(nsd.TrendsSource)
    src._cache = {"A": (1.0, 0.0)}
    src.plan(["A", "B", "C", "D"], now=10_000.0)
    assert len(src._due) == nsd.TRENDS_PER_TICK


# ------------------------------------------------------------ detector replay


def test_detector_replay_emits_injected_spike_and_state_round_trips():
    det = nsd.NarrativeSpikeDetector(topics=["Spain", "Japan"])
    rng = np.random.default_rng(3)

    def point(topic, tick, boost=1.0, mock_trends=True):
        return nsd.SignalPoint(
            tick=tick,
            topic=topic,
            mastodon=float(rng.poisson(6 * boost)),
            bluesky=float(rng.poisson(200 * boost)),
            trends=1.0,
            wikipedia=float(rng.poisson(4)),
            data_sources={"mastodon": "live", "bluesky": "live", "trends": "mock" if mock_trends else "live", "wikipedia": "live"},
        )

    spikes = []
    for t in range(90):
        det._tick_count = t
        boost = 6.0 if t == 85 else 1.0
        spikes += det.process([point("Spain", t, boost), point("Japan", t)])
    assert [s.topic for s in spikes] == ["Spain"]
    s = spikes[0]
    assert set(s.source_names) >= {"mastodon", "bluesky"}
    assert "σ above 3h baseline" in s.summary and s.data_sources["trends"] == "mock"
    assert s.z_scores["trends"] is None

    rows = det.trending()
    assert {r["topic"] for r in rows} == {"Spain", "Japan"}
    for r in rows:
        assert 0.0 <= r["severity"] <= 1.0 and set(r["sources"]) == set(ss.SOURCES)

    det2 = nsd.NarrativeSpikeDetector(topics=["Spain", "Japan"])
    det2.load_state(json.loads(json.dumps(det.export_state())))
    assert det2.export_state() == det.export_state()
    assert det2.warmup_remaining() == 0
