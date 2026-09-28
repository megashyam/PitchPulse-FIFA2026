"""
Narrative spike detector.

Reads four live signals per tracked topic every 60s and hands them to
SpikeScorer (agents/spike_scorer.py) for surge detection.

Signals (per hour, per topic):
    mastodon   posts in the last 30 min of /api/v2/search results
    bluesky    posts in the last 15 min (searchPosts, sort=latest, since=)
    wikipedia  edits in the last 60 min to the topic's article
    trends     lift: last 5 min of the "now 1-H" series over its earlier median

When a source is unavailable a MockSource value is shown in the UI with
data_sources[src] == "mock", but the scorer receives None for it: mock values
are never scored and never enter the baseline.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from agents.spike_scorer import (
    MAX_SPIKES_PER_TICK,
    SOURCES,
    ScoreResult,
    SpikeScorer,
    trends_lift,
    window_rate,
)
from ml.executors import IO_EXECUTOR

log = logging.getLogger(__name__)

HTTP_TIMEOUT = 8.0
# Mastodon search is slower than the other sources under normal load.
MASTODON_TIMEOUT = 15.0

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

TOPICS = [
    # Names match WC2026_TEAMS so the UI's "THIS MATCH" markers line up.
    "Argentina",
    "France",
    "England",
    "Spain",
    "Brazil",
    "Germany",
    "Netherlands",
    "Portugal",
    "Italy",
    "Croatia",
    "Morocco",
    "Belgium",
    "USA",
    "Mexico",
    "Uruguay",
    "Colombia",
    "Japan",
    "Senegal",
    "Canada",
    "South Korea",
    "Denmark",
    "Switzerland",
    "Nigeria",
    "Australia",
    "WC2026",
    "WorldCup2026",
]

WIKI_ARTICLE_OVERRIDES = {
    "USA": "United States men's national soccer team",
    "Canada": "Canada men's national soccer team",
    "Australia": "Australia men's national soccer team",
    "WC2026": "2026 FIFA World Cup",
    "WorldCup2026": "2026 FIFA World Cup",
}

SOURCE_LABELS = {
    "mastodon": "Mastodon",
    "bluesky": "Bluesky",
    "trends": "Trends",
    "wikipedia": "Wikipedia",
}

MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "mastodon.social")
MASTODON_UA = "wc2026-narrative/1.0 (+https://localhost)"

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(content: str) -> str:
    text = _HTML_TAG_RE.sub(" ", content)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_iso_ts(value: str) -> Optional[float]:
    """UTC ISO-8601 (with or without Z / fractional seconds) -> epoch."""
    if not value:
        return None
    try:
        s = value.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(value[:19])
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wiki_article(topic: str) -> str:
    return WIKI_ARTICLE_OVERRIDES.get(topic, f"{topic} national football team")


def format_source(src: str, value: float, z: Optional[float] = None) -> str:
    if src == "trends":
        text = f"Trends {value:.1f}× last hr"
    elif src == "wikipedia":
        text = f"Wikipedia {value:.0f} edits/hr"
    else:
        text = f"{SOURCE_LABELS[src]} {value:.0f} posts/hr"
    if z is not None:
        text += f" ({z:.1f}σ above 3h baseline)"
    return text


@dataclass
class SignalPoint:
    tick: int
    topic: str
    mastodon: float
    bluesky: float
    trends: float
    wikipedia: float
    timestamp: float = field(default_factory=time.time)
    # "live" | "mock" per source
    data_sources: Dict[str, str] = field(default_factory=dict)

    def values(self) -> Dict[str, float]:
        return {s: getattr(self, s) for s in SOURCES}

    def live_values(self) -> Dict[str, Optional[float]]:
        """What the scorer sees: mock values become None."""
        return {
            s: (getattr(self, s) if self.data_sources.get(s) == "live" else None)
            for s in SOURCES
        }


@dataclass
class NarrativeSpike:
    spike_id: str
    topic: str
    tick: int
    severity: float
    sources: Dict[str, float]
    source_names: List[str]
    summary: str
    timestamp: float = field(default_factory=time.time)
    arc: Optional[str] = None
    data_sources: Optional[Dict[str, str]] = None
    z_scores: Optional[Dict[str, Optional[float]]] = None

    def to_dict(self) -> dict:
        return {
            "spike_id": self.spike_id,
            "topic": self.topic,
            "tick": self.tick,
            "severity": round(self.severity, 3),
            "sources": {k: round(v, 2) for k, v in self.sources.items()},
            "source_names": self.source_names,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "arc": self.arc,
            "data_sources": self.data_sources or {},
            "z_scores": _round_z(self.z_scores),
        }


def _round_z(zs: Optional[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    return {k: (round(v, 2) if v is not None else None) for k, v in (zs or {}).items()}


class MockSource:
    """Synthetic display-only value used when a real source is unavailable.
    Phase comes from a stable hash of the topic name."""

    def __init__(self, source_name: str, base_rate: float, spike_factor: float):
        self.name = source_name
        self.base = base_rate
        self.factor = spike_factor
        self._tick = 0

    def _phase_offset(self, topic: str) -> float:
        h = int(hashlib.sha1(f"{self.name}:{topic}".encode()).hexdigest()[:8], 16)
        return (h % 1440) / 1440 * 2 * math.pi

    def read(self, topic: str) -> float:
        self._tick += 1
        offset = self._phase_offset(topic)
        hour_phase = ((self._tick % 1440) / 1440 * 2 * math.pi) + offset
        baseline = self.base * (1 + 0.4 * math.sin(hour_phase))
        noise = random.gauss(0, self.base * 0.15)
        spike = (
            random.gauss(self.base * self.factor, self.base * 0.5)
            if random.random() < 0.05
            else 0.0
        )
        return max(0.0, baseline + noise + spike)


class MastodonSource:
    """Authenticated /api/v2/search. Results aren't strictly time-ordered, so
    posts are sorted by created_at and counted inside the window."""

    def __init__(self):
        self._instance = MASTODON_INSTANCE
        self._token = os.getenv("MASTODON_ACCESS_TOKEN", "")
        self._available = bool(self._token)
        self._consecutive_failures = 0
        self._max_failures_before_mock = 5
        self._mock = MockSource("mastodon", base_rate=6.0, spike_factor=4.0)
        self._last_samples: Dict[str, List[dict]] = {}

        if self._available:
            log.info(f"Mastodon source: authenticated search on {self._instance}")
        else:
            log.warning("Mastodon source: MASTODON_ACCESS_TOKEN not set — mock only")

    def get_samples(self, topic: str) -> List[dict]:
        return self._last_samples.get(topic, [])

    async def read(self, client: httpx.AsyncClient, topic: str) -> tuple[float, str]:
        if not self._available:
            return self._mock.read(topic), "mock"

        try:
            resp = await client.get(
                f"https://{self._instance}/api/v2/search",
                params={
                    "q": topic,
                    "type": "statuses",
                    "limit": MASTODON_PAGE,
                    "resolve": "false",
                },
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "User-Agent": MASTODON_UA,
                    "Accept": "application/json",
                },
                timeout=MASTODON_TIMEOUT,
            )
            resp.raise_for_status()
            posts = resp.json().get("statuses", [])
            self._consecutive_failures = 0

            dated = []
            for p in posts:
                ts = _parse_iso_ts(p.get("created_at", ""))
                if ts is not None:
                    dated.append((ts, p))
            dated.sort(key=lambda x: x[0], reverse=True)
            now = time.time()
            rate = window_rate(
                [ts for ts, _ in dated], now, MASTODON_WINDOW_S, MASTODON_PAGE
            )

            samples = []
            for ts, p in dated[:8]:
                text = _strip_html(p.get("content", ""))
                if not text:
                    continue
                acct = (p.get("account", {}) or {}).get("acct", "anon")
                samples.append(
                    {
                        "text": text[:220],
                        "source": "mastodon",
                        "author": f"@{acct}",
                        "permalink": p.get("url"),
                        "timestamp": ts,
                    }
                )
            self._last_samples[topic] = samples
            return rate, "live"

        except httpx.HTTPStatusError as e:
            self._consecutive_failures += 1
            if e.response.status_code == 401:
                log.warning("Mastodon: access token rejected (401)")
                self._available = False
            self._maybe_disable()
            return self._mock.read(topic), "mock"
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures == 1:
                log.warning(f"Mastodon search error for '{topic}': {e}")
            self._maybe_disable()
            return self._mock.read(topic), "mock"

    def _maybe_disable(self) -> None:
        if self._consecutive_failures >= self._max_failures_before_mock:
            if self._available:
                log.warning(
                    f"Mastodon source: {self._max_failures_before_mock} consecutive "
                    f"failures — mock for the rest of the session."
                )
            self._available = False


class BlueskySource:
    """searchPosts with sort=latest and since=now-15min. Re-authenticates once
    on a 401."""

    def __init__(self):
        self._session_token: Optional[str] = None
        self._last_samples: Dict[str, List[dict]] = {}
        self._handle = os.getenv("BLUESKY_HANDLE", "")
        self._password = os.getenv("BLUESKY_APP_PASSWORD", "")
        self._available = bool(self._handle and self._password)
        self._mock = MockSource("bluesky", base_rate=200.0, spike_factor=4.0)
        if not self._available:
            log.info("Bluesky source: no credentials — mock only")

    def get_samples(self, topic: str) -> List[dict]:
        return self._last_samples.get(topic, [])

    async def _authenticate(self, client: httpx.AsyncClient) -> bool:
        try:
            resp = await client.post(
                "https://bsky.social/xrpc/com.atproto.server.createSession",
                json={"identifier": self._handle, "password": self._password},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            self._session_token = resp.json().get("accessJwt")
            log.info(
                "Bluesky source: authenticated"
                if self._session_token
                else "Bluesky auth returned no token"
            )
            return bool(self._session_token)
        except Exception as e:
            log.info(f"Bluesky source: auth error ({e}) — mock active")
            return False

    async def _search(self, client: httpx.AsyncClient, topic: str, now: float):
        return await client.get(
            "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
            params={
                "q": topic,
                "sort": "latest",
                "since": _iso(now - BLUESKY_WINDOW_S),
                "limit": BLUESKY_PAGE,
            },
            headers={"Authorization": f"Bearer {self._session_token}"},
            timeout=HTTP_TIMEOUT,
        )

    async def read(self, client: httpx.AsyncClient, topic: str) -> tuple[float, str]:
        if not self._available:
            return self._mock.read(topic), "mock"

        if not self._session_token:
            if not await self._authenticate(client):
                return self._mock.read(topic), "mock"

        try:
            now = time.time()
            resp = await self._search(client, topic, now)
            if resp.status_code == 401:
                self._session_token = None
                if not await self._authenticate(client):
                    return self._mock.read(topic), "mock"
                resp = await self._search(client, topic, now)
            resp.raise_for_status()
            posts = resp.json().get("posts", [])

            stamps = [_parse_iso_ts(p.get("indexedAt", "")) for p in posts]
            rate = window_rate(stamps, now, BLUESKY_WINDOW_S, BLUESKY_PAGE)

            samples = []
            for p, ts in list(zip(posts, stamps))[:8]:
                text = (p.get("record", {}) or {}).get("text", "").strip()
                if not text:
                    continue
                author = (p.get("author", {}) or {}).get("handle", "anon")
                samples.append(
                    {
                        "text": text[:220],
                        "source": "bluesky",
                        "author": f"@{author}",
                        "permalink": None,
                        "timestamp": ts if ts is not None else now,
                    }
                )
            self._last_samples[topic] = samples
            return rate, "live"

        except Exception as e:
            log.debug(f"Bluesky read error: {e}")
            return self._mock.read(topic), "mock"


class TrendsSource:
    """Lift from the per-minute "now 1-H" series. pytrends is blocking, so it
    runs on IO_EXECUTOR. At most TRENDS_PER_TICK topics are fetched per tick
    (stalest first) and any error backs the whole source off."""

    def __init__(self):
        self._available = False
        try:
            from pytrends.request import TrendReq

            self._pytrends = TrendReq(
                # retries>0 builds urllib3 Retry(method_whitelist=...), removed in urllib3 2
                hl="en-US", tz=0, timeout=(5, 10), retries=0, backoff_factor=0
            )
            self._available = True
            log.info("Trends source: pytrends ready")
        except ImportError:
            log.info("Trends source: pytrends not installed — mock only")
        self._cache: Dict[str, tuple] = {}  # topic -> (lift, fetched_at)
        self._due: set = set()
        self._backoff_until = 0.0
        self._mock = MockSource("trends", base_rate=1.0, spike_factor=2.0)

    def plan(self, topics: List[str], now: Optional[float] = None) -> None:
        """Pick which topics may hit the network this tick."""
        now = time.time() if now is None else now
        stale = [
            t
            for t in topics
            if t not in self._cache or now - self._cache[t][1] >= TRENDS_CACHE_S
        ]
        stale.sort(key=lambda t: self._cache.get(t, (None, 0.0))[1])
        self._due = set(stale[:TRENDS_PER_TICK])

    def _blocking_read(self, topic: str) -> Optional[float]:
        self._pytrends.build_payload([topic], timeframe="now 1-H")
        df = self._pytrends.interest_over_time()
        if df.empty or topic not in df.columns:
            return None
        if "isPartial" in df.columns:
            df = df[~df["isPartial"].astype(bool)]
        return trends_lift(df[topic].tolist())

    def _cached(self, topic: str, now: float, max_age: float) -> Optional[float]:
        hit = self._cache.get(topic)
        if hit and hit[0] is not None and now - hit[1] < max_age:
            return hit[0]
        return None

    async def read(self, topic: str) -> tuple[float, str]:
        if not self._available:
            return self._mock.read(topic), "mock"

        now = time.time()
        fresh = self._cached(topic, now, TRENDS_CACHE_S)
        if fresh is not None:
            return fresh, "live"

        if topic in self._due and now >= self._backoff_until:
            self._due.discard(topic)
            loop = asyncio.get_running_loop()
            try:
                lift = await loop.run_in_executor(
                    IO_EXECUTOR, self._blocking_read, topic
                )
                self._cache[topic] = (lift, now)
                if lift is not None:
                    return lift, "live"
            except Exception as e:
                log.info(f"Trends error for {topic}: {e} — backing off 30 min")
                self._backoff_until = now + TRENDS_BACKOFF_S

        stale = self._cached(topic, now, TRENDS_STALE_S)
        if stale is not None:
            return stale, "live"
        return self._mock.read(topic), "mock"


class WikipediaSource:
    """Edits in the last hour to the topic's article."""

    def __init__(self):
        self._mock = MockSource("wikipedia", base_rate=1.0, spike_factor=4.0)
        self._cache: Dict[str, tuple] = {}  # topic -> (rate, fetched_at)

    async def read(self, client: httpx.AsyncClient, topic: str) -> tuple[float, str]:
        now = time.time()
        hit = self._cache.get(topic)
        if hit and now - hit[1] < WIKI_CACHE_S:
            return hit[0], "live"
        try:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "revisions",
                    "titles": wiki_article(topic),
                    "rvprop": "timestamp",
                    "rvlimit": WIKI_PAGE,
                    "rvend": _iso(now - WIKI_WINDOW_S),
                    "redirects": 1,
                    "format": "json",
                    "formatversion": 2,
                },
                headers={"User-Agent": "wc2026-narrative/1.0"},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                raise ValueError(f"no article for {topic!r}")
            stamps = [
                _parse_iso_ts(r.get("timestamp", ""))
                for r in pages[0].get("revisions", [])
            ]
            rate = window_rate(stamps, now, WIKI_WINDOW_S, WIKI_PAGE)
            self._cache[topic] = (rate, now)
            return rate, "live"
        except Exception as e:
            log.debug(f"Wikipedia read error for {topic}: {e}")
            return self._mock.read(topic), "mock"


class NarrativeSpikeDetector:
    def __init__(self, topics: List[str] = TOPICS, scorer: Optional[SpikeScorer] = None):
        self.topics = topics
        self._tick_count = 0
        self.scorer = scorer or SpikeScorer()
        self._last_points: Dict[str, SignalPoint] = {}
        self._last_alert: Dict[str, float] = {}

        self._mastodon = MastodonSource()
        self._bluesky = BlueskySource()
        self._trends = TrendsSource()
        self._wikipedia = WikipediaSource()
        self.sources = {
            "mastodon": self._mastodon,
            "bluesky": self._bluesky,
            "trends": self._trends,
            "wikipedia": self._wikipedia,
        }

    def get_last_samples(self, topic: str) -> List[dict]:
        samples = []
        samples.extend(self._mastodon.get_samples(topic))
        samples.extend(self._bluesky.get_samples(topic))
        return samples

    def warmup_remaining(self, topic: Optional[str] = None) -> int:
        return self.scorer.warmup_remaining(topic)

    def export_state(self) -> dict:
        return {"tick": self._tick_count, "scorer": self.scorer.export_state()}

    def load_state(self, state: dict) -> None:
        if not state:
            return
        self._tick_count = int(state.get("tick", self._tick_count))
        self.scorer.load_state(state.get("scorer", {}))

    async def _read_topic(self, client: httpx.AsyncClient, topic: str) -> SignalPoint:
        mastodon_res, bluesky_res, trends_res, wiki_res = await asyncio.gather(
            self._mastodon.read(client, topic),
            self._bluesky.read(client, topic),
            self._trends.read(topic),
            self._wikipedia.read(client, topic),
        )
        return SignalPoint(
            tick=self._tick_count,
            topic=topic,
            mastodon=mastodon_res[0],
            bluesky=bluesky_res[0],
            trends=trends_res[0],
            wikipedia=wiki_res[0],
            data_sources={
                "mastodon": mastodon_res[1],
                "bluesky": bluesky_res[1],
                "trends": trends_res[1],
                "wikipedia": wiki_res[1],
            },
        )

    def _summary(self, topic: str, point: SignalPoint, res: ScoreResult) -> str:
        if res.surging:
            parts = [format_source(s, getattr(point, s), res.z[s]) for s in res.surging]
            return f"{topic} — " + ", ".join(parts)
        if res.warming:
            return (
                f"{topic} — warming up "
                f"({self.warmup_remaining(topic)} ticks of live baseline left)"
            )
        scored = [(z, s) for s, z in res.z.items() if z is not None]
        z, s = max(scored)
        return f"{topic} — within baseline (top: {format_source(s, getattr(point, s), z)})"

    def _make_spike(self, point: SignalPoint, res: ScoreResult) -> NarrativeSpike:
        raw = f"{point.topic}:{point.tick}:{int(point.timestamp)}"
        return NarrativeSpike(
            spike_id=hashlib.sha1(raw.encode()).hexdigest()[:12],
            topic=point.topic,
            tick=point.tick,
            severity=res.severity,
            sources=point.values(),
            source_names=list(res.surging),
            summary=self._summary(point.topic, point, res).replace(" — ", " spike — ", 1),
            timestamp=point.timestamp,
            data_sources=point.data_sources,
            z_scores=dict(res.z),
        )

    def process(self, points: List[SignalPoint]) -> List[NarrativeSpike]:
        """Score one tick's points and return the spikes to emit."""
        results = []
        for p in points:
            self._last_points[p.topic] = p
            try:
                results.append(self.scorer.update(p.topic, p.live_values()))
            except Exception as e:
                log.warning(f"scoring error for {p.topic}: {e}", exc_info=True)

        spikes = []
        for res in self.scorer.emit(results, MAX_SPIKES_PER_TICK):
            point = self._last_points[res.topic]
            spike = self._make_spike(point, res)
            self._last_alert[res.topic] = point.timestamp
            spikes.append(spike)
            log.info(
                f"Spike detected: {res.topic} surging={res.surging} "
                f"severity={res.severity:.2f} id={spike.spike_id}"
            )
        return spikes

    def trending(self, top_n: int = 12) -> List[dict]:
        """Every tracked topic ranked by its latest surge score."""
        now = time.time()
        rows: list = []
        for topic in self.topics:
            point = self._last_points.get(topic)
            res = self.scorer.latest.get(topic)
            if point is None or res is None:
                continue
            max_z = max((z for z in res.z.values() if z is not None), default=0.0)
            rows.append(
                (
                    res.severity,
                    max_z,
                    {
                        "spike_id": f"trend-{hashlib.sha1(topic.encode()).hexdigest()[:10]}",
                        "topic": topic,
                        "tick": self._tick_count,
                        "severity": round(res.severity, 3),
                        "sources": {k: round(v, 2) for k, v in point.values().items()},
                        "source_names": list(res.surging),
                        "summary": self._summary(topic, point, res),
                        "timestamp": now,
                        "arc": None,
                        "is_spike": res.in_episode
                        or (now - self._last_alert.get(topic, 0)) < 600,
                        "data_sources": point.data_sources,
                        "z_scores": _round_z(res.z),
                        "warming_up": res.warming,
                    },
                )
            )
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        return [r[2] for r in rows[:top_n]]

    async def tick(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> List[NarrativeSpike]:
        """Read all topics, score, and return up to MAX_SPIKES_PER_TICK spikes."""
        self._tick_count += 1
        self._trends.plan(self.topics)
        async with httpx.AsyncClient() as client:
            points: List[SignalPoint] = await asyncio.gather(
                *[self._read_topic(client, topic) for topic in self.topics]
            )
        return self.process(list(points))


_detector: Optional[NarrativeSpikeDetector] = None


def get_detector() -> NarrativeSpikeDetector:
    global _detector
    if _detector is None:
        _detector = NarrativeSpikeDetector()
    return _detector
