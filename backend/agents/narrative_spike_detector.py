"""
Real-time IsolationForest-based narrative anomaly detection pipeline over multi-source
social and search intelligence signals.

This module implements the core signal-processing layer for detecting unusual
topic activity across four independent information streams:

    - Mastodon: Real-time social discussion velocity
    - Bluesky: Distributed social engagement signals
    - Google Trends: Search interest momentum
    - Wikipedia: Reference/news activity changes

The detector transforms heterogeneous source measurements into normalized
feature vectors and applies per-topic IsolationForest models to identify
statistical deviations from historical behavior. Each tracked topic maintains
its own rolling activity window, allowing anomaly detection to adapt to the
unique baseline behavior of different entities.

Architecture Overview
---------------------
The detector operates as a streaming time-series anomaly detection system:

    External Sources
        |
        +-- Mastodon social activity
        +-- Bluesky social activity
        +-- Google Trends search interest
        +-- Wikipedia edit activity
        |
        v
    Async Signal Aggregation
        |
        v
    SignalPoint Feature Vector
        |
        v
    Topic-Specific Rolling Windows
        |
        v
    IsolationForest Anomaly Models
        |
        v
    NarrativeSpike Events

Anomaly Detection Workflow
--------------------------
For each detection cycle:

1. Collect current observations from all providers.
2. Append SignalPoint objects to topic-specific rolling windows.
3. Periodically retrain IsolationForest models.
4. Score the latest observation against historical behavior.
5. Apply anomaly threshold and cooldown rules.
6. Generate NarrativeSpike objects for significant deviations.
7. Attribute contributing sources.
8. Emit structured events for downstream narrative agents.

Concurrency
-----------
Network providers run concurrently through asyncio.
Blocking APIs (Google Trends) execute through IO_EXECUTOR to avoid blocking
the event loop.


Outputs
-------
SignalPoint:
    Timestamped multi-source observation.

NarrativeSpike:
    Detected anomaly containing topic, severity, source values,
    contributing providers, summary, and provenance.


Reliability
-----------
Includes async HTTP handling, authentication recovery, cached lookups,
deterministic mock fallback, adaptive baselines, and singleton lifecycle
management.
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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import numpy as np
from sklearn.ensemble import IsolationForest

from ml.executors import IO_EXECUTOR

log = logging.getLogger(__name__)

CONTAMINATION = 0.05
WINDOW_TICKS = 4320
MIN_TICKS = 30
SCORE_THRESH = -0.10
MAX_SPIKES_PER_TICK = 3
HTTP_TIMEOUT = 8.0
MASTODON_TIMEOUT = 15.0

TOPICS = [
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

MIN_ACTIVITY_FLOOR = {
    "mastodon": 8.0,
    "bluesky": 10.0,
    "trends": 40.0,
    "wikipedia": 2.0,
}

MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "mastodon.social")
MASTODON_UA = "wc2026-narrative/1.0 (+https://localhost)"

_TAG_STRIP_RE = re.compile(r"[^A-Za-z0-9]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _topic_to_hashtag(topic: str) -> str:
    """
    Normalize a topic label into a hashtag-compatible identifier.

    Removes unsupported characters so external social APIs can safely consume
    the generated topic tag during source queries.

    Args:
        topic: Raw topic label (team name, country, or tournament keyword).

    Returns:
        str: Sanitized hashtag string containing only alphanumeric characters.
    """
    return _TAG_STRIP_RE.sub("", topic)


def _strip_html(content: str) -> str:
    text = _HTML_TAG_RE.sub(" ", content)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_mastodon_timestamp(created_at: str) -> float:
    """
    Parse Mastodon UTC timestamps into Unix epoch seconds.

    Mastodon returns timestamps in ISO-8601 UTC format. The conversion explicitly
    preserves UTC semantics to avoid host timezone offsets affecting recency
    filtering and activity calculations.

    Args:
        created_at: ISO-8601 timestamp returned by Mastodon.

    Returns:
        float: UTC timestamp in seconds since Unix epoch. Falls back to current
            time when parsing fails.
    """
    try:
        s = created_at[:19]
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


@dataclass
class SignalPoint:
    """
    Immutable representation of a multi-source signal observation.

    A SignalPoint captures one sampling interval ("tick") across all monitored
    providers. The values are later transformed into a numerical feature vector
    for unsupervised anomaly detection using IsolationForest.

    Attributes:
        tick: Sequential detector iteration identifier.
        topic: Entity or keyword being monitored.
        mastodon: Mastodon activity rate.
        bluesky: Bluesky activity rate.
        trends: Google Trends interest score.
        wikipedia: Wikipedia edit activity rate.
        timestamp: Observation creation time.
        data_sources: Indicates whether each measurement came from live or mock data.
    """

    tick: int
    topic: str
    mastodon: float
    bluesky: float
    trends: float
    wikipedia: float
    timestamp: float = field(default_factory=time.time)
    # "live" | "mock" per source — provenance for the UI (audit H4/H11-adjacent)
    data_sources: Dict[str, str] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """
        Convert the signal snapshot into an ML-ready feature representation.

        The resulting vector provides the input space for anomaly detection where
        each dimension represents activity from an independent information source.

        Returns:
            np.ndarray: Four-dimensional feature vector ordered as:
                [mastodon, bluesky, trends, wikipedia].
        """
        return np.array([self.mastodon, self.bluesky, self.trends, self.wikipedia])


@dataclass
class NarrativeSpike:
    """
    Structured representation of a detected narrative anomaly.

    A NarrativeSpike stores the anomaly score interpretation, source attribution,
    and contextual summary required by downstream narrative generation agents.

    Attributes:
        spike_id: Stable identifier for the detected anomaly.
        topic: Topic associated with the spike.
        tick: Detection cycle where the anomaly occurred.
        severity: Normalized anomaly magnitude between 0 and 1.
        sources: Raw activity measurements from each provider.
        source_names: Providers contributing most strongly to the anomaly.
        summary: Human-readable signal description.
        arc: Generated narrative explanation, if available.
        data_sources: Live/mock provenance metadata.
    """

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

    def to_dict(self) -> dict:
        """
        Serialize the anomaly object into an API-compatible dictionary.

        Rounds numerical values for frontend consumption while preserving source
        attribution, provenance metadata, and generated narrative information.

        Returns:
            dict: JSON-compatible representation of the detected spike.
        """
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
        }


class MockSource:
    """
    Deterministic synthetic signal generator used as a provider fallback.

    Creates realistic-looking activity streams using topic-specific phase offsets,
    periodic baseline variation, Gaussian noise, and occasional injected spikes.

    This keeps the anomaly detection pipeline operational during API outages,
    missing credentials, or local development without external dependencies.
    """

    def __init__(self, source_name: str, base_rate: float, spike_factor: float):
        self.name = source_name
        self.base = base_rate
        self.factor = spike_factor
        self._tick = 0

    def _phase_offset(self, topic: str) -> float:
        h = int(hashlib.sha1(f"{self.name}:{topic}".encode()).hexdigest()[:8], 16)
        return (h % 1440) / 1440 * 2 * math.pi

    def read(self, topic: str) -> float:
        """
        Generate one synthetic activity observation.

        Args:
            topic: Topic used to create a deterministic signal pattern.

        Returns:
            float: Simulated activity measurement for the current tick.
        """
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
    """
    Asynchronous Mastodon data provider for social activity extraction.

    Uses authenticated full-text search to collect recent topic activity,
    extracts representative samples, and tracks provider reliability. When API
    access fails repeatedly, the source automatically degrades to deterministic
    mock generation to maintain pipeline availability.
    """

    def __init__(self):
        self._instance = MASTODON_INSTANCE
        self._token = os.getenv("MASTODON_ACCESS_TOKEN", "")
        self._available = bool(self._token)
        self._consecutive_failures = 0
        self._max_failures_before_mock = 5
        self._zero_result_ticks: Dict[str, int] = {}
        self._mock = MockSource("mastodon", base_rate=10.0, spike_factor=6.0)
        self._last_samples: Dict[str, List[dict]] = {}

        if self._available:
            log.info(
                f"Mastodon source: authenticated search on {self._instance} (async)"
            )
        else:
            log.warning(
                "Mastodon source: MASTODON_ACCESS_TOKEN not set — using mock data."
            )

    def get_samples(self, topic: str) -> List[dict]:
        return self._last_samples.get(topic, [])

    async def read(self, client: httpx.AsyncClient, topic: str) -> tuple[float, str]:
        """
        Retrieve Mastodon activity metrics and representative samples.

        Queries recent statuses, applies UTC-based recency filtering, extracts
        cleaned text samples, and returns both the activity volume and data provenance.

        Args:
            client: Shared asynchronous HTTP client.
            topic: Topic query used for social search.

        Returns:
            tuple[float, str]: Activity measurement and source type ("live" or "mock").
        """
        if not self._available:
            return self._mock.read(topic), "mock"

        try:
            resp = await client.get(
                f"https://{self._instance}/api/v2/search",
                params={
                    "q": topic,
                    "type": "statuses",
                    "limit": 40,
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

            cutoff = time.time() - 24 * 3600
            recent = []
            for p in posts:
                ts = _parse_mastodon_timestamp(p.get("created_at", ""))
                if ts >= cutoff:
                    recent.append((p, ts))

            if not recent:
                self._zero_result_ticks[topic] = (
                    self._zero_result_ticks.get(topic, 0) + 1
                )
                return 0.0, "live"

            self._zero_result_ticks[topic] = 0

            samples = []
            for p, ts in recent[:8]:
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
            return float(len(recent)), "live"

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
                    f"failures — switching to mock data for remainder of session."
                )
            self._available = False


class BlueskySource:
    """
    Asynchronous Bluesky activity provider with authentication recovery.

    Handles session creation, token refresh, post retrieval, and automatic
    fallback to synthetic signals when credentials are unavailable or API access
    fails.
    """

    def __init__(self):
        self._session_token: Optional[str] = None
        self._last_samples: Dict[str, List[dict]] = {}
        self._handle = os.getenv("BLUESKY_HANDLE", "")
        self._password = os.getenv("BLUESKY_APP_PASSWORD", "")
        self._available = bool(self._handle and self._password)
        self._mock = MockSource("bluesky", base_rate=8.0, spike_factor=5.0)
        if not self._available:
            log.info("Bluesky source: no credentials — mock active")

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

    async def read(self, client: httpx.AsyncClient, topic: str) -> tuple[float, str]:
        if not self._available:
            return self._mock.read(topic), "mock"

        if not self._session_token:
            if not await self._authenticate(client):
                return self._mock.read(topic), "mock"

        try:
            resp = await client.get(
                "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
                params={"q": topic, "limit": 25},
                headers={"Authorization": f"Bearer {self._session_token}"},
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code == 401:

                self._session_token = None
                if not await self._authenticate(client):
                    return self._mock.read(topic), "mock"
                resp = await client.get(
                    "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
                    params={"q": topic, "limit": 25},
                    headers={"Authorization": f"Bearer {self._session_token}"},
                    timeout=HTTP_TIMEOUT,
                )
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("posts", [])

            samples = []
            for p in posts[:8]:
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
                        "timestamp": time.time(),
                    }
                )
            self._last_samples[topic] = samples
            return float(len(posts)), "live"

        except Exception as e:
            log.debug(f"Bluesky read error: {e}")
            return self._mock.read(topic), "mock"


class TrendsSource:
    """
    Google Trends activity provider integrated into the async pipeline.

    Because pytrends performs blocking network operations, requests are delegated
    to an IO executor to prevent blocking the detector event loop. Results are
    cached to reduce external API pressure and improve tick latency.
    """

    def __init__(self):
        self._available = False
        try:
            from pytrends.request import TrendReq

            self._pytrends = TrendReq(
                hl="en-US", tz=0, timeout=(5, 10), retries=1, backoff_factor=0.5
            )
            self._available = True
            log.info("Trends source: pytrends ready")
        except ImportError:
            log.info("Trends source: pytrends not installed — mock active")
        self._cache: Dict[str, tuple] = {}
        self._mock = MockSource("trends", base_rate=45.0, spike_factor=3.0)

    def _blocking_read(self, topic: str) -> Optional[float]:
        """
        Execute synchronous Google Trends retrieval.

        This method is isolated from the async pipeline and executed through the
        shared IO executor.

        Args:
            topic: Search term used for trend retrieval.

        Returns:
            Optional[float]: Latest normalized interest score when available.
        """
        try:
            self._pytrends.build_payload([topic], timeframe="now 1-H")
            df = self._pytrends.interest_over_time()
            if df.empty or topic not in df.columns:
                return None
            return float(df[topic].iloc[-1])
        except Exception as e:
            log.debug(f"Trends read error for {topic}: {e}")
            return None

    async def read(self, topic: str) -> tuple[float, str]:
        if not self._available:
            return self._mock.read(topic), "mock"

        now = time.time()
        if topic in self._cache:
            val, exp = self._cache[topic]
            if now < exp:
                return val, "live"

        loop = asyncio.get_running_loop()
        val = await loop.run_in_executor(IO_EXECUTOR, self._blocking_read, topic)
        if val is None:
            return self._mock.read(topic), "mock"
        self._cache[topic] = (val, now + 300)
        return val, "live"


class WikipediaSource:
    """
    Wikipedia edit activity provider used as a reference-source signal.

    Tracks changes around monitored topics and provides a lower-volume knowledge
    base activity signal that complements social platform measurements.
    """

    def __init__(self):
        self._mock = MockSource("wikipedia", base_rate=2.5, spike_factor=4.0)

    async def read(self, client: httpx.AsyncClient, topic: str) -> tuple[float, str]:
        try:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "recentchanges",
                    "rcsearch": topic,
                    "rclimit": 50,
                    "rctype": "edit",
                    "rcprop": "title|timestamp",
                    "format": "json",
                },
                headers={"User-Agent": "wc2026-narrative/1.0"},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            edits = resp.json().get("query", {}).get("recentchanges", [])
            return float(len(edits)) / 60.0, "live"
        except Exception as e:
            log.debug(f"Wikipedia read error: {e}")
            return self._mock.read(topic), "mock"


class NarrativeSpikeDetector:
    """
    Multi-source anomaly detection engine for narrative intelligence.

    Aggregates social, search, and reference signals into time-series feature
    vectors and applies per-topic IsolationForest models to identify unusual
    activity patterns.

    The detector maintains rolling historical windows, trains independent anomaly
    models per topic, attributes spikes back to contributing sources, and exposes
    results for downstream narrative generation.
    """

    def __init__(self, topics: List[str] = TOPICS):
        self.topics = topics
        self._tick_count = 0
        self._windows: Dict[str, deque] = {
            t: deque(maxlen=WINDOW_TICKS) for t in topics
        }
        self._models: Dict[str, Optional[IsolationForest]] = {t: None for t in topics}
        self._last_spike: Dict[str, float] = {}

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

    async def _read_topic(self, client: httpx.AsyncClient, topic: str) -> SignalPoint:
        """Fetch all four sources for one topic CONCURRENTLY (audit H3)."""
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

    def _fit_model(self, topic: str) -> None:
        """
        Train an IsolationForest anomaly model for a monitored topic.

        Uses the topic's rolling historical signal window as the training distribution.
        Models are retrained periodically to adapt to changing baseline activity.

        Args:
            topic: Topic whose historical window should train the model.
        """
        window = self._windows[topic]
        if len(window) < MIN_TICKS:
            return
        X = np.array([p.to_vector() for p in window])
        model = IsolationForest(
            contamination=CONTAMINATION, random_state=42, n_estimators=100
        )
        model.fit(X)
        self._models[topic] = model

    def _score(self, topic: str, point: SignalPoint) -> Optional[float]:
        """
        Calculate anomaly score for a new signal observation.

        Uses the topic-specific IsolationForest model to determine whether the latest
        multi-source activity pattern deviates from historical behavior.

        Args:
            topic: Topic associated with the signal.
            point: Current signal observation.

        Returns:
            Optional[float]: IsolationForest decision score, or None when the model
                has insufficient training history.
        """
        model = self._models.get(topic)
        if model is None:
            return None
        vec = point.to_vector().reshape(1, -1)
        return float(model.decision_function(vec)[0])

    def _make_spike(
        self, topic: str, point: SignalPoint, score: float
    ) -> NarrativeSpike:
        """
        Convert an anomaly detection result into a narrative spike object.

        Performs source attribution by comparing current activity against rolling
        baselines, generates a human-readable summary, and normalizes anomaly
        magnitude into a severity score.

        Args:
            topic: Topic associated with the anomaly.
            point: Signal observation triggering detection.
            score: Raw IsolationForest anomaly score.

        Returns:
            NarrativeSpike: Structured anomaly representation.
        """
        severity = min(1.0, max(0.0, (-score - SCORE_THRESH) / 0.5))
        sources = {
            "mastodon": point.mastodon,
            "bluesky": point.bluesky,
            "trends": point.trends,
            "wikipedia": point.wikipedia,
        }
        window = self._windows[topic]
        if len(window) >= 10:
            means = {
                s: np.mean([getattr(p, s) for p in window])
                for s in ("mastodon", "bluesky", "trends", "wikipedia")
            }
            driving = [
                s
                for s, v in sources.items()
                if v > means[s] * 1.2 and v > MIN_ACTIVITY_FLOOR[s]
            ]
        else:
            driving = list(sources.keys())

        raw = f"{topic}:{point.tick}"
        spike_id = hashlib.sha1(raw.encode()).hexdigest()[:12]

        parts = []
        if point.mastodon > 15:
            parts.append(f"Mastodon {point.mastodon:.0f} posts/min")
        if point.bluesky > 15:
            parts.append(f"Bluesky {point.bluesky:.0f} mentions/min")
        if point.trends > 70:
            parts.append(f"Trends index {point.trends:.0f}")
        if point.wikipedia > 3:
            parts.append(f"Wikipedia {point.wikipedia:.1f} edits/min")
        if not parts:
            parts.append(f"signal {severity:.0%} above 72h baseline")
        summary = f"{topic} spike — " + ", ".join(parts)

        return NarrativeSpike(
            spike_id=spike_id,
            topic=topic,
            tick=point.tick,
            severity=severity,
            sources=sources,
            source_names=driving,
            summary=summary,
            timestamp=point.timestamp,
            data_sources=point.data_sources,
        )

    def trending(self, top_n: int = 12) -> List[dict]:
        """
        Generate a ranked view of currently active narrative topics.

        Unlike anomaly detection, this ranking focuses on relative activity lift
        against each topic's rolling baseline to identify broadly trending subjects.

        Args:
            top_n: Maximum number of topics returned.

        Returns:
            List[dict]: Ranked topic activity summaries containing severity,
                source attribution, and provenance metadata.
        """
        now = time.time()
        rows: list = []
        for topic in self.topics:
            window = self._windows.get(topic)
            if not window:
                continue
            cur = window[-1]
            sources = {
                "mastodon": cur.mastodon,
                "bluesky": cur.bluesky,
                "trends": cur.trends,
                "wikipedia": cur.wikipedia,
            }
            driving: List[str] = []
            activity = sum(sources[s] / MIN_ACTIVITY_FLOOR[s] for s in sources)
            if len(window) >= 5:
                lifts = []
                for s in ("mastodon", "bluesky", "trends", "wikipedia"):
                    m = float(np.mean([getattr(p, s) for p in window]))
                    lifts.append(max(0.0, (sources[s] - m) / (m + 1e-6)))
                    if sources[s] > m * 1.2 and sources[s] > MIN_ACTIVITY_FLOOR[s]:
                        driving.append(s)
                severity = float(min(1.0, np.mean(lifts)))
            else:
                severity = float(min(1.0, max(0.0, (activity - 4.0) / 4.0)))
                driving = [
                    s for s in sources if sources[s] > MIN_ACTIVITY_FLOOR[s] * 1.5
                ]

            parts = []
            if cur.mastodon > 15:
                parts.append(f"Mastodon {cur.mastodon:.0f} posts/min")
            if cur.bluesky > 15:
                parts.append(f"Bluesky {cur.bluesky:.0f} mentions/min")
            if cur.trends > 70:
                parts.append(f"Trends index {cur.trends:.0f}")
            if cur.wikipedia > 3:
                parts.append(f"Wikipedia {cur.wikipedia:.1f} edits/min")
            if not parts:
                parts.append(f"{severity:.0%} above 72h baseline")
            summary = f"{topic} — " + ", ".join(parts)

            rows.append(
                (
                    severity,
                    activity,
                    {
                        "spike_id": f"trend-{hashlib.sha1(topic.encode()).hexdigest()[:10]}",
                        "topic": topic,
                        "tick": self._tick_count,
                        "severity": round(severity, 3),
                        "sources": {k: round(v, 2) for k, v in sources.items()},
                        "source_names": driving,
                        "summary": summary,
                        "timestamp": now,
                        "arc": None,
                        "is_spike": (now - self._last_spike.get(topic, 0)) < 600,
                        "data_sources": cur.data_sources,
                    },
                )
            )
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        return [r[2] for r in rows[:top_n]]

    async def tick(self, loop: asyncio.AbstractEventLoop) -> List[NarrativeSpike]:
        """
        Execute one complete anomaly detection cycle.

        The tick concurrently collects signals from all providers, updates rolling
        feature windows, retrains anomaly models when required, scores new activity,
        and returns newly detected narrative spikes.

        The pipeline is designed for continuous operation with asynchronous IO,
        executor-backed blocking workloads, provider fallbacks, and per-topic anomaly
        models.

        Args:
            loop: Async event loop used for background model training.

        Returns:
            List[NarrativeSpike]: Newly detected narrative anomalies ordered by
                severity.
        """
        self._tick_count += 1
        now = time.time()

        async with httpx.AsyncClient() as client:
            points: List[SignalPoint] = await asyncio.gather(
                *[self._read_topic(client, topic) for topic in self.topics]
            )

        spikes: List[NarrativeSpike] = []
        for topic, point in zip(self.topics, points):
            try:
                self._windows[topic].append(point)

                if self._tick_count % 30 == 1 or self._models[topic] is None:
                    await loop.run_in_executor(IO_EXECUTOR, self._fit_model, topic)

                score = self._score(topic, point)
                if score is None:
                    continue

                cooldown = self._last_spike.get(topic, 0)
                if score < SCORE_THRESH and (now - cooldown) > 300:
                    spike = self._make_spike(topic, point, score)
                    spikes.append(spike)
                    self._last_spike[topic] = now
                    log.info(
                        f"Spike detected: {topic} score={score:.3f} "
                        f"severity={spike.severity:.2f} id={spike.spike_id}"
                    )
            except Exception as e:
                log.warning(
                    f"tick post-processing error for {topic}: {e}", exc_info=True
                )

        spikes.sort(key=lambda s: s.severity, reverse=True)
        return spikes[:MAX_SPIKES_PER_TICK]


_detector: Optional[NarrativeSpikeDetector] = None


def get_detector() -> NarrativeSpikeDetector:
    """
    Return the singleton narrative spike detection engine.

    Maintains one shared detector instance so rolling windows, trained models,
    and provider state persist across API requests and worker cycles.

    Returns:
        NarrativeSpikeDetector: Initialized global detector instance.
    """
    global _detector
    if _detector is None:
        _detector = NarrativeSpikeDetector()
    return _detector
