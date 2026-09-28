"""
Surge detection for narrative topics. Pure: no network, no sklearn.

Each source value is transformed (log1p(rate/hr) for counts, log(lift) for
Trends) and scored as a one-sided robust z against its own 3h live baseline.
A topic alerts when >=2 live sources surge together, or one surges hard.
Mock/unavailable values arrive as None and are never scored or stored.

Also holds the window-counting helpers the live sources use, so the offline
eval runs the exact same maths.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

SOURCES = ("mastodon", "bluesky", "trends", "wikipedia")

BASELINE_TICKS = 180  # 3h of 60s ticks
MIN_BASELINE = 60  # live points before a source is scored
GAP = 5  # most recent points left out of the baseline
MIN_SCALE = 0.35  # log units
Z_CORROB = 3.0
Z_SOLO = 6.0
Z_EXIT = 1.5
EXIT_TICKS = 5
MAX_SPIKES_PER_TICK = 3
# A zero reading from a source whose baseline median is at least this rate
# (per hour) is treated as an outage: not scored, not stored.
OUTAGE_ZERO_RATE = 30.0
# Trends index (0-100) the earlier part of the hour must reach for a lift.
TRENDS_MIN_MEDIAN = 3.0

# Raw floor a surging source must also clear (per hour; lift for trends).
MIN_COUNT = {
    "mastodon": 6.0,
    "bluesky": 20.0,
    "trends": 1.5,
    "wikipedia": 3.0,
}


# ---------------------------------------------------------------- signal maths


def window_rate(
    timestamps: Iterable[float], now: float, window_s: float, page_limit: int
) -> float:
    """Items per hour in the trailing window.

    If the page came back full and every item is inside the window, the true
    count is unknown, so the rate is estimated from the span the page covers.
    """
    ts = [t for t in timestamps if t is not None]
    recent = [t for t in ts if now - window_s <= t <= now + 60]
    if not recent:
        return 0.0
    if len(ts) >= page_limit and len(recent) == len(ts):
        span = max(now - min(recent), 60.0)
        return len(recent) / span * 3600.0
    return len(recent) / window_s * 3600.0


def trends_lift(values: Sequence[float], recent_n: int = 5) -> Optional[float]:
    """Mean of the last `recent_n` points over the median of the earlier ones.

    Computed within one Trends response, so Google's per-query 0-100 rescaling
    cancels out.
    """
    vals = [float(v) for v in values]
    if len(vals) < recent_n + 10:
        return None
    recent, earlier = vals[-recent_n:], vals[:-recent_n]
    med = float(np.median(earlier))
    if med < TRENDS_MIN_MEDIAN:  # too sparse to measure a lift
        return None
    return float(np.mean(recent)) / med


def transform(source: str, value: float) -> float:
    if source == "trends":
        return math.log(max(value, 1e-3))
    return math.log1p(max(value, 0.0))


# ---------------------------------------------------------------- scorer


@dataclass
class ScoreResult:
    topic: str
    z: Dict[str, Optional[float]]
    raw: Dict[str, Optional[float]]
    surging: List[str]
    severity: float
    candidate: bool  # alert rule met and topic not already in an episode
    warming: bool
    in_episode: bool = False
    solo: bool = False


@dataclass
class _TopicState:
    hist: Dict[str, deque] = field(
        default_factory=lambda: {s: deque(maxlen=BASELINE_TICKS) for s in SOURCES}
    )
    in_episode: bool = False
    quiet_ticks: int = 0


def _severity(zs: Dict[str, Optional[float]], solo: bool) -> float:
    pos = sorted((max(z, 0.0) for z in zs.values() if z is not None), reverse=True)
    if not pos:
        return 0.0
    if solo:
        combined = pos[0]
    else:
        top2 = (pos + [0.0, 0.0])[:2]
        combined = sum(top2) / 2.0
    return float(min(1.0, max(0.0, (combined - 3.0) / 5.0)))


class SpikeScorer:
    def __init__(
        self,
        z_corrob: float = Z_CORROB,
        z_solo: float = Z_SOLO,
        z_exit: float = Z_EXIT,
        exit_ticks: int = EXIT_TICKS,
        min_count: Optional[Dict[str, float]] = None,
    ):
        self.z_corrob = z_corrob
        self.z_solo = z_solo
        self.z_exit = z_exit
        self.exit_ticks = exit_ticks
        self.min_count = dict(MIN_COUNT if min_count is None else min_count)
        self._state: Dict[str, _TopicState] = {}
        self.latest: Dict[str, ScoreResult] = {}

    def _topic(self, topic: str) -> _TopicState:
        st = self._state.get(topic)
        if st is None:
            st = self._state[topic] = _TopicState()
        return st

    @staticmethod
    def _z(hist: deque, x: float) -> Optional[float]:
        if len(hist) < MIN_BASELINE:
            return None
        base = np.fromiter(hist, dtype=float)[:-GAP]
        med = float(np.median(base))
        mad = float(np.median(np.abs(base - med)))
        return (x - med) / max(1.4826 * mad, MIN_SCALE)

    @staticmethod
    def _busy(hist: deque) -> bool:
        if len(hist) < MIN_BASELINE:
            return False
        return float(np.median(np.fromiter(hist, dtype=float))) >= math.log1p(
            OUTAGE_ZERO_RATE
        )

    def update(self, topic: str, values: Dict[str, Optional[float]]) -> ScoreResult:
        """Score one tick; `values` maps source -> raw value, None if mock."""
        st = self._topic(topic)
        zs: Dict[str, Optional[float]] = {}
        raw: Dict[str, Optional[float]] = {}
        for s in SOURCES:
            v = values.get(s)
            raw[s] = v
            if v is None:
                zs[s] = None
                continue
            if v <= 0 and s != "trends" and self._busy(st.hist[s]):
                zs[s] = None
                continue
            x = transform(s, v)
            zs[s] = self._z(st.hist[s], x)
            st.hist[s].append(x)

        surging = [
            s
            for s in SOURCES
            if zs[s] is not None
            and zs[s] >= self.z_corrob
            and raw[s] >= self.min_count[s]
        ]
        solo = len(surging) < 2 and any(zs[s] >= self.z_solo for s in surging)
        alert = len(surging) >= 2 or solo

        if st.in_episode:
            if any(z is not None and z >= self.z_exit for z in zs.values()):
                st.quiet_ticks = 0
            else:
                st.quiet_ticks += 1
                if st.quiet_ticks >= self.exit_ticks:
                    st.in_episode = False
                    st.quiet_ticks = 0

        warming = all(z is None for z in zs.values())
        res = ScoreResult(
            topic=topic,
            z=zs,
            raw=raw,
            surging=sorted(surging, key=lambda s: -zs[s]),
            severity=0.0 if warming else _severity(zs, solo),
            candidate=alert and not st.in_episode,
            warming=warming,
            in_episode=st.in_episode,
            solo=solo,
        )
        self.latest[topic] = res
        return res

    def mark_alerted(self, topic: str) -> None:
        st = self._topic(topic)
        st.in_episode = True
        st.quiet_ticks = 0
        if topic in self.latest:
            self.latest[topic].in_episode = True
            self.latest[topic].candidate = False

    def emit(
        self, results: Iterable[ScoreResult], cap: int = MAX_SPIKES_PER_TICK
    ) -> List[ScoreResult]:
        """Start episodes for the top `cap` candidates; the rest wait a tick."""
        cands = sorted(
            (r for r in results if r.candidate),
            key=lambda r: (r.severity, max(z or 0.0 for z in r.z.values())),
            reverse=True,
        )[:cap]
        for r in cands:
            self.mark_alerted(r.topic)
        return cands

    def warmup_remaining(self, topic: Optional[str] = None) -> int:
        """Ticks until a live source of `topic` (or any topic) can be scored."""
        topics = [topic] if topic is not None else list(self._state)
        if not topics:
            return MIN_BASELINE
        best = MIN_BASELINE
        for t in topics:
            st = self._state.get(t)
            if st is None:
                continue
            have = max(len(h) for h in st.hist.values())
            best = min(best, max(0, MIN_BASELINE - have))
        return best

    def export_state(self) -> dict:
        return {
            "version": 1,
            "topics": {
                t: {
                    "hist": {s: list(h) for s, h in st.hist.items()},
                    "in_episode": st.in_episode,
                    "quiet_ticks": st.quiet_ticks,
                }
                for t, st in self._state.items()
            },
        }

    def load_state(self, state: dict) -> None:
        if not state or state.get("version") != 1:
            return
        for t, d in state.get("topics", {}).items():
            st = self._topic(t)
            for s in SOURCES:
                st.hist[s] = deque(
                    (float(x) for x in d.get("hist", {}).get(s, [])),
                    maxlen=BASELINE_TICKS,
                )
            st.in_episode = bool(d.get("in_episode", False))
            st.quiet_ticks = int(d.get("quiet_ticks", 0))
