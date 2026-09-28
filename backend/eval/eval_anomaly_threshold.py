"""Offline evaluation of the narrative surge detector.

Simulates per-minute post/edit arrivals and Trends intensity for N topic-days,
pushes them through the same window-counting helpers the live sources use
(window_rate, trends_lift) and then through the production SpikeScorer.

Labels are synthetic and event-level:
  positive  multi-source surge: 2-4 sources, x2-8, 3-min ramp, 5-30 min plateau
  negative  single-source burst, outage (source -> 0), source switching to
            mock, slow 4h rise to x3 across all sources

Thresholds are tuned on dev seeds; every reported number comes from held-out
seeds, with Wilson / exact Poisson CIs. Baselines share the same robust z:
any single source (the corroboration ablation) and each source on its own.

No network, no credentials. Run from backend/:
    PYTHONPATH=. python eval/eval_anomaly_threshold.py --json anomaly_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import chi2

from agents import spike_scorer as ss
from agents.narrative_spike_detector import (
    BLUESKY_PAGE,
    BLUESKY_WINDOW_S,
    MASTODON_PAGE,
    MASTODON_WINDOW_S,
    TRENDS_PER_TICK,
    WIKI_CACHE_S,
    WIKI_PAGE,
    WIKI_WINDOW_S,
    MockSource,
)
from agents.spike_scorer import SOURCES, SpikeScorer, trends_lift, window_rate

WARMUP = 240  # ticks before the scored day
DAY = 1440
SLOT = 120
MATCH_TAIL = 10  # an alert up to 10 min after an event still counts
Z_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
TEST_SEED0 = 1000  # held-out seeds start here; tuning never sees them
N_TOPICS = 26  # Trends refresh cadence follows the real round-robin

COUNT_SRC = {
    "mastodon": (MASTODON_WINDOW_S, MASTODON_PAGE, 1),
    "bluesky": (BLUESKY_WINDOW_S, BLUESKY_PAGE, 1),
    "wikipedia": (WIKI_WINDOW_S, WIKI_PAGE, WIKI_CACHE_S // 60),
}
MOCK_BASE = {"mastodon": 6.0, "bluesky": 200.0, "trends": 1.0, "wikipedia": 1.0}


@dataclass
class Event:
    kind: str  # surge | burst | outage | mock | slow_rise
    start: int
    end: int
    sources: List[str]
    factor: float = 1.0


@dataclass
class TopicDay:
    values: List[Dict[str, float]] = field(default_factory=list)
    prov: List[Dict[str, str]] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)


# ---------------------------------------------------------------- simulation


def _schedule(rng) -> List[Event]:
    slots = list(rng.permutation(DAY // SLOT))
    # slow rise takes 3 contiguous slots
    rise_at = int(rng.integers(0, DAY // SLOT - 2))
    rise_slots = {rise_at, rise_at + 1, rise_at + 2}
    free = [s for s in slots if s not in rise_slots]
    events = [
        Event("slow_rise", WARMUP + rise_at * SLOT, WARMUP + (rise_at + 3) * SLOT, list(SOURCES), 3.0)
    ]

    def at(slot):
        return WARMUP + slot * SLOT + int(rng.integers(10, 40))

    for slot in free[:3]:
        k = int(rng.integers(2, 5))
        start = at(slot)
        plateau = int(rng.integers(5, 31))
        events.append(
            Event("surge", start, start + 3 + plateau, list(rng.choice(SOURCES, k, replace=False)), float(rng.uniform(2, 8)))
        )
    s = at(free[3])
    events.append(Event("burst", s, s + 3 + int(rng.integers(5, 31)), [str(rng.choice(SOURCES))], float(rng.uniform(3, 8))))
    s = at(free[4])
    events.append(Event("outage", s, s + int(rng.integers(30, 81)), [str(rng.choice(SOURCES))]))
    s = at(free[5])
    events.append(Event("mock", s, s + int(rng.integers(60, 81)), [str(rng.choice(SOURCES))]))
    return events


def _multiplier(events: List[Event], src: str, t: np.ndarray) -> np.ndarray:
    m = np.ones_like(t, dtype=float)
    for e in events:
        if src not in e.sources:
            continue
        if e.kind in ("surge", "burst"):
            ramp = np.clip((t - e.start + 1) / 3.0, 0, 1)
            inside = (t >= e.start) & (t < e.end)
            m = np.where(inside, m * (1 + (e.factor - 1) * ramp), m)
        elif e.kind == "slow_rise":
            ramp_end = e.start + 240
            f = np.where(t < ramp_end, 1 + (e.factor - 1) * (t - e.start) / 240.0, e.factor)
            m = np.where((t >= e.start) & (t < e.end), m * f, m)
        elif e.kind == "outage":
            m = np.where((t >= e.start) & (t < e.end), 0.0, m)
    return m


def simulate(seed: int) -> TopicDay:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = WARMUP + DAY
    t = np.arange(n)
    phase = rng.uniform(0, 2 * np.pi)
    cycle = 1 + 0.6 * np.sin(2 * np.pi * t / DAY + phase)
    base = {
        "mastodon": rng.uniform(3, 20),
        "bluesky": rng.uniform(60, 600),
        "wikipedia": rng.uniform(0.5, 6),
    }
    events = _schedule(rng)
    day = TopicDay(events=events)

    series: Dict[str, List[Optional[float]]] = {}
    for src, (window_s, page, every) in COUNT_SRC.items():
        rate_min = base[src] / 60.0 * cycle * _multiplier(events, src, t)
        stamps = [
            np.sort(m * 60 + rng.uniform(0, 60, rng.poisson(r)))
            for m, r in enumerate(rate_min)
        ]
        wmin = window_s // 60
        offset = int(rng.integers(0, every))
        out, last = [], None
        for m in range(n):
            if last is None or (m + offset) % every == 0:
                now = (m + 1) * 60.0
                recent = np.concatenate(stamps[max(0, m - wmin + 1) : m + 1])[::-1][:page]
                last = window_rate(recent.tolist(), now, window_s, page)
            out.append(last)
        series[src] = out

    # Trends: noisy per-minute intensity, rescaled 0-100 per query, refreshed
    # on the production round-robin cadence.
    intensity = cycle * _multiplier(events, "trends", t)
    every = max(1, N_TOPICS // TRENDS_PER_TICK)
    offset = int(rng.integers(0, every))
    out, last = [], None
    for m in range(n):
        if m >= 61 and (last is None or (m + offset) % every == 0):
            win = intensity[m - 60 : m + 1] * rng.lognormal(0, 0.1, 61)
            idx = np.round(win / (win.max() or 1.0) * 100)
            last = trends_lift(idx[:-1].tolist())  # last point is partial
        out.append(last)
    series["trends"] = out

    mock = {s: MockSource(s, MOCK_BASE[s], 4.0) for s in SOURCES}
    for m in range(n):
        vals, prov = {}, {}
        for s in SOURCES:
            is_mock = series[s][m] is None or any(
                e.kind == "mock" and s in e.sources and e.start <= m < e.end for e in events
            )
            vals[s] = mock[s].read("sim") if is_mock else float(series[s][m])
            prov[s] = "mock" if is_mock else "live"
        day.values.append(vals)
        day.prov.append(prov)
    return day


# ---------------------------------------------------------------- detectors


def run_detector(
    day: TopicDay, z_corrob: float, z_solo: float, sources: Sequence[str] = SOURCES
) -> List[int]:
    """Production SpikeScorer, optionally limited to a subset of sources."""
    sc = SpikeScorer(z_corrob=z_corrob, z_solo=z_solo)
    alerts = []
    for m, (vals, prov) in enumerate(zip(day.values, day.prov)):
        live = {
            s: (vals[s] if prov[s] == "live" and s in sources else None) for s in SOURCES
        }
        if sc.emit([sc.update("topic", live)]) and m >= WARMUP:
            alerts.append(m)
    return alerts


def run_scorer(day: TopicDay, z_corrob: float) -> List[int]:
    return run_detector(day, z_corrob, ss.Z_SOLO)


# Each detector maps a threshold z to run_detector kwargs. The baselines use
# the same robust z, so the comparison isolates cross-source corroboration.
DETECTORS: Dict[str, Callable[[float], dict]] = {
    "production (corroborated)": lambda z: {"z_corrob": z, "z_solo": ss.Z_SOLO},
    "any single source (no corroboration)": lambda z: {"z_corrob": z, "z_solo": z},
    **{
        f"{s} only": (lambda z, s=s: {"z_corrob": z, "z_solo": z, "sources": (s,)})
        for s in SOURCES
    },
}


# ---------------------------------------------------------------- metrics


def wilson(k: int, n: int, z: float = 1.96) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def poisson_rate_ci(k: int, exposure: float) -> List[float]:
    """Exact (Garwood) 95% CI for a count, as a rate per unit exposure."""
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return [round(lo / exposure, 3), round(hi / exposure, 3)]


def _in(e: Event, m: int) -> bool:
    return e.start <= m <= e.end + MATCH_TAIL


def score_day(day: TopicDay, alerts: List[int]) -> dict:
    surges = [e for e in day.events if e.kind == "surge"]
    detected, latency = [], []
    for e in surges:
        hits = [a for a in alerts if _in(e, a)]
        detected.append(bool(hits))
        if hits:
            latency.append(hits[0] - e.start)
    true_alerts = sum(any(_in(e, a) for e in surges) for a in alerts)
    by_type: Dict[str, int] = {}
    for a in alerts:
        if any(_in(e, a) for e in surges):
            continue
        kind = next((e.kind for e in day.events if e.kind != "surge" and _in(e, a)), "background")
        by_type[kind] = by_type.get(kind, 0) + 1
    return {
        "events": surges,
        "detected": detected,
        "latency": latency,
        "alerts": len(alerts),
        "true_alerts": true_alerts,
        "false_by_type": by_type,
    }


def aggregate(per_day: List[dict]) -> dict:
    detected = [d for p in per_day for d in p["detected"]]
    latency = [x for p in per_day for x in p["latency"]]
    alerts = sum(p["alerts"] for p in per_day)
    true_alerts = sum(p["true_alerts"] for p in per_day)
    false_alerts = alerts - true_alerts
    false_by: Dict[str, int] = {k: 0 for k in ("burst", "outage", "mock", "slow_rise", "background")}
    for p in per_day:
        for k, v in p["false_by_type"].items():
            false_by[k] = false_by.get(k, 0) + v
    recall = sum(detected) / len(detected) if detected else 0.0
    precision = true_alerts / alerts if alerts else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "topic_days": len(per_day),
        "events": len(detected),
        "events_detected": int(sum(detected)),
        "event_recall": round(recall, 4),
        "event_recall_ci95": wilson(int(sum(detected)), len(detected)),
        "alerts": alerts,
        "true_alerts": true_alerts,
        "alert_precision": round(precision, 4),
        "alert_precision_ci95": wilson(true_alerts, alerts),
        "f1": round(f1, 4),
        "false_alerts_per_topic_day": round(false_alerts / len(per_day), 3),
        "false_alerts_per_topic_day_ci95": poisson_rate_ci(false_alerts, len(per_day)),
        "median_latency_min": float(np.median(latency)) if latency else None,
        "false_alerts_by_type": false_by,
    }


def recall_by_bucket(per_day: List[dict]) -> dict:
    buckets: Dict[str, List[bool]] = {}
    for p in per_day:
        for e, hit in zip(p["events"], p["detected"]):
            mag = "x2-4" if e.factor < 4 else "x4-8"
            buckets.setdefault(f"{len(e.sources)} sources, {mag}", []).append(hit)
    return {
        k: {"n": len(v), "recall": round(float(np.mean(v)), 3), "ci95": wilson(int(sum(v)), len(v))}
        for k, v in sorted(buckets.items())
    }


def evaluate(days: List[TopicDay], kwargs: dict) -> List[dict]:
    return [score_day(d, run_detector(d, **kwargs)) for d in days]


# ---------------------------------------------------------------- main


def _row(label: str, m: dict) -> None:
    lat = m["median_latency_min"]
    rc, pc, fc = m["event_recall_ci95"], m["alert_precision_ci95"], m["false_alerts_per_topic_day_ci95"]
    print(
        f"{label:<50} recall {m['event_recall']:.2f} [{rc[0]:.2f},{rc[1]:.2f}]  "
        f"prec {m['alert_precision']:.2f} [{(pc[0] or 0):.2f},{(pc[1] or 0):.2f}]  "
        f"FA/day {m['false_alerts_per_topic_day']:.2f} [{fc[0]:.2f},{fc[1]:.2f}]  "
        f"lat {'-' if lat is None else lat}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-seeds", type=int, default=20, help="topic-days for threshold tuning")
    ap.add_argument("--test-seeds", type=int, default=100, help="held-out topic-days")
    ap.add_argument("--json", help="write report to this path")
    args = ap.parse_args()

    dev = [simulate(s) for s in range(args.dev_seeds)]
    test = [simulate(TEST_SEED0 + s) for s in range(args.test_seeds)]

    print(
        f"\nNarrative surge detector: thresholds tuned on {len(dev)} dev topic-days "
        f"(seeds 0-{len(dev) - 1}), reported on {len(test)} held-out topic-days "
        f"(seeds {TEST_SEED0}-{TEST_SEED0 + len(test) - 1})\n"
    )

    report_detectors = {}
    prod_test_days = None
    for name, make in DETECTORS.items():
        sweep = {f"{z:.1f}": aggregate(evaluate(dev, make(z))) for z in Z_GRID}
        # Best dev F1; ties go to the higher (more conservative) threshold.
        best_z = max(Z_GRID, key=lambda z: (sweep[f"{z:.1f}"]["f1"], z))
        entry = {"dev_sweep": sweep, "tuned_z": best_z, "test": aggregate(evaluate(test, make(best_z)))}
        if name.startswith("production"):
            prod_test_days = evaluate(test, make(ss.Z_CORROB))
            entry["test_at_shipped_z"] = {"z": ss.Z_CORROB, **aggregate(prod_test_days)}
        report_detectors[name] = entry

    for name, e in report_detectors.items():
        _row(f"{name} (z={e['tuned_z']:.1f}, dev-tuned)", e["test"])
        if "test_at_shipped_z" in e:
            _row(f"{name} (z={ss.Z_CORROB:.1f}, shipped)", e["test_at_shipped_z"])

    buckets = recall_by_bucket(prod_test_days)
    print("\nProduction (shipped z) recall by event type, held-out:")
    for k, v in buckets.items():
        print(f"  {k:<22} n={v['n']:<3} recall={v['recall']} CI={v['ci95']}")

    if args.json:
        report = {
            "data": "synthetic per-minute arrivals with injected, labelled events",
            "split": {
                "dev_seeds": [0, len(dev) - 1],
                "test_seeds": [TEST_SEED0, TEST_SEED0 + len(test) - 1],
                "tuning": "each detector's z picked by best dev F1; test reported once",
            },
            "labels": {
                "positive": "multi-source surge: 2-4 sources, x2-8, 3-min ramp, 5-30 min plateau",
                "hard_negatives": [
                    "single-source burst x3-8",
                    "outage (source -> 0)",
                    "source switches to mock",
                    "slow 4h rise to x3",
                ],
                "match_window": f"event span + {MATCH_TAIL} min",
            },
            "ci": "Wilson 95% for recall/precision; exact Poisson 95% for false alerts per topic-day",
            "production": {
                "z_corrob": ss.Z_CORROB,
                "z_solo": ss.Z_SOLO,
                "baseline_ticks": ss.BASELINE_TICKS,
                **report_detectors["production (corroborated)"]["test_at_shipped_z"],
                "recall_by_event": buckets,
            },
            "detectors": report_detectors,
            "caveat": (
                "Synthetic benchmark. Positives are multi-source by construction, which "
                "favours a corroboration rule; the single-source baselines show how much "
                "of the precision comes from that design choice."
            ),
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
