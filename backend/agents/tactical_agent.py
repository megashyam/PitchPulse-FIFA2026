"""
Tactical-fingerprint matcher for the match dashboard's TacticalCard.

Numeric kNN: every TacticalProfiles object is loaded once (TTL cache) and
the live team is matched by Euclidean distance on z-scored features both
sides measure — possession, whole-pitch PPDA, pass accuracy and shots (per
90 for a match in progress). z-scores use the indexed population's mean and
SD. A feature the live feed has not reported yet is left out of the
distance. The response gives the distance and the per-feature gap
(live − profile). Returns None when Weaviate is empty or unavailable.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.weaviate_client import get_weaviate_client, TACTICAL_PROFILES
from schemas.schema import MatchState, TeamStats

log = logging.getLogger(__name__)

FEATURES = ("possession", "ppda_full", "pass_accuracy", "shots")
CACHE_TTL_S = 600.0

_cache: Tuple[float, List[dict], Optional[np.ndarray]] = (float("-inf"), [], None)


def _load() -> Tuple[List[dict], Optional[np.ndarray]]:
    """(profiles, feature matrix) for profiles carrying every feature."""
    global _cache
    at, profiles, X = _cache
    if time.monotonic() - at < CACHE_TTL_S and profiles:
        return profiles, X
    objs = get_weaviate_client().fetch_all(TACTICAL_PROFILES)
    profiles = [o for o in objs if all(o.get(f) is not None for f in FEATURES)]
    X = np.array([[float(o[f]) for f in FEATURES] for o in profiles]) if profiles else None
    _cache = (time.monotonic(), profiles, X)
    return profiles, X


def _live_features(stats: TeamStats, opp: TeamStats, elapsed: int) -> np.ndarray:
    """Live values in FEATURES order; NaN where the feed has no data yet."""
    actions = stats.tackles + stats.interceptions + stats.fouls
    per90 = 90.0 / max(elapsed, 15) if elapsed < 90 else 1.0
    return np.array([
        stats.possession if stats.possession > 0 else np.nan,
        stats.ppda_full(opp) if actions > 0 and opp.passes_accurate > 0 else np.nan,
        stats.pass_accuracy if stats.passes_total > 0 else np.nan,
        stats.shots_total * per90,
    ])


def nearest(live: np.ndarray, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the k nearest rows of X, z-scored on X,
    over the non-NaN entries of `live`."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    m = ~np.isnan(live)
    d = np.sqrt((((X[:, m] - live[m]) / sd[m]) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    return idx, d[idx]


def _gaps(live: np.ndarray, o: dict) -> Dict[str, Optional[float]]:
    return {
        f: None if np.isnan(v) else round(float(v) - float(o[f]), 1)
        for f, v in zip(FEATURES, live)
    }


async def match_team(
    state: MatchState,
    side: str,  # "home" | "away"
    top_k: int = 3,
) -> Optional[dict]:
    """Best historical fingerprint match for one team, or None."""
    wv = get_weaviate_client()
    if not wv.ready:
        return None

    if side == "home":
        team, opp, stats, opp_stats = state.home_name, state.away_name, state.home_stats, state.away_stats
    else:
        team, opp, stats, opp_stats = state.away_name, state.home_name, state.away_stats, state.home_stats

    profiles, X = await asyncio.to_thread(_load)
    if X is None:
        return None
    live = _live_features(stats, opp_stats, state.elapsed or 0)
    if np.isnan(live[0]):
        return None  # no possession yet: too early to match
    idx, dist = nearest(live, X, top_k)
    best = profiles[idx[0]]

    return {
        "team": team,
        "opponent": opp,
        "live_possession": round(stats.possession, 1),
        "live": {f: None if np.isnan(v) else round(float(v), 1) for f, v in zip(FEATURES, live)},
        "match": {
            "team": best.get("team"),
            "opponent": best.get("opponent"),
            "competition": best.get("competition"),
            "season": best.get("season"),
            "distance": round(float(dist[0]), 2),
            "gaps": _gaps(live, best),
            "possession": best.get("possession"),
            "ppda_full": best.get("ppda_full"),
            "pass_accuracy": best.get("pass_accuracy"),
            "shots": best.get("shots"),
            "ppda": best.get("ppda"),
            "ppda_mid_third": best.get("ppda_mid_third"),
            "ppda_att_third": best.get("ppda_att_third"),
            "press_intensity": best.get("press_intensity"),
            "content": best.get("content"),
        },
        "alternatives": [
            {
                "team": profiles[i].get("team"),
                "opponent": profiles[i].get("opponent"),
                "season": profiles[i].get("season"),
                "ppda_full": profiles[i].get("ppda_full"),
                "distance": round(float(d), 2),
            }
            for i, d in zip(idx[1:], dist[1:])
        ],
    }
