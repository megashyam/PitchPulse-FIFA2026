"""
Market W/D/L client for The Odds API.

TTL-cached with stale-if-error reuse. Each bookmaker's line is de-vigged
on its own (Shin, ml.prior_builder.shin_devig) and the fair probabilities
are averaged across books, then returned as zero-margin decimal odds (1/p).
Team names go through ml.team_names.canonical.

Env:
    ODDS_API_KEY     enable live odds (empty → {} → Elo priors everywhere)
    ODDS_CACHE_TTL   seconds to cache the odds map (default 5400, ~16
                     requests/day, inside the 500/month free tier)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict, Optional, Tuple

import httpx
import numpy as np

from ml.team_names import canonical

log = logging.getLogger(__name__)

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"  # update to the correct key when WC2026 goes live
CACHE_TTL = float(os.getenv("ODDS_CACHE_TTL", "5400"))

# (home_name, away_name) -> (decimal_home, decimal_draw, decimal_away)
OddsMap = Dict[Tuple[str, str], Tuple[float, float, float]]


async def _fetch_odds() -> Optional[OddsMap]:
    """H2H odds per match: {(home, away): (odds_home, odds_draw, odds_away)}.

    None on transport/HTTP failure (distinct from an empty market list).
    """
    if not ODDS_API_KEY:
        log.debug("No ODDS_API_KEY set — Elo priors will be used for all matches")
        return {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us,eu",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
            )
            r.raise_for_status()
            events = r.json()
            log.info(
                "Odds API: %s used / %s remaining this month",
                r.headers.get("x-requests-used", "?"),
                r.headers.get("x-requests-remaining", "?"),
            )
    except httpx.HTTPStatusError as exc:
        log.warning(f"Odds API HTTP error {exc.response.status_code}: {exc}")
        return None
    except Exception as exc:
        log.warning(f"Odds API error: {exc}")
        return None

    from ml.prior_builder import shin_devig  # local: prior_builder imports config

    odds_map: OddsMap = {}
    for event in events:
        raw_home = (event.get("home_team") or "").strip()
        raw_away = (event.get("away_team") or "").strip()
        if not raw_home or not raw_away:
            continue

        fair = []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                by_name = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                hp, dp, ap = by_name.get(raw_home), by_name.get("Draw"), by_name.get(raw_away)
                if hp and dp and ap and min(hp, dp, ap) > 1.0:
                    fair.append(shin_devig((hp, dp, ap))[0])

        if not fair:
            continue
        p = np.mean(fair, axis=0)
        odds_map[(canonical(raw_home), canonical(raw_away))] = tuple(
            round(float(1.0 / x), 4) for x in p
        )

    log.info(f"Odds API: loaded {len(odds_map)} match odds")
    return odds_map


class _OddsApiClient:
    """TTL-cached client for The Odds API."""

    def __init__(self) -> None:
        self._cache: Optional[OddsMap] = None
        self._cached_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_all_odds(self) -> OddsMap:
        now = time.monotonic()
        if self._cache is not None and now - self._cached_at < CACHE_TTL:
            return self._cache

        async with self._lock:  # coalesce concurrent refreshes to one request
            now = time.monotonic()
            if self._cache is not None and now - self._cached_at < CACHE_TTL:
                return self._cache

            fresh = await _fetch_odds()
            if fresh is not None:
                self._cache = fresh
                self._cached_at = now
            elif self._cache is not None:
                # stale-if-error: keep serving the last good snapshot rather
                # than flipping the whole prior table to Elo mid-match
                log.warning("Odds API fetch failed — serving stale cached odds")
            else:
                self._cache = {}
                self._cached_at = now  # back off; don't hammer a failing API
        return self._cache or {}


_singleton = _OddsApiClient()


def get_oddsapi_client() -> _OddsApiClient:
    """Singleton Odds API client."""
    return _singleton
