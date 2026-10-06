"""
Market W/D/L and over/under client for The Odds API.

TTL-cached with stale-if-error reuse. Each bookmaker's 1X2 line is de-vigged
on its own (Shin, ml.prior_builder.shin_devig) and the fair probabilities
are averaged across books, then returned as zero-margin decimal odds (1/p).
Totals use the half-goal line quoted by the most books (ties go to the one
nearest 2.5), each book's Over/Under de-vigged two-way and P(over) averaged.
Team names go through ml.team_names.canonical.

Env:
    ODDS_API_KEY     enable live odds (empty → {} → Elo priors everywhere)
    ODDS_CACHE_TTL   seconds to cache the odds map (default 5400, ~16
                     requests/day; each costs 4 quota units, 2 markets x
                     2 regions)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple

import httpx
import numpy as np

from ml.team_names import canonical

from config import ODDS_API_KEY, ODDS_CACHE_TTL as CACHE_TTL

log = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"  # update to the correct key when WC2026 goes live

# (home_name, away_name) -> (decimal_home, decimal_draw, decimal_away)
# (odds_home, odds_draw, odds_away[, p_over, line])
OddsMap = Dict[Tuple[str, str], Tuple[float, ...]]


async def _fetch_odds() -> Optional[OddsMap]:
    """H2H + totals odds per match: {(home, away): (odds_home, odds_draw, odds_away[, p_over, line])}.

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
                    "markets": "h2h,totals",
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
        over: Dict[float, list] = {}  # line -> per-book fair P(over)
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                outcomes = market.get("outcomes", [])
                if market.get("key") == "h2h":
                    by_name = {o["name"]: o["price"] for o in outcomes}
                    hp, dp, ap = by_name.get(raw_home), by_name.get("Draw"), by_name.get(raw_away)
                    if hp and dp and ap and min(hp, dp, ap) > 1.0:
                        fair.append(shin_devig((hp, dp, ap))[0])
                elif market.get("key") == "totals":
                    by_side = {(o["name"], o.get("point")): o["price"] for o in outcomes}
                    for (side, line), price in by_side.items():
                        under = by_side.get(("Under", line))
                        if side == "Over" and line is not None and float(line) % 1 == 0.5:
                            if price > 1.0 and under and under > 1.0:
                                over.setdefault(float(line), []).append(
                                    shin_devig((price, under))[0][0]
                                )

        if not fair:
            continue
        p = np.mean(fair, axis=0)
        entry = tuple(round(float(1.0 / x), 4) for x in p)
        if over:
            line = max(over, key=lambda k: (len(over[k]), -abs(k - 2.5)))
            entry += (round(float(np.mean(over[line])), 4), line)
        odds_map[(canonical(raw_home), canonical(raw_away))] = entry

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
                # stale-if-error: keep serving the last good snapshot instead of
                # flipping the prior table to Elo mid-match
                log.warning("Odds API fetch failed — serving stale cached odds")
            else:
                self._cache = {}
                self._cached_at = now  # back off; don't hammer a failing API
        return self._cache or {}


_singleton = _OddsApiClient()


def get_oddsapi_client() -> _OddsApiClient:
    """Singleton Odds API client."""
    return _singleton
