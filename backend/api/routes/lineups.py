"""
Lineups endpoint.

    GET /matches/{fixture_id}/lineups

Sources, in priority order:
    1. ESPN confirmed XI (formation, positions, subs), written to
       match:{id}:lineups by the match producer. source="espn".
    2. API-Sports confirmed XI, if API_SPORTS_KEY is set. source="api-sports".
    3. Zafronix 2026 squad with a projected XI (starter/captain/shirt number
       heuristic). source="zafronix_squad", projected=True.
    4. Empty XI, source="unavailable".

Zafronix has no player photos, so headshots come from a best-effort,
Redis-cached Wikipedia thumbnail lookup, with per-player fallback to the
numbered circle on the frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse

import httpx
from fastapi import APIRouter, HTTPException, Request

from api.schemas.schema import MatchState

router = APIRouter()
log = logging.getLogger(__name__)

API_KEY = os.getenv("API_SPORTS_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"

ZAFRONIX_KEY = os.getenv("ZAFRONIX_API_KEY", "")
ZAFRONIX_BASE = "https://api.zafronix.com/fifa/worldcup/v1"
ZAFRONIX_SEASON = int(os.getenv("ZAFRONIX_SEASON", "2026"))

PHOTO_CACHE_TTL = 7 * 86_400  # 7 days — headshots don't change
PHOTO_LOOKUP_TIMEOUT = 4.0
ROSTER_CACHE_TTL = 6 * 3600

# Zafronix position codes (GK/DF/MF/FW) → a plausible pitch-role formation.
# Zafronix rosters aren't ordered as a starting XI, so we pick the most
# senior 11 (captain + starters first, then by shirt number) and infer a
# formation from how many of each line that gives us.
_POS_LINE = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}


def _formation_from_counts(n_def: int, n_mid: int, n_fwd: int) -> str:
    """Nearest pitch-layout formation for a (DF, MF, FW) count.

    Only shapes whose frontend slot split matches the counts are mapped.
    """
    key = (n_def, n_mid, n_fwd)
    known = {
        (4, 3, 3): "4-3-3",
        (4, 4, 2): "4-4-2",
        (4, 5, 1): "4-1-4-1",
        (3, 5, 2): "3-5-2",
        (5, 3, 2): "5-3-2",
        (3, 4, 3): "3-4-3",
        (4, 1, 5): "4-2-3-1",
    }
    if key in known:
        return known[key]
    # Nearest by defender count, then a sane default.
    return {3: "3-4-3", 4: "4-3-3", 5: "5-3-2"}.get(n_def, "4-3-3")


async def _photo_for(r, client: httpx.AsyncClient, name: str) -> str | None:
    """Best-effort Wikipedia summary thumbnail, Redis-cached. Never raises."""
    if not name:
        return None
    cache_key = f"player:photo:{name.lower()}"
    cached = await r.get(cache_key)
    if cached is not None:
        return cached or None  # "" = known no-photo sentinel

    url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(name)}"
    )
    photo = ""
    try:
        resp = await client.get(url, timeout=PHOTO_LOOKUP_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            thumb = data.get("thumbnail") or {}
            if thumb.get("source") and thumb.get("width", 0) >= 80:
                photo = thumb["source"]
    except Exception:
        pass

    await r.setex(cache_key, PHOTO_CACHE_TTL, photo)
    return photo or None


# ── Tier 1: Zafronix 2026 roster ───────────────────────────────────────────

# Feed (ESPN) team names → the names Zafronix uses (FIFA English naming).
_ZAFRONIX_NAME_ALIAS = {
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Iran (Islamic Republic)": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "China PR": "China",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Cabo Verde": "Cape Verde",
}


def _zafronix_name(team_name: str) -> str:
    return _ZAFRONIX_NAME_ALIAS.get(team_name, team_name)


async def _fetch_zafronix_roster(
    client: httpx.AsyncClient, team_name: str
) -> tuple[list[dict] | None, str]:
    """Zafronix roster for one team, or (None, reason) on failure.

    Tries the FIFA-canonical alias first, then the raw name.
    """
    candidates = []
    aliased = _zafronix_name(team_name)
    candidates.append(aliased)
    if team_name != aliased:
        candidates.append(team_name)

    last_reason = "unknown"
    for name in candidates:
        url = f"{ZAFRONIX_BASE}/teams/{urllib.parse.quote(name)}/roster"
        try:
            resp = await client.get(
                url,
                params={"year": ZAFRONIX_SEASON},
                headers={"X-API-Key": ZAFRONIX_KEY},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) >= 11:
                    return data, "ok"
                n = len(data) if isinstance(data, list) else 0
                last_reason = f"200 OK but only {n} players for '{name}' (year={ZAFRONIX_SEASON}) — 2026 squad may not be populated in Zafronix yet"
                log.info(f"Zafronix roster '{name}': {last_reason}")
            elif resp.status_code == 404:
                last_reason = f"404 for '{name}' — team name doesn't match Zafronix's naming (try adding it to _ZAFRONIX_NAME_ALIAS)"
                log.info(f"Zafronix roster: {last_reason}")
            elif resp.status_code in (401, 403):
                last_reason = f"{resp.status_code} — ZAFRONIX_API_KEY is missing, invalid, or expired"
                log.warning(f"Zafronix roster: {last_reason}")
                return None, last_reason  # auth won't improve by trying the other name
            else:
                last_reason = f"HTTP {resp.status_code} for '{name}'"
                log.info(f"Zafronix roster: {last_reason}")
        except Exception as exc:
            last_reason = f"request failed: {exc}"
            log.warning(f"Zafronix roster fetch failed for '{name}': {exc}")

    return None, last_reason


def _select_starting_xi(roster: list[dict]) -> tuple[list[dict], str]:
    """Plausible starting XI from a full squad, plus its formation.

    Grouped DF, MF, FW first (the frontend assigns pitch slots by index);
    starter, captain and jersey break ties within a group.
    """

    def sort_key(p: dict):
        return (
            0 if p.get("starter") else 1,
            0 if p.get("captain") else 1,
            p.get("jersey") or 99,
        )

    # Standard shapes to try, in order of preference — (def, mid, fwd) counts
    # only, since Zafronix tags players as DF/MF/FW with no DM/AM
    # granularity to split further. Must match keys in
    # _formation_from_counts' `known` dict exactly, so the chosen formation
    # string's assumed per-line counts always equal the xi's real counts.
    # Picking the target shape from what's actually AVAILABLE (rather than
    # deriving a formation from an arbitrary top-11 cut) is what guarantees
    # every slot gets a player of the right position type.
    _TARGET_SHAPES = [
        (4, 3, 3),
        (4, 4, 2),
        (3, 5, 2),
        (5, 3, 2),
        (3, 4, 3),
    ]

    keepers = sorted([p for p in roster if p.get("position") == "GK"], key=sort_key)
    defs = sorted([p for p in roster if p.get("position") == "DF"], key=sort_key)
    mids = sorted([p for p in roster if p.get("position") == "MF"], key=sort_key)
    fwds = sorted([p for p in roster if p.get("position") == "FW"], key=sort_key)

    # Pick the first shape the available squad can actually fill.
    n_def, n_mid, n_fwd = 4, 3, 3
    for d, m, f in _TARGET_SHAPES:
        if len(defs) >= d and len(mids) >= m and len(fwds) >= f:
            n_def, n_mid, n_fwd = d, m, f
            break
    else:
        # Squad is short in some line — take what's there.
        n_def = min(len(defs), 5) or 4
        n_mid = min(len(mids), 5) or 3
        n_fwd = min(len(fwds), 3) or 3

    xi: list[dict] = []
    if keepers:
        xi.append(keepers[0])
    # Order matters: DF block, then MF block, then FW block — this is what
    # keeps array index aligned with H_POS's GK→DEF→MID→FWD slot ordering.
    xi.extend(defs[:n_def])
    xi.extend(mids[:n_mid])
    xi.extend(fwds[:n_fwd])

    # If the squad was short somewhere, pad from whatever's left over so we
    # still return 11 total rather than an incomplete XI.
    if len(xi) < 11:
        leftover = [p for p in (defs + mids + fwds) if p not in xi]
        xi.extend(leftover[: 11 - len(xi)])

    n_def = sum(1 for p in xi if p.get("position") == "DF")
    n_mid = sum(1 for p in xi if p.get("position") == "MF")
    n_fwd = sum(1 for p in xi if p.get("position") == "FW")
    # If counts don't reach 10 outfield (missing position data), pad mids.
    formation = _formation_from_counts(n_def or 4, n_mid or 3, n_fwd or 3)
    return xi, formation


async def _build_zafronix_team(
    r, client: httpx.AsyncClient, photo_client: httpx.AsyncClient, team_name: str
) -> tuple[dict | None, str]:
    roster, reason = await _fetch_zafronix_roster(client, team_name)
    if not roster:
        return None, reason

    xi, formation = _select_starting_xi(roster)
    names = [p.get("name") or "" for p in xi]
    photos = await asyncio.gather(
        *[_photo_for(r, photo_client, n) for n in names],
        return_exceptions=True,
    )

    starting = []
    for p, photo in zip(xi, photos):
        starting.append(
            {
                "number": p.get("jersey") or 0,
                "name": p.get("name") or "",
                "position": p.get("position") or "",
                "grid": "",
                "captain": bool(p.get("captain")),
                "photo": photo if isinstance(photo, str) else None,
            }
        )

    return {
        "team": team_name,
        "formation": formation,
        "startingXI": starting,
        "coach": None,
    }, "ok"


async def _fetch_from_zafronix(
    r, home_name: str, away_name: str
) -> tuple[dict | None, str]:
    if not ZAFRONIX_KEY:
        return None, "ZAFRONIX_API_KEY is not set in this process's environment"
    cache_key = f"zafronix:lineup:v2:{home_name}:{away_name}"
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached), "ok (cached)"

    async with httpx.AsyncClient() as client, httpx.AsyncClient(
        headers={"User-Agent": "wc2026-lineups/1.0"}
    ) as photo_client:
        (home_entry, home_reason), (away_entry, away_reason) = await asyncio.gather(
            _build_zafronix_team(r, client, photo_client, home_name),
            _build_zafronix_team(r, client, photo_client, away_name),
        )

    if not home_entry or not away_entry:
        reason = f"home({home_name})={home_reason} · away({away_name})={away_reason}"
        log.info(f"Zafronix lineup unavailable: {reason}")
        return None, reason

    result = {
        "home": home_entry,
        "away": away_entry,
        "source": "zafronix_squad",
        "projected": True,
    }
    await r.setex(cache_key, ROSTER_CACHE_TTL, json.dumps(result))
    return result, "ok"


# ── Tier 2: API-Sports live lineups ────────────────────────────────────────


def _parse_player(raw: dict) -> dict:
    pl = raw.get("player", {})
    stats = raw.get("statistics", [{}])
    games = stats[0].get("games", {}) if stats else {}
    player_id = pl.get("id")
    return {
        "number": pl.get("number") or 0,
        "name": pl.get("name") or "",
        "position": pl.get("pos") or "",
        "grid": games.get("number") or "",
        "photo": (
            f"https://media.api-sports.io/football/players/{player_id}.png"
            if player_id
            else None
        ),
    }


async def _fetch_from_api_sports(fixture_id: str) -> dict | None:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{BASE_URL}/fixtures/lineups",
                headers={"x-apisports-key": API_KEY},
                params={"fixture": fixture_id},
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        log.warning(f"lineup fetch failed: {exc}")
        return None

    response = data.get("response", [])
    if len(response) < 2:
        return None

    result = {}
    for entry in response[:2]:
        team = entry.get("team", {})
        result[team.get("id")] = {
            "team": team.get("name", ""),
            "formation": entry.get("formation", "4-3-3"),
            "startingXI": [_parse_player(p) for p in entry.get("startXI", [])],
            "coach": (entry.get("coach") or {}).get("name", ""),
        }
    return result if len(result) == 2 else None


# ── Route ──────────────────────────────────────────────────────────────────


@router.get("/{fixture_id}/lineups")
async def lineups(fixture_id: str, request: Request):
    r = request.app.state.redis

    raw = await r.get(f"match:{fixture_id}:state")
    if not raw:
        raise HTTPException(404, "Fixture not found")
    state = MatchState.model_validate_json(raw)

    # Tier 1: confirmed XI from ESPN, written by the match producer.
    confirmed = await r.get(f"match:{fixture_id}:lineups")
    if confirmed:
        return json.loads(confirmed)

    # Tier 2: confirmed XI from API-Sports (paid key).
    if API_KEY:
        api_data = await _fetch_from_api_sports(str(state.fixture_id))
        if api_data:
            teams = list(api_data.values())
            home_entry = next((t for t in teams if t["team"] == state.home_name), teams[0])
            away_entry = next((t for t in teams if t["team"] == state.away_name), teams[1])
            return {"home": home_entry, "away": away_entry, "source": "api-sports"}

    # Tier 3: real 2026 squad with a PROJECTED XI — not a confirmed lineup.
    zx, zx_reason = await _fetch_from_zafronix(r, state.home_name, state.away_name)
    if zx:
        return zx

    # Tier 4: nothing known — empty XI, no guessed formation.
    return {
        "home": {"team": state.home_name, "formation": "", "startingXI": [], "coach": None},
        "away": {"team": state.away_name, "formation": "", "startingXI": [], "coach": None},
        "source": "unavailable",
        "zafronix_debug": zx_reason,
    }
