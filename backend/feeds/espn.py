"""
ESPN public soccer API client (free, no key).

Scores, status/clock, team stats, key events, confirmed lineups and Opta
play-by-play with shot coordinates, for league slug ``fifa.world``.

    scoreboard  {SITE}/{league}/scoreboard?dates=2026&limit=300   all fixtures
    standings   {API}/{league}/standings                          groups
    summary     {WEB}/{league}/summary?event={id}                 one match

Fetchers return raw JSON. Parsers validate it into the Espn* models in
schemas/espn_schema.py, then build domain models with `from_espn`.

Uses httpx's default User-Agent (ESPN's CDN returns 403 for custom UAs).
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from pydantic import ValidationError

from schemas.espn_schema import EspnEvent, EspnKeyEvent, EspnStandings, EspnSummary
from schemas.schema import (
    Fixture,
    Lineup,
    MatchDetail,
    MatchEvent,
    StandingRow,
    TeamIds,
    TeamStats,
)
from ml import shot_xg
from ml.team_names import canonical

from config import ESPN_LEAGUE as LEAGUE, ESPN_SEASON as SEASON

log = logging.getLogger(__name__)

SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
WEB = "https://site.web.api.espn.com/apis/site/v2/sports/soccer"
API = "https://site.api.espn.com/apis/v2/sports/soccer"


async def fetch_scoreboard(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(
        f"{SITE}/{LEAGUE}/scoreboard",
        params={"dates": SEASON, "limit": 300},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("events", [])


async def fetch_standings(client: httpx.AsyncClient) -> dict:
    r = await client.get(f"{API}/{LEAGUE}/standings", timeout=20)
    r.raise_for_status()
    return r.json()


async def fetch_summary(client: httpx.AsyncClient, event_id: int | str) -> dict:
    r = await client.get(f"{WEB}/{LEAGUE}/summary", params={"event": event_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_event(raw: dict) -> Optional[Fixture]:
    """Scoreboard event → Fixture; None (logged) if malformed."""
    try:
        return Fixture.from_espn(EspnEvent.model_validate(raw))
    except ValidationError as e:
        log.warning(f"Skipping malformed ESPN event {raw.get('id')}: {e}")
        return None


def parse_standings(raw: dict) -> dict[str, list[StandingRow]]:
    """Group letter → rows sorted by final rank."""
    out: dict[str, list[StandingRow]] = {}
    for g in EspnStandings.model_validate(raw).children:
        rows = [StandingRow.from_espn(e) for e in g.standings.entries]
        out[g.name.split()[-1]] = sorted(rows, key=lambda r: r.rank or 99)
    return dict(sorted(out.items()))


def match_events(key_events: list[EspnKeyEvent], ids: TeamIds) -> list[MatchEvent]:
    """Tracked key events (goals, cards, subs, missed pens) in match order."""
    events = [e for k in key_events if (e := MatchEvent.from_espn(k, ids))]
    return sorted(events, key=lambda e: (e.elapsed, e.extra or 0))


def parse_summary(raw: dict, fixture: Fixture) -> MatchDetail:
    """Match summary → stats, events, lineups and shots with model xG."""
    summary = EspnSummary.model_validate(raw)
    ids = fixture.espn_ids()

    def side_of(team_id: str) -> int:
        return ids.get(team_id, (0, ""))[0]

    # Commentary plays carry the team name only, not its id.
    by_name = {name: side for side, name in ids.values()}
    shots = shot_xg.shots_from_commentary(summary.commentary)
    sides = [side_of(s.team_id) or by_name.get(canonical(s.team_name), 0) for s in shots]
    xg = {1: 0.0, 2: 0.0}
    for s, side in zip(shots, sides):
        if side:
            xg[side] += s.xg

    stats = {1: TeamStats(), 2: TeamStats()}
    for t in summary.boxscore.teams:
        if side := side_of(t.team.id):
            stats[side] = TeamStats.from_espn(t, xg[side])

    lineups = {}
    for r in summary.rosters:
        if (side := side_of(r.team.id)) and r.roster:
            lineups["home" if side == 1 else "away"] = Lineup.from_espn(r, ids[r.team.id][1])

    return MatchDetail(
        home_stats=stats[1],
        away_stats=stats[2],
        events=match_events(summary.key_events, ids),
        lineups=lineups if len(lineups) == 2 else None,
        shots=[{**s.to_dict(), "side": side} for s, side in zip(shots, sides)],
    )
