"""
ESPN public soccer API client (free, no key).

Scores, status/clock, team stats, key events, confirmed lineups and Opta
play-by-play with shot coordinates, for league slug ``fifa.world``.

    scoreboard  {SITE}/{league}/scoreboard?dates=2026&limit=300   all fixtures
    standings   {API}/{league}/standings                          groups
    summary     {WEB}/{league}/summary?event={id}                 one match

Uses httpx's default User-Agent (ESPN's CDN returns 403 for custom UAs).
Parsers are pure (dict in, dict/model out).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import httpx

from api.schemas.schema import MatchEvent, TeamStats
from ml import shot_xg
from ml.team_names import canonical

log = logging.getLogger(__name__)

LEAGUE = os.getenv("ESPN_LEAGUE", "fifa.world")
SEASON = os.getenv("ESPN_SEASON", "2026")
SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
WEB = "https://site.web.api.espn.com/apis/site/v2/sports/soccer"
API = "https://site.api.espn.com/apis/v2/sports/soccer"

HOSTS = {"United States": "USA", "Mexico": "Mexico", "Canada": "Canada"}

# ESPN season slug → (stage code, round label)
STAGES = {
    "group-stage": ("group", "Group Stage"),
    "round-of-32": ("r32", "Round of 32"),
    "round-of-16": ("r16", "Round of 16"),
    "quarterfinals": ("qf", "Quarter-final"),
    "semifinals": ("sf", "Semi-final"),
    "3rd-place-match": ("3rd", "3rd Place"),
    "final": ("final", "Final"),
}

_STATUS = {
    "STATUS_SCHEDULED": "NS",
    "STATUS_FIRST_HALF": "1H",
    "STATUS_HALFTIME": "HT",
    "STATUS_SECOND_HALF": "2H",
    "STATUS_END_OF_REGULATION": "ET",
    "STATUS_OVERTIME": "ET",
    "STATUS_FIRST_HALF_EXTRA_TIME": "ET",
    "STATUS_HALFTIME_ET": "ET",
    "STATUS_SECOND_HALF_EXTRA_TIME": "ET",
    "STATUS_END_OF_EXTRATIME": "P",
    "STATUS_SHOOTOUT": "P",
    "STATUS_FULL_TIME": "FT",
    "STATUS_FINAL": "FT",
    "STATUS_FINAL_AET": "AET",
    "STATUS_FINAL_PEN": "PEN",
    "STATUS_POSTPONED": "PST",
    "STATUS_CANCELED": "CANC",
    "STATUS_ABANDONED": "ABD",
    "STATUS_SUSPENDED": "SUSP",
    "STATUS_DELAYED": "NS",
}
_PERIOD_STATUS = {1: "1H", 2: "2H", 3: "ET", 4: "ET", 5: "P"}

STATUS_LONG = {
    "NS": "Not Started",
    "1H": "First Half",
    "HT": "Halftime",
    "2H": "Second Half",
    "ET": "Extra Time",
    "P": "Penalties",
    "FT": "Match Finished",
    "AET": "After Extra Time",
    "PEN": "Penalties Finished",
    "PST": "Postponed",
    "CANC": "Cancelled",
    "ABD": "Abandoned",
    "SUSP": "Suspended",
}

_KEY_EVENT_TYPES = {
    "Goal": "goal",
    "Goal - Header": "goal",
    "Goal - Free-kick": "goal",
    "Goal - Volley": "goal",
    "Penalty - Scored": "penalty_goal",
    "Own Goal": "own_goal",
    "Yellow Card": "yellow",
    "Red Card": "red",
    "VAR - (Red) Card Upgrade": "red",
    "Substitution": "substitution",
    "Penalty - Missed": "penalty_missed",
    "Penalty - Saved": "penalty_missed",
}


# ── HTTP ───────────────────────────────────────────────────────────────────


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


# ── Scoreboard event → fixture record ─────────────────────────────────────


def parse_status(status: dict) -> tuple[str, Optional[int], Optional[int]]:
    """ESPN status → (code, minute, stoppage); "90'+4'" → 90, 4."""
    t = status.get("type") or {}
    code = _STATUS.get(t.get("name", ""))
    if code is None:
        state = t.get("state")
        if state == "pre":
            code = "NS"
        elif state == "post":
            code = "FT"
        else:
            code = _PERIOD_STATUS.get(int(status.get("period") or 1), "1H")
    if code in ("NS", "PST", "CANC"):
        return code, None, None
    minute, extra = shot_xg.parse_clock(status.get("displayClock", ""))
    if not minute and status.get("clock"):
        minute = int(float(status["clock"]) // 60) + 1
    if code == "HT":
        minute, extra = 45, extra
    return code, minute or None, extra


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_event(ev: dict) -> Optional[dict]:
    try:
        comp = ev["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        home, away = sides["home"], sides["away"]
    except (KeyError, IndexError):
        return None

    stage, label = STAGES.get((ev.get("season") or {}).get("slug", ""), ("group", "Group Stage"))
    group = ((comp.get("group") or {}).get("name") or "").replace("Group", "").strip() or None
    rnd = f"Group Stage - Group {group}" if stage == "group" and group else label
    status, minute, extra = parse_status(ev.get("status") or comp.get("status") or {})
    home_name = canonical(home["team"]["displayName"])
    away_name = canonical(away["team"]["displayName"])
    venue = comp.get("venue") or {}
    country = (venue.get("address") or {}).get("country", "")
    started = status not in ("NS", "PST", "CANC")
    host_side = (
        "home" if HOSTS.get(home_name) == country
        else "away" if HOSTS.get(away_name) == country
        else None
    )

    winner = None
    if (ev.get("status") or {}).get("type", {}).get("completed"):
        if home.get("winner"):
            winner = home_name
        elif away.get("winner"):
            winner = away_name

    return {
        "fixture_id": int(ev["id"]),
        "date": ev.get("date"),
        "kickoff": _parse_dt(ev.get("date")),
        "stage": stage,
        "group": group,
        "round": rnd,
        "status": status,
        "elapsed": minute,
        "elapsed_extra": extra,
        "home_name": home_name,
        "away_name": away_name,
        "home_espn_id": str(home["team"].get("id", "")),
        "away_espn_id": str(away["team"].get("id", "")),
        "home_logo": home["team"].get("logo", ""),
        "away_logo": away["team"].get("logo", ""),
        "home_score": _int(home.get("score")) if started else None,
        "away_score": _int(away.get("score")) if started else None,
        "home_pens": _int(home.get("shootoutScore")),
        "away_pens": _int(away.get("shootoutScore")),
        "winner": winner,
        "venue": venue.get("fullName", ""),
        # Host nations play at home in their own country; everything else is neutral.
        "host_side": host_side,
        "neutral": host_side is None,
    }


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# ── Standings → groups ────────────────────────────────────────────────────


def parse_standings(st: dict) -> dict[str, list[dict]]:
    """Group letter → rows sorted by final rank."""
    out: dict[str, list[dict]] = {}
    for g in st.get("children", []):
        letter = g.get("name", "").split()[-1]
        rows = []
        for e in (g.get("standings") or {}).get("entries", []):
            s = {x["name"]: x.get("value") for x in e.get("stats", [])}
            rows.append(
                {
                    "team": canonical(e["team"]["displayName"]),
                    "espn_id": str(e["team"].get("id", "")),
                    "rank": int(s.get("rank") or 0),
                    "points": int(s.get("points") or 0),
                    "gd": int(s.get("pointDifferential") or 0),
                    "gf": int(s.get("pointsFor") or 0),
                    "played": int(s.get("gamesPlayed") or 0),
                }
            )
        out[letter] = sorted(rows, key=lambda r: r["rank"] or 99)
    return dict(sorted(out.items()))


# ── Summary → stats / events / lineups / shots ────────────────────────────

_STAT_KEYS = {
    "possessionPct": "possession",
    "totalShots": "shots_total",
    "shotsOnTarget": "shots_on_goal",
    "totalPasses": "passes_total",
    "accuratePasses": "passes_accurate",
    "wonCorners": "corner_kicks",
    "foulsCommitted": "fouls",
    "offsides": "offsides",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "saves": "goalkeeper_saves",
}


def _team_stats(stats: list[dict], xg: float) -> TeamStats:
    v: dict[str, float] = {}
    for s in stats:
        k = _STAT_KEYS.get(s.get("name", ""))
        if k is None:
            continue
        try:
            v[k] = float(s.get("displayValue") or 0)
        except ValueError:
            v[k] = 0.0
    total, on = int(v.get("shots_total", 0)), int(v.get("shots_on_goal", 0))
    p, pa = int(v.get("passes_total", 0)), int(v.get("passes_accurate", 0))
    return TeamStats(
        possession=v.get("possession", 0.0),
        shots_total=total,
        shots_on_goal=on,
        shots_off_goal=max(0, total - on),
        passes_total=p,
        passes_accurate=pa,
        pass_accuracy=round(pa / p * 100, 1) if p else 0.0,
        corner_kicks=int(v.get("corner_kicks", 0)),
        fouls=int(v.get("fouls", 0)),
        offsides=int(v.get("offsides", 0)),
        yellow_cards=int(v.get("yellow_cards", 0)),
        red_cards=int(v.get("red_cards", 0)),
        goalkeeper_saves=int(v.get("goalkeeper_saves", 0)),
        expected_goals=round(xg, 2),
    )


def _events(key_events: list[dict], ids: dict[str, tuple[int, str]]) -> list[MatchEvent]:
    """ids: ESPN team id → (1|2, canonical name)."""
    out: list[MatchEvent] = []
    for k in key_events:
        etype = _KEY_EVENT_TYPES.get((k.get("type") or {}).get("text", ""))
        if etype is None or k.get("shootout"):
            continue
        team = ids.get(str((k.get("team") or {}).get("id", "")))
        if team is None:
            continue
        side, name = team
        if etype == "own_goal":
            # ESPN credits the benefiting side; downstream expects the
            # conceding (own-goal scorer's) side.
            side = 3 - side
            name = next(n for s, n in ids.values() if s == side)
        minute, extra = shot_xg.parse_clock((k.get("clock") or {}).get("displayValue", ""))
        parts = k.get("participants") or []
        player = (parts[0].get("athlete") or {}).get("displayName") if parts else None
        detail = None
        if etype == "substitution" and len(parts) > 1:
            detail = f"replaces {(parts[1].get('athlete') or {}).get('displayName', '')}"
        out.append(
            MatchEvent(
                elapsed=minute,
                extra=extra,
                team_id=side,
                team_name=name,
                player_name=player,
                type=etype,
                detail=detail,
                source="espn",
            )
        )
    out.sort(key=lambda e: (e.elapsed, e.extra or 0))
    return out


_LINE = {"G": 0, "D": 1, "M": 2, "F": 3}
_POS_LINE = {
    "G": "G",
    "SW": "D", "CD": "D", "CD-L": "D", "CD-R": "D", "LB": "D", "RB": "D", "LWB": "D", "RWB": "D",
    "DM": "M", "CM": "M", "CM-L": "M", "CM-R": "M", "LM": "M", "RM": "M", "M": "M",
    "AM": "M", "AM-L": "M", "AM-R": "M",
    "F": "F", "CF": "F", "CF-L": "F", "CF-R": "F", "LF": "F", "RF": "F", "RCF": "F", "LCF": "F",
}
# Left→right order within a line.
_POS_X = {
    "LB": 0, "LWB": 0, "LM": 0, "LF": 0, "AM-L": 0,
    "CD-L": 1, "CM-L": 1, "CF-L": 1,
    "CD": 2, "SW": 2, "DM": 2, "CM": 2, "AM": 2, "F": 2, "CF": 2, "M": 2, "G": 2,
    "CD-R": 3, "CM-R": 3, "CF-R": 3, "RCF": 3,
    "RB": 4, "RWB": 4, "RM": 4, "RF": 4, "AM-R": 4,
}


def _lineup(roster: dict, name: str) -> dict:
    starters, subs = [], []
    for p in roster.get("roster", []):
        pos = (p.get("position") or {}).get("abbreviation") or ""
        a = p.get("athlete") or {}
        entry = {
            "number": _int(p.get("jersey")) or 0,
            "name": a.get("displayName", ""),
            "position": pos,
            "line": _POS_LINE.get(pos, "M"),
            "grid": str(p.get("formationPlace") or ""),
            "captain": bool(p.get("captain")),
            "subbed_in": bool(p.get("subbedIn")),
            "subbed_out": bool(p.get("subbedOut")),
            "photo": None,
        }
        (starters if p.get("starter") else subs).append(entry)
    starters.sort(key=lambda e: (_LINE.get(e["line"], 2), _POS_X.get(e["position"], 2)))
    return {
        "team": name,
        "formation": roster.get("formation") or "",
        "startingXI": starters,
        "substitutes": subs,
        "coach": None,
    }


def parse_summary(summary: dict, fixture: dict) -> dict:
    """→ {home_stats, away_stats, events, lineups, shots}.

    `fixture` is the parse_event() record for the same match.
    """
    ids = {
        fixture["home_espn_id"]: (1, fixture["home_name"]),
        fixture["away_espn_id"]: (2, fixture["away_name"]),
    }
    shots = shot_xg.shots_from_commentary(summary.get("commentary", []))
    # Commentary plays carry the team name only, not its id.
    by_name = {name: side for side, name in ids.values()}
    xg = {1: 0.0, 2: 0.0}
    sides = []
    for s in shots:
        side = ids.get(s.team_id, (0, ""))[0] or by_name.get(canonical(s.team_name), 0)
        sides.append(side)
        if side:
            xg[side] += s.xg
    stats = {1: TeamStats(), 2: TeamStats()}
    for t in (summary.get("boxscore") or {}).get("teams", []):
        side = ids.get(str((t.get("team") or {}).get("id", "")), (0, ""))[0]
        if side:
            stats[side] = _team_stats(t.get("statistics", []), xg[side])

    lineups = {}
    for r in summary.get("rosters", []):
        side = ids.get(str((r.get("team") or {}).get("id", "")), (0, ""))[0]
        if side and r.get("roster"):
            lineups["home" if side == 1 else "away"] = _lineup(r, ids[str(r["team"]["id"])][1])

    return {
        "home_stats": stats[1],
        "away_stats": stats[2],
        "events": _events(summary.get("keyEvents", []), ids),
        "lineups": lineups if len(lineups) == 2 else None,
        "shots": [{**s.to_dict(), "side": side} for s, side in zip(shots, sides)],
    }
