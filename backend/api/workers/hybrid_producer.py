"""
Hybrid live-match state producer.

This worker builds a unified fixture intelligence pipeline by combining:
    - live World Cup 2026 match feeds from worldcup26.ir
    - StatsBomb historical event data as statistical proxies
    - Redis-backed state persistence for API and SSE consumers

The producer normalizes inconsistent upstream match formats, maintains
deterministic fixture timelines, synthesizes missing goal events from score
changes, and exposes a single canonical MatchState representation.

Key responsibilities:
    - Poll live fixture data and normalize match metadata.
    - Map live fixtures to historically similar StatsBomb matches using
      team similarity and Elo-based matching.
    - Replay historical event streams into minute-level statistical snapshots.
    - Persist active/completed fixture states into Redis.
    - Publish state changes only when meaningful updates occur.

The design intentionally separates:
    - live event truth (worldcup26.ir)
    - statistical context (StatsBomb proxies)
    - persistence/event distribution (Redis)

This allows downstream consumers such as REST APIs, SSE streams, and
counterfactual workers to operate on a stable deterministic match timeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import redis.asyncio as aioredis

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from api.schemas.schema import MatchEvent, MatchState, TeamStats
from api.schemas.event_types import COMPLETED_STATUSES
from ml.statsbomb import (
    SB_BASE,
    SEASON_IDS as SB_SEASONS,
    card_from_event,
    gk_is_save,
    shot_is_on_target,
    sort_events,
)
from ml.team_names import to_sim, to_statsbomb
from ml.wc_2026_config import WC2026_TEAMS

log = logging.getLogger(__name__)

_TEAM_ELO: dict[str, float] = {t.name: t.elo for t in WC2026_TEAMS}
_DEFAULT_ELO = 1600.0


def _elo_of(statsbomb_name: str) -> float:
    """Return the best available Elo for a StatsBomb team name."""
    return _TEAM_ELO.get(to_sim(statsbomb_name), _DEFAULT_ELO)


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
WC26_BASE = os.getenv("WC26_BASE", "https://worldcup26.ir")
SB_COMPETITION = int(os.getenv("SB_COMPETITION", "43"))
SOURCE_TZ = ZoneInfo(os.getenv("WC26_SOURCE_TZ", "America/New_York"))
FT_ACTIVE_GRACE_S = int(os.getenv("FT_ACTIVE_GRACE_S", str(6 * 3600)))

LIVE_STATUSES = {"1H", "HT", "2H", "ET", "P", "LIVE"}

STATUS_MAP: dict[str, str] = {
    "scheduled": "NS",
    "not_started": "NS",
    "in_play": "1H",
    "in play": "1H",
    "live": "1H",
    "halftime": "HT",
    "half_time": "HT",
    "half-time": "HT",
    "second_half": "2H",
    "extra_time": "ET",
    "penalties": "P",
    "finished": "FT",
    "completed": "FT",
    "NS": "NS",
    "1H": "1H",
    "HT": "HT",
    "2H": "2H",
    "ET": "ET",
    "P": "P",
    "FT": "FT",
    "AET": "AET",
    "PEN": "PEN",
}

# Representative minute per status when the feed omits a numeric clock.
_STATUS_MINUTE = {
    "HT": 45,
    "1H": 30,
    "2H": 70,
    "ET": 105,
    "P": 120,
    "FT": 90,
    "AET": 120,
    "PEN": 120,
    "NS": 0,
}

_MINUTE_RE = re.compile(r"(\d{1,3})")
_FID_RE = re.compile(r"\d+")
_SCORER_RE = re.compile(r'"([^"\d][^"]*?)\s+(\d{1,3})(?:\+(\d{1,2}))?\'')
_MINUTE_ONLY_RE = re.compile(r"(\d{1,3})(?:\+(\d{1,2}))?'")


class StatsBank:
    """
    Historical match intelligence cache built from StatsBomb event data.

    StatsBank lazily loads StatsBomb match timelines and converts raw event
    streams into cumulative minute-by-minute TeamStats snapshots.

    Responsibilities:
        - Load historical competition matches.
        - Match live fixtures to historical proxy games.
        - Cache processed event timelines.
        - Provide statistics and proxy events at arbitrary match minutes.

    The cache is fixture-aware and handles reversed team orientations so
    historical statistics remain correctly attributed to live home/away teams.
    """

    def __init__(self) -> None:
        self._timelines: dict[int, dict[int, tuple[TeamStats, TeamStats]]] = {}
        self._extra_events: dict[int, list[dict]] = {}
        self._matches: list[dict] = []
        self._by_teams: dict[tuple[str, str], int] = {}
        self._by_mid: dict[int, tuple[str, str]] = {}
        self._assigned: dict[int, int] = {}
        self._flipped: dict[int, bool] = {}

    async def load_matches(self, client: httpx.AsyncClient) -> None:
        """
        Load StatsBomb match metadata for configured seasons.

        Builds internal lookup indexes for:
            - exact team pair matching
            - reversed fixture matching
            - match ID to team mapping

        This runs once during worker startup to avoid repeated external API calls.
        """
        loaded_seasons = 0
        for sid in SB_SEASONS:
            url = f"{SB_BASE}/matches/{SB_COMPETITION}/{sid}.json"
            try:
                r = await client.get(url, timeout=30)
                r.raise_for_status()
                season_matches = r.json()
                self._matches.extend(season_matches)
                loaded_seasons += 1
                log.info(
                    f"StatsBomb: loaded {len(season_matches)} matches (season={sid})"
                )
            except Exception as e:
                log.warning(f"StatsBomb season {sid} failed: {e}")

        for m in self._matches:
            h = m["home_team"]["home_team_name"]
            a = m["away_team"]["away_team_name"]
            mid = m["match_id"]
            self._by_teams.setdefault((h, a), mid)
            self._by_teams.setdefault((a, h), mid)
            self._by_mid[mid] = (h, a)

        if not loaded_seasons:
            log.error(
                "Could not load any StatsBomb match list — hybrid stats will be empty"
            )

    async def get_timeline(
        self,
        client: httpx.AsyncClient,
        home_name: str,
        away_name: str,
        fixture_id: int,
    ) -> dict[int, tuple[TeamStats, TeamStats]]:
        """
        Retrieve or construct the statistical timeline for a live fixture.

        The method assigns a historical StatsBomb match as a statistical proxy,
        downloads its event stream when needed, builds cumulative snapshots, and
        returns the timeline oriented to the live fixture's home/away teams.

        Results are cached by StatsBomb match ID to avoid duplicate processing.
        """
        if fixture_id in self._assigned:
            mid = self._assigned[fixture_id]
            if mid in self._timelines:
                return self._orient(fixture_id, self._timelines[mid])

        mid, flipped = self._find_match(home_name, away_name)
        self._assigned[fixture_id] = mid
        self._flipped[fixture_id] = flipped

        if mid in self._timelines:
            return self._orient(fixture_id, self._timelines[mid])

        try:
            url = f"{SB_BASE}/events/{mid}.json"
            log.info(
                f"Loading StatsBomb events mid={mid} "
                f"(proxy for {home_name} vs {away_name})"
            )
            r = await client.get(url, timeout=30)
            r.raise_for_status()
            events = r.json()
            teams = self._by_mid.get(mid, ("Home", "Away"))
            self._timelines[mid] = _build_timeline(events, teams)
            self._extra_events[mid] = _extract_card_sub_events(events, teams)
            log.info(
                f"StatsBomb timeline ready: {len(self._timelines[mid])} snapshots, "
                f"{len(self._extra_events[mid])} card/sub events "
                f"for fixture {fixture_id}"
            )
        except Exception as e:
            log.error(f"Could not load StatsBomb events for mid={mid}: {e}")
            self._timelines[mid] = {}
            self._extra_events[mid] = []

        return self._orient(fixture_id, self._timelines[mid])

    def _orient(
        self,
        fixture_id: int,
        timeline: dict[int, tuple[TeamStats, TeamStats]],
    ) -> dict[int, tuple[TeamStats, TeamStats]]:
        """Return the timeline oriented to the LIVE fixture's (home, away).

        The raw timeline is cached by StatsBomb match_id in the historical
        match's own home/away order; when this fixture matched with the sides
        reversed, swap each snapshot so proxy stats land on the correct live
        team. Never mutates the shared cache.
        """
        if not self._flipped.get(fixture_id) or not timeline:
            return timeline
        return {m: (away_s, home_s) for m, (home_s, away_s) in timeline.items()}

    def get_assigned_match_id(self, fixture_id: int) -> Optional[int]:
        """The StatsBomb match_id backing this fixture's stats proxy, once
        assigned by get_timeline(). None before the first successful call."""
        return self._assigned.get(fixture_id)

    def events_at_minute(self, fixture_id: int, minute: int) -> list[dict]:
        """All proxy card/sub events at or before `minute` for this fixture's
        assigned historical match. Deterministic — safe to call every poll."""
        mid = self._assigned.get(fixture_id)
        if mid is None:
            return []
        flipped = self._flipped.get(fixture_id, False)
        out: list[dict] = []
        for e in self._extra_events.get(mid, []):
            if e["minute"] > minute:
                continue
            out.append(
                {**e, "is_home": (not e["is_home"]) if flipped else e["is_home"]}
            )
        return out

    def _find_match(self, home_name: str, away_name: str) -> tuple[int, bool]:
        """
        Select the most appropriate historical StatsBomb proxy fixture.

        Matching strategy:
            1. Exact team pairing.
            2. Reversed team pairing.
            3. Minimum Elo-distance similarity across all historical fixtures.

        Returns both the selected match ID and whether team orientation must be
        flipped before applying statistics to the live fixture.
        """
        h = to_statsbomb(home_name)
        a = to_statsbomb(away_name)

        if (h, a) in self._by_teams:
            mid = self._by_teams[(h, a)]
            log.info(f"StatsBomb: exact match for {h} vs {a} → mid={mid}")
            return mid, False
        if (a, h) in self._by_teams:
            mid = self._by_teams[(a, h)]
            log.info(f"StatsBomb: exact match (reversed) for {h} vs {a} → mid={mid}")
            return mid, True

        live_home_elo = _TEAM_ELO.get(to_sim(home_name), _DEFAULT_ELO)
        live_away_elo = _TEAM_ELO.get(to_sim(away_name), _DEFAULT_ELO)

        best_mid: Optional[int] = None
        best_flipped = False
        best_cost = float("inf")
        for m in self._matches:
            th = m["home_team"]["home_team_name"]
            ta = m["away_team"]["away_team_name"]
            eh = _elo_of(th)
            ea = _elo_of(ta)

            cost_direct = abs(eh - live_home_elo) + abs(ea - live_away_elo)
            cost_flipped = abs(eh - live_away_elo) + abs(ea - live_home_elo)
            if cost_direct <= cost_flipped:
                cost, flipped = cost_direct, False
            else:
                cost, flipped = cost_flipped, True
            if cost < best_cost:
                best_cost, best_mid, best_flipped = cost, m["match_id"], flipped

        if best_mid is not None:
            teams = self._by_mid.get(best_mid, ("?", "?"))
            log.info(
                f"StatsBomb: no exact match for {h} vs {a} — nearest by Elo is "
                f"{teams[0]} vs {teams[1]} (mid={best_mid}, flipped={best_flipped}, "
                f"Δelo={best_cost:.0f})"
            )
            return best_mid, best_flipped

        raise RuntimeError("No StatsBomb matches available")

    @staticmethod
    def stats_at_minute(
        timeline: dict[int, tuple[TeamStats, TeamStats]],
        minute: int,
    ) -> tuple[TeamStats, TeamStats]:
        """Stats snapshot at or before the given minute."""
        if not timeline:
            return TeamStats(), TeamStats()
        available = sorted(timeline.keys())
        minute = max(available[0], min(minute, available[-1]))
        best = max((m for m in available if m <= minute), default=available[0])
        return timeline[best]


def _extract_card_sub_events(
    events: list[dict],
    match_teams: tuple[str, str],
) -> list[dict]:
    """
    Extract non-goal match events from StatsBomb timelines.

    Converts historical yellow cards, red cards, and substitutions into
    lightweight proxy events that can enrich live fixture timelines.

    Player identities are intentionally omitted because these events originate
    from historical proxy matches rather than the actual fixture.
    """
    home_name, _away_name = match_teams
    out: list[dict] = []

    for ev in events:
        team_name = ev.get("team", {}).get("name", "")
        if team_name not in match_teams:
            continue
        is_home = team_name == home_name
        minute = ev.get("minute", 0)

        card = card_from_event(ev)
        if card:
            out.append({"minute": minute, "type": card, "is_home": is_home})
            continue

        if (ev.get("type") or {}).get("name") == "Substitution":
            out.append({"minute": minute, "type": "substitution", "is_home": is_home})

    yellows = [e for e in out if e["type"] == "yellow"][:6]
    reds = [e for e in out if e["type"] == "red"][:2]
    subs = [e for e in out if e["type"] == "substitution"][:6]
    return sorted(yellows + reds + subs, key=lambda e: e["minute"])


def _snapshot(tallies: defaultdict, possession: float) -> TeamStats:
    p = int(tallies["passes_total"])
    pa = int(tallies["passes_accurate"])
    return TeamStats(
        possession=possession,
        shots_total=int(tallies["shots_total"]),
        shots_on_goal=int(tallies["shots_on_goal"]),
        shots_off_goal=int(tallies["shots_off_goal"]),
        passes_total=p,
        passes_accurate=pa,
        pass_accuracy=round(pa / p * 100, 1) if p else 0.0,
        corner_kicks=int(tallies["corner_kicks"]),
        offsides=int(tallies["offsides"]),
        fouls=int(tallies["fouls"]),
        yellow_cards=int(tallies["yellow_cards"]),
        red_cards=int(tallies["red_cards"]),
        goalkeeper_saves=int(tallies["goalkeeper_saves"]),
        expected_goals=round(tallies["expected_goals"], 3),
    )


def _build_timeline(
    events: list[dict],
    match_teams: tuple[str, str],
) -> dict[int, tuple[TeamStats, TeamStats]]:
    """
    Convert a StatsBomb event stream into minute-level statistical snapshots.

    Events are replayed chronologically and accumulated into TeamStats objects
    containing:
        - shots
        - possession estimate
        - passing metrics
        - fouls/cards
        - goalkeeper saves
        - expected goals

    The resulting timeline enables querying match statistics at any elapsed
    minute.
    """
    home_name, away_name = match_teams
    h: defaultdict = defaultdict(float)
    a: defaultdict = defaultdict(float)
    timeline: dict[int, tuple[TeamStats, TeamStats]] = {}

    def take_snapshot(minute: int) -> None:
        total = h["ev_count"] + a["ev_count"]
        home_poss = round(h["ev_count"] / total * 100, 1) if total else 50.0
        timeline[minute] = (
            _snapshot(h, home_poss),
            _snapshot(a, round(100 - home_poss, 1)),
        )

    last_minute: Optional[int] = None

    for ev in sort_events(events):
        team_name = ev.get("team", {}).get("name", "")
        if team_name not in match_teams:
            continue
        is_home = team_name == home_name

        s = h if is_home else a
        etype = (ev.get("type") or {}).get("name", "")
        minute = ev.get("minute", 0)

        if last_minute is not None and minute != last_minute:
            take_snapshot(last_minute)
        last_minute = minute

        s["ev_count"] += 1

        if etype == "Pass":
            s["passes_total"] += 1
            if ev.get("pass", {}).get("outcome") is None:
                s["passes_accurate"] += 1
            if ev.get("pass", {}).get("type", {}).get("name") == "Corner":
                s["corner_kicks"] += 1

        elif etype == "Shot":
            s["shots_total"] += 1
            shot = ev.get("shot", {})
            xg = shot.get("statsbomb_xg")
            s["expected_goals"] += float(xg) if xg is not None else 0.0
            outcome = (shot.get("outcome") or {}).get("name", "")
            if shot_is_on_target(outcome):
                s["shots_on_goal"] += 1
            else:
                s["shots_off_goal"] += 1

        elif etype == "Goal Keeper":
            gk_out = (ev.get("goalkeeper", {}).get("outcome") or {}).get("name", "")
            if gk_is_save(gk_out):
                s["goalkeeper_saves"] += 1

        elif etype in ("Foul Committed", "Bad Behaviour"):
            if etype == "Foul Committed":
                s["fouls"] += 1
            card = card_from_event(ev)
            if card == "red":
                s["red_cards"] += 1
            elif card == "yellow":
                s["yellow_cards"] += 1

        elif etype == "Offside":
            s["offsides"] += 1

    if last_minute is not None:
        take_snapshot(last_minute)

    return timeline


# worldcup26.ir polling
def _derive_status(
    finished: str, time_elapsed: str, raw_status: str = ""
) -> tuple[str, int, bool]:
    """
    Normalize inconsistent upstream match status representations.

    Handles:
        - explicit status strings
        - numeric clocks
        - stoppage time
        - halftime
        - extra time
        - penalty shootouts

    Returns a standardized status code, elapsed minute, and whether the
    elapsed value was inferred rather than directly provided.
    """
    finished = (finished or "").upper()
    te = (time_elapsed or "").lower().strip()
    rs = (raw_status or "").lower().strip()

    if finished == "TRUE" or te in (
        "finished",
        "ft",
        "fulltime",
        "full_time",
        "full-time",
    ):
        return "FT", 90, True

    for token in (rs, te):
        if token in STATUS_MAP:
            code = STATUS_MAP[token]
            return code, _STATUS_MINUTE.get(code, 0), code != "NS"

    if ("half" in te and "time" in te) or te == "ht":
        return "HT", 45, False  # HT clock genuinely is 45'
    if "pen" in te or "pen" in rs:
        return "P", 120, True
    if "extra" in te or te == "et" or "extra" in rs:
        return "ET", 105, True

    m = _MINUTE_RE.match(te)
    if m:
        minute = int(m.group(1))
        if minute > 120:
            return "FT", 90, True
        if minute > 90:
            return "ET", minute, False
        return ("2H" if minute > 45 else "1H"), minute, False

    return "NS", 0, False


def _opt_int(v) -> Optional[int]:
    """Parse a feed score and preserve unknown values as None."""
    if v in (None, "", "null", "NULL", "None"):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_fixture_id(raw_id, home_name: str, away_name: str) -> int:
    """
    Normalize a raw worldcup26.ir fixture payload.

    Converts provider-specific fields into the internal fixture format used by
    the producer, including:
        - fixture identifiers
        - teams
        - scores
        - match status
        - kickoff timestamps
        - goal events

    Invalid or incomplete fixtures are ignored safely.
    """
    m = _FID_RE.search(str(raw_id or ""))
    if m:
        return 2_026_000 + int(m.group(0)[:6])
    digest = hashlib.md5(f"{home_name}{away_name}".encode()).hexdigest()[:8]
    return int(digest, 16) % 1_000_000 + 2_026_000


def _parse_kickoff_local(local_date: str) -> Optional[datetime]:
    """Parse the feed's local kickoff time and convert it into UTC.

    The source date is expressed in SOURCE_TZ, so the parser must attach that
    zone before converting; otherwise downstream windows and reminders drift.
    """
    if not local_date:
        return None
    try:
        dt = datetime.strptime(local_date, "%m/%d/%Y %H:%M")
        return dt.replace(tzinfo=SOURCE_TZ).astimezone(timezone.utc)
    except Exception:
        return None


def _parse_scorer_events(home_name: str, away_name: str, g: dict) -> list[MatchEvent]:
    """Build goal events from the feed's Postgres-array-style scorer strings.
    Handles accented names and stoppage-time minutes ("90+2'")."""
    out: list[MatchEvent] = []
    for team_id, team_name, raw in (
        (1, home_name, g.get("home_scorers", "")),
        (2, away_name, g.get("away_scorers", "")),
    ):
        s = str(raw or "")
        if s.lower() in ("", "null", "none", "{}"):
            continue

        found = _SCORER_RE.findall(s)
        if found:
            for player_name, min_str, extra_str in found:
                out.append(
                    MatchEvent(
                        elapsed=int(min_str),
                        extra=int(extra_str) if extra_str else None,
                        team_id=team_id,
                        team_name=team_name,
                        player_name=player_name.strip() or None,
                        type="goal",
                        detail="worldcup26.ir",
                    )
                )
        else:
            for m in _MINUTE_ONLY_RE.finditer(s):
                out.append(
                    MatchEvent(
                        elapsed=int(m.group(1)),
                        extra=int(m.group(2)) if m.group(2) else None,
                        team_id=team_id,
                        team_name=team_name,
                        player_name=None,
                        type="goal",
                        detail="worldcup26.ir",
                    )
                )
    return out


def _parse_wc26_game(g: dict) -> Optional[dict]:
    """Parse one worldcup26.ir game dict into a normalised record.

    Known field names: home_team_name_en / away_team_name_en, home_score /
    away_score (strings or null), finished ("TRUE"/"FALSE"), time_elapsed
    ("finished"/"45"/"halftime"/null), id ("1".."104"), group, type,
    local_date ("06/11/2026 13:00"), home_scorers/away_scorers."""
    try:
        home_name = g.get("home_team_name_en") or g.get("home_team_name") or ""
        away_name = g.get("away_team_name_en") or g.get("away_team_name") or ""
        if not home_name or not away_name:
            return None

        home_score = _opt_int(g.get("home_score"))
        away_score = _opt_int(g.get("away_score"))

        finished = str(g.get("finished", "FALSE")).upper()
        time_elapsed = str(g.get("time_elapsed") or "").lower().strip()
        raw_status = str(
            g.get("status") or g.get("state") or g.get("match_status") or ""
        )
        status, elapsed, estimated = _derive_status(finished, time_elapsed, raw_status)

        fid = _parse_fixture_id(g.get("id") or g.get("_id"), home_name, away_name)

        group = g.get("group", "")
        match_type = str(g.get("type", "group")).lower()
        if match_type == "group":
            rnd = f"Group Stage - Group {group}" if group else "Group Stage"
        else:
            rnd = match_type.replace("_", " ").title()

        return {
            "fixture_id": fid,
            "home_name": home_name,
            "away_name": away_name,
            "home_score": home_score,  # Optional[int] — None = feed unknown
            "away_score": away_score,
            "status": status,
            "elapsed": elapsed,
            "elapsed_estimated": estimated,
            "kickoff": _parse_kickoff_local(g.get("local_date", "")),
            "round": rnd,
            "venue": "",
            "pre_events": _parse_scorer_events(home_name, away_name, g),
        }
    except Exception as e:
        log.debug(f"_parse_wc26_game failed: {e}  raw={g}")
        return None


async def fetch_wc26_games(client: httpx.AsyncClient) -> list[dict]:
    """
    Fetch the complete World Cup 2026 fixture feed.

    Handles multiple possible response schemas from the upstream provider and
    returns a normalized list of raw fixture payloads.

    Network failures are isolated so the worker can continue operating.
    """
    try:
        r = await client.get(f"{WC26_BASE}/get/games", timeout=15)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list):
            games = data
        elif isinstance(data, dict):
            games = (
                data.get("games")
                or data.get("matches")
                or data.get("data")
                or data.get("fixtures")
                or []
            )
        else:
            games = []

        live = [
            g
            for g in games
            if str(g.get("finished", "FALSE")).upper() != "TRUE"
            and g.get("time_elapsed") not in (None, "", "finished", "null")
        ]
        log.info(
            f"worldcup26.ir: {len(games)} games total, "
            f"{len(live)} currently live/in-progress"
        )
        return games
    except Exception as e:
        log.warning(f"worldcup26.ir fetch failed: {e}")
        return []


# Score-change event synthesis


class ScoreTracker:
    """
    Maintains deterministic goal state across producer polling cycles.

    Some live feeds provide scores without scorer metadata. ScoreTracker fills
    this gap by detecting score deltas and synthesizing goal events.

    State is hydrated from Redis on restart to prevent duplicate event creation
    and preserve deterministic replay behavior.
    """

    def __init__(self) -> None:
        self._scores: dict[int, tuple[int, int]] = {}
        self._events: dict[int, list[MatchEvent]] = {}
        self._hydrated: set[int] = set()

    async def hydrate(self, r: aioredis.Redis, fixture_id: int) -> None:
        if fixture_id in self._hydrated:
            return
        self._hydrated.add(fixture_id)
        raw = await r.get(f"match:{fixture_id}:state")
        if not raw:
            return
        try:
            prev = MatchState.model_validate_json(raw)
        except Exception:
            return
        self._scores[fixture_id] = (prev.home_score, prev.away_score)
        self._events[fixture_id] = [e for e in prev.events if e.detail == "synthesised"]
        if self._events[fixture_id]:
            log.info(
                f"  [{fixture_id}] tracker hydrated: score "
                f"{prev.home_score}-{prev.away_score}, "
                f"{len(self._events[fixture_id])} synthesised event(s)"
            )

    def last_score(self, fixture_id: int) -> tuple[int, int]:
        return self._scores.get(fixture_id, (0, 0))

    def update(
        self,
        fixture_id: int,
        home_name: str,
        away_name: str,
        home_score: int,
        away_score: int,
        elapsed: int,
    ) -> list[MatchEvent]:
        """
        Detect score changes and generate missing goal events.

        Compares the current fixture score against the previous observed state and
        creates synthetic goal events for any unexplained score increase.

        Returns the complete accumulated event history for the fixture.
        """
        prev_h, prev_a = self._scores.get(fixture_id, (0, 0))

        # Replay restart (mock producer): score genuinely reset to 0-0.
        if home_score == 0 and away_score == 0 and (prev_h > 0 or prev_a > 0):
            self._events[fixture_id] = []
            prev_h, prev_a = 0, 0

        events = self._events.setdefault(fixture_id, [])

        for team_id, name, delta in (
            (1, home_name, home_score - prev_h),
            (2, away_name, away_score - prev_a),
        ):
            for _ in range(max(0, delta)):
                events.append(
                    MatchEvent(
                        elapsed=elapsed,
                        team_id=team_id,
                        team_name=name,
                        player_name=None,  # not available without scorer data
                        type="goal",
                        detail="synthesised",
                    )
                )
                log.info(f"  [{fixture_id}] Synthesised goal: {name} {elapsed}'")

        self._scores[fixture_id] = (home_score, away_score)
        return list(events)


# Redis persistence

FT_AT_TTL = 40 * 24 * 3600  # keep the FT timestamp for the CF feed's lifetime


async def _worker_visible(r: aioredis.Redis, fid: int, status: str) -> bool:
    """
    Determine whether a fixture should remain in the active worker queue.

    Live fixtures remain immediately visible. Completed fixtures stay active
    temporarily during a configurable grace period so API clients and SSE
    consumers can receive final updates before archival.
    """
    if status not in COMPLETED_STATUSES:
        return True
    key = f"match:{fid}:ft_at"
    ts = await r.get(key)
    if ts is None:
        await r.set(key, str(time.time()), ex=FT_AT_TTL)
        return True
    try:
        return (time.time() - float(ts)) < FT_ACTIVE_GRACE_S
    except ValueError:
        return True


async def persist(
    r: aioredis.Redis,
    state: MatchState,
    *,
    changed: bool,
    worker_visible: bool,
) -> None:
    """
    Persist canonical MatchState data into Redis.

    Stores:
        - fixture state snapshots
        - active/completed fixture indexes
        - state-change notifications

    Pub/sub updates are emitted only when the serialized state changes,
    reducing unnecessary downstream processing.
    """
    ttl = 43_200 if state.status_short == "NS" else 3_600
    pipe = r.pipeline(transaction=True)
    pipe.setex(f"match:{state.fixture_id}:state", ttl, state.model_dump_json())
    if worker_visible:
        pipe.sadd("matches:active", str(state.fixture_id))
        pipe.srem("matches:completed", str(state.fixture_id))
    else:
        pipe.sadd("matches:completed", str(state.fixture_id))
        pipe.srem("matches:active", str(state.fixture_id))
    await pipe.execute()

    if changed:
        # Publish only when the persisted match state actually changes.
        await r.publish("match_update", json.dumps({"fixture_id": state.fixture_id}))
        log.info(
            f"  [{state.fixture_id}] "
            f"{state.status_short} {str(state.elapsed or '-'):>3}' "
            f"{state.home_name} {state.home_score}-{state.away_score} "
            f"{state.away_name}"
        )


async def expire_stale(r: aioredis.Redis, seen_ids: set[int]) -> None:
    """
    Remove fixtures that disappeared from the upstream feed.

    Cleans Redis active/completed indexes to prevent stale fixtures from
    remaining visible after the source provider removes them.
    """
    for set_name in ("matches:active", "matches:completed"):
        stored = await r.smembers(set_name)
        for fid_str in stored:
            try:
                known = int(fid_str) in seen_ids
            except ValueError:
                known = False
            if not known:
                await r.srem(set_name, fid_str)
                log.info(f"  [{fid_str}] removed from {set_name}")


_STATUS_LONG = {
    "NS": "Not Started",
    "1H": "First Half",
    "HT": "Halftime",
    "2H": "Second Half",
    "ET": "Extra Time",
    "P": "Penalties",
    "FT": "Match Finished",
    "AET": "After Extra Time",
    "PEN": "Penalties Finished",
}


async def _loop(r: aioredis.Redis) -> None:
    """
    Execute the hybrid producer polling lifecycle.

    The worker continuously:
        1. Fetches live fixtures.
        2. Normalizes match state.
        3. Loads statistical proxy timelines.
        4. Generates events and statistics.
        5. Persists updated MatchState objects.
        6. Publishes changes for downstream consumers.

    The loop is designed for long-running async execution.
    """
    bank = StatsBank()
    tracker = ScoreTracker()
    # fid → last persisted payload (excluding updated_at) for change detection
    last_payload: dict[int, str] = {}

    headers = {"User-Agent": "wc2026-hybrid/1.0", "Accept": "application/json"}

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        await bank.load_matches(client)

        log.info(
            f"Hybrid producer running — worldcup26.ir={WC26_BASE} "
            f"poll={POLL_INTERVAL}s source_tz={SOURCE_TZ}"
        )

        while True:
            raw_games = await fetch_wc26_games(client)
            seen_ids: set[int] = set()

            for raw in raw_games:
                try:
                    game = _parse_wc26_game(raw)
                    if game is None:
                        continue

                    fid = game["fixture_id"]
                    status = game["status"]
                    home_name = game["home_name"]
                    away_name = game["away_name"]
                    seen_ids.add(fid)

                    if status == "NS":
                        state = MatchState(
                            fixture_id=fid,
                            league_id=43,
                            season=2026,
                            round=game["round"],
                            venue=game["venue"],
                            status_short="NS",
                            status_long="Not Started",
                            elapsed=None,
                            kickoff_time=game["kickoff"],
                            home_id=1,
                            home_name=home_name,
                            home_score=0,
                            away_id=2,
                            away_name=away_name,
                            away_score=0,
                            stats_source="unavailable",
                        )
                        await _persist_if_changed(r, state, last_payload)
                        continue

                    await tracker.hydrate(r, fid)
                    prev_h, prev_a = tracker.last_score(fid)
                    home_score = (
                        game["home_score"] if game["home_score"] is not None else prev_h
                    )
                    away_score = (
                        game["away_score"] if game["away_score"] is not None else prev_a
                    )

                    # Load StatsBomb proxy stats for the fixture at the current minute.
                    try:
                        timeline = await bank.get_timeline(
                            client, home_name, away_name, fid
                        )
                    except Exception as e:
                        log.error(f"  [{fid}] get_timeline failed: {e}")
                        timeline = {}

                    # Use feed-provided scorers when available, otherwise synthesize goals from score deltas.
                    goal_events = game["pre_events"]
                    if not goal_events:
                        goal_events = tracker.update(
                            fid,
                            home_name,
                            away_name,
                            home_score,
                            away_score,
                            game["elapsed"] or 0,
                        )

                    # Clamp completed matches so stoppage-time goals never appear after the final whistle.
                    elapsed = game["elapsed"]
                    estimated = game["elapsed_estimated"]
                    if status in COMPLETED_STATUSES:
                        max_ev = max((e.elapsed for e in goal_events), default=90)
                        elapsed = max(90, max_ev)
                        estimated = True

                    h_stats, a_stats = bank.stats_at_minute(timeline, elapsed or 0)

                    proxy_events = [
                        MatchEvent(
                            elapsed=pe["minute"],
                            team_id=1 if pe["is_home"] else 2,
                            team_name=home_name if pe["is_home"] else away_name,
                            player_name=None,
                            type=pe["type"],
                            detail="statsbomb proxy",
                        )
                        for pe in bank.events_at_minute(fid, elapsed or 0)
                    ]

                    events = sorted(goal_events + proxy_events, key=lambda e: e.elapsed)

                    state = MatchState(
                        fixture_id=fid,
                        league_id=43,
                        season=2026,
                        round=game["round"],
                        venue=game["venue"],
                        status_short=status,
                        status_long=_STATUS_LONG.get(status, status),
                        elapsed=elapsed,
                        elapsed_estimated=estimated,
                        kickoff_time=game["kickoff"],
                        home_id=1,
                        home_name=home_name,
                        home_score=home_score,
                        home_stats=h_stats,
                        away_id=2,
                        away_name=away_name,
                        away_score=away_score,
                        away_stats=a_stats,
                        events=events,
                        stats_source="statsbomb_proxy" if timeline else "unavailable",
                        stats_proxy_match_id=bank.get_assigned_match_id(fid),
                    )
                    await _persist_if_changed(r, state, last_payload)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    fid_hint = raw.get("id") if isinstance(raw, dict) else None
                    log.error(
                        f"  [{fid_hint}] fixture processing failed, skipping: {e}",
                        exc_info=True,
                    )
                    continue

            if seen_ids:
                await expire_stale(r, seen_ids)
            elif not raw_games:
                log.warning(
                    "worldcup26.ir returned 0 games. "
                    "Check WC26_BASE or whether the tournament is on a rest day."
                )

            # Drop change-detection entries for fixtures the feed no longer returns.
            for gone in set(last_payload) - seen_ids:
                last_payload.pop(gone, None)

            await asyncio.sleep(POLL_INTERVAL)


async def _persist_if_changed(
    r: aioredis.Redis, state: MatchState, last_payload: dict[int, str]
) -> None:
    """
    Persist fixture state only when meaningful changes occur.

    Uses serialized state comparison to prevent redundant Redis writes and
    unnecessary pub/sub notifications.
    """
    payload = state.model_dump_json(exclude={"updated_at"})
    changed = last_payload.get(state.fixture_id) != payload
    last_payload[state.fixture_id] = payload
    visible = await _worker_visible(r, state.fixture_id, state.status_short)
    await persist(r, state, changed=changed, worker_visible=visible)


async def run(redis_client: aioredis.Redis) -> None:
    """
    Application lifecycle entry point for the async worker.

    Handles graceful cancellation and top-level crash logging when launched
    alongside the API service.
    """
    try:
        await _loop(redis_client)
    except asyncio.CancelledError:
        log.info("Hybrid producer cancelled")
    except Exception as exc:
        log.error(f"Hybrid producer crashed: {exc}", exc_info=True)


async def main() -> None:
    """
    Standalone worker entry point.

    Creates the Redis connection and launches the hybrid producer loop when
    executed directly as a Python module.
    """
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    log.info(f"Redis: {REDIS_URL}")
    await _loop(r)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
