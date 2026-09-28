"""
Played results and live matches for conditioning the tournament simulator.

Read from the producer's MatchStates (matches:active ∪ matches:completed),
falling back to the committed snapshot.
"""

from __future__ import annotations

from typing import Dict, Tuple

import redis.asyncio as aioredis

from api.schemas.event_types import COMPLETED_STATUSES, LIVE_STATUSES, RED_TYPES
from api.schemas.schema import MatchState
from ml.tournament_sim import LiveMatch, results_from_fixtures
from ml.wc_2026_config import FIXTURE_BY_ID


def red_counts(state: MatchState, upto: int | None = None) -> Tuple[int, int]:
    """Real sending-offs per side, optionally only up to minute `upto`."""
    h = a = 0
    for e in state.events:
        if e.type not in RED_TYPES or e.source != "espn":
            continue
        if upto is not None and e.elapsed > upto:
            continue
        if e.team_id == 1:
            h += 1
        else:
            a += 1
    return h, a


def result_of(state: MatchState) -> dict:
    winner = None
    if state.home_score != state.away_score:
        winner = state.home_name if state.home_score > state.away_score else state.away_name
    elif state.home_pens is not None and state.away_pens is not None:
        winner = state.home_name if state.home_pens > state.away_pens else state.away_name
    return {"home_score": state.home_score, "away_score": state.away_score, "winner": winner}


async def load_states(r: aioredis.Redis) -> list[MatchState]:
    ids = (await r.smembers("matches:active")) | (await r.smembers("matches:completed"))
    if not ids:
        return []
    raws = await r.mget([f"match:{i}:state" for i in ids])
    out = []
    for raw in raws:
        if raw:
            try:
                out.append(MatchState.model_validate_json(raw))
            except Exception:
                continue
    return out


async def sim_conditions(
    r: aioredis.Redis,
) -> Tuple[Dict[int, dict], Dict[int, LiveMatch]]:
    """(results, live) for run_simulation from current match states."""
    results: Dict[int, dict] = {}
    live: Dict[int, LiveMatch] = {}
    for s in await load_states(r):
        if s.fixture_id not in FIXTURE_BY_ID:
            continue
        if s.status_short in COMPLETED_STATUSES:
            results[s.fixture_id] = result_of(s)
        elif s.status_short in LIVE_STATUSES:
            rh, ra = red_counts(s)
            live[s.fixture_id] = LiveMatch(
                minute=s.elapsed or 0,
                home_score=s.home_score,
                away_score=s.away_score,
                red_home=rh,
                red_away=ra,
                extra=s.elapsed_extra,
            )
    if not results and not live:
        results = results_from_fixtures()
    return results, live


async def results_before(r: aioredis.Redis, kickoff) -> Dict[int, dict]:
    """Completed results for fixtures that kicked off before `kickoff`."""
    if kickoff is None:
        return {}
    out = {
        s.fixture_id: result_of(s)
        for s in await load_states(r)
        if s.status_short in COMPLETED_STATUSES
        and s.kickoff_time is not None
        and s.kickoff_time < kickoff
        and s.fixture_id in FIXTURE_BY_ID
    }
    return out or results_from_fixtures(before=kickoff.strftime("%Y-%m-%dT%H:%MZ"))
