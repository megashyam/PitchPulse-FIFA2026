"""
Counterfactual, prediction and live-probability endpoints.

    GET      /matches/{id}/counterfactual          latest CF result
    GET      /matches/{id}/counterfactual/feed     full history (up to 20)
    GET      /matches/{id}/counterfactual/stream   SSE (pub/sub-backed)
    GET/POST /matches/{id}/counterfactual/trigger  force one cycle (token-gated)
    GET      /matches/{id}/prediction              per-match W/D/L + tournament path
    GET      /matches/{id}/live-prob               in-play W/D/L (ml/in_play.py)
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from agents import counterfactual_agent
from api.routes._security import require_trigger_token
from api.routes._sse import next_message, subscription
from api.schemas.event_types import COMPLETED_STATUSES
from api.schemas.schema import MatchState
from api.tournament_state import red_counts
from ml.in_play import inplay_wdl
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import match_wdl
from ml.tournament_sim import LiveMatch, live_ko_advance
from ml.wc_2026_config import FIXTURE_BY_ID

router = APIRouter()
log = logging.getLogger(__name__)

TTL_LIVE = 3_600
TTL_COMPLETED = 2_592_000  # 30 days
FEED_MAX = 19  # up to 20 entries


def _ttl_for(status_short: str) -> int:
    return TTL_COMPLETED if status_short in COMPLETED_STATUSES else TTL_LIVE


async def _pre_match_wdl(state: MatchState, odds_table):
    """(p_home, p_draw, p_away, source); the shared pre-match prior."""
    fx = FIXTURE_BY_ID.get(state.fixture_id) or {}
    wdl = match_wdl(
        state.home_name,
        state.away_name,
        host_side=fx.get("host_side"),
        odds_table=odds_table,
    )
    quoted = bool(odds_table) and (
        (state.home_name, state.away_name) in odds_table
        or (state.away_name, state.home_name) in odds_table
    )
    return (*wdl, "market_odds" if quoted else "elo")


def _point(p: float) -> dict:
    # A model prior is a point estimate: no sampling interval exists.
    return {"p": round(p, 4), "ci_lo": round(p, 4), "ci_hi": round(p, 4)}


# ── Per-match pre-tournament prediction ────────────────────────────────────


@router.get("/{fixture_id}/prediction")
async def get_match_prediction(fixture_id: str, request: Request):
    """Pre-match W/D/L (market odds or Elo) plus tournament-path odds."""
    r = request.app.state.redis
    state_raw = await r.get(f"match:{fixture_id}:state")

    if not state_raw:
        raise HTTPException(404, f"No match state for fixture {fixture_id}")

    state = MatchState.model_validate_json(state_raw)

    odds_client = get_oddsapi_client()
    odds_table = await odds_client.get_all_odds()
    p_win, p_draw, p_loss, source = await _pre_match_wdl(state, odds_table)

    tournament_raw = await r.get("predict:tournament:latest")
    home_tournament = away_tournament = None

    if tournament_raw:
        tournament = json.loads(tournament_raw)
        for team in tournament.get("teams", []):
            if team["name"] == state.home_name:
                home_tournament = team
            elif team["name"] == state.away_name:
                away_tournament = team

    return {
        "fixture_id": int(fixture_id),
        "home_name": state.home_name,
        "away_name": state.away_name,
        "status_short": state.status_short,
        "elapsed": state.elapsed,
        "match_odds": {
            "home_win": _point(p_win),
            "draw": _point(p_draw),
            "away_win": _point(p_loss),
        },
        "source": source,
        "home_tournament": home_tournament,
        "away_tournament": away_tournament,
    }


# ── In-play live probability ──────────────────────────────────────────────


@router.get("/{fixture_id}/live-prob")
async def get_live_prob(fixture_id: str, request: Request):
    """In-play W/D/L for the current minute, score and reds (ml/in_play.py)."""
    r = request.app.state.redis
    state_raw = await r.get(f"match:{fixture_id}:state")
    if not state_raw:
        raise HTTPException(404, f"No match state for fixture {fixture_id}")

    state = MatchState.model_validate_json(state_raw)

    odds_client = get_oddsapi_client()
    odds_table = await odds_client.get_all_odds()
    p_win, p_draw, p_loss, source = await _pre_match_wdl(state, odds_table)

    red_h, red_a = red_counts(state)

    minute = state.elapsed or 0
    wdl = inplay_wdl(
        (p_win, p_draw, p_loss),
        minute,
        state.home_score,
        state.away_score,
        red_h,
        red_a,
        extra=state.elapsed_extra,
    )
    # Knockout ties: probability of going through, extra time and pens included.
    fx = FIXTURE_BY_ID.get(state.fixture_id) or {}
    home_advance = None
    if fx.get("stage", "group") != "group" and state.status_short not in COMPLETED_STATUSES:
        home_advance = round(
            live_ko_advance(
                (p_win, p_draw, p_loss),
                LiveMatch(minute, state.home_score, state.away_score, red_h, red_a, state.elapsed_extra),
            ),
            4,
        )

    return {
        "fixture_id": int(fixture_id),
        "elapsed": minute,
        "status_short": state.status_short,
        "home_win": round(wdl[0], 4),
        "draw": round(wdl[1], 4),
        "away_win": round(wdl[2], 4),
        "home_advance": home_advance,
        "pre_match_source": source,
    }


# ── Counterfactual narrator ────────────────────────────────────────────────


async def _pending_response(r, fixture_id: str) -> dict:
    """200 with status "pending" when no counterfactual exists yet (not a 404)."""
    state_raw = await r.get(f"match:{fixture_id}:state")
    match_status = None
    if state_raw:
        try:
            state = MatchState.model_validate_json(state_raw)
            match_status = state.status_short
        except Exception:
            pass
    return {
        "fixture_id": int(fixture_id),
        "status": "pending",
        "match_status": match_status,
        "entries": [],
        "message": "No counterfactual data yet — appears after a goal or red card.",
    }


@router.get("/{fixture_id}/counterfactual")
async def get_counterfactual(fixture_id: str, request: Request):
    """Latest counterfactual result for this fixture."""
    r = request.app.state.redis
    raw = await r.get(f"match:{fixture_id}:counterfactual:latest")
    if not raw:
        return await _pending_response(r, fixture_id)
    return json.loads(raw)


@router.get("/{fixture_id}/counterfactual/feed")
async def get_counterfactual_feed(fixture_id: str, request: Request):
    """Counterfactual history for this match (up to 20), newest first."""
    r = request.app.state.redis
    feed_raw = await r.lrange(f"match:{fixture_id}:counterfactual:feed", 0, FEED_MAX)
    if not feed_raw:
        return await _pending_response(r, fixture_id)
    return {"fixture_id": int(fixture_id), "entries": [json.loads(e) for e in feed_raw]}


@router.get("/{fixture_id}/counterfactual/stream")
async def counterfactual_stream(fixture_id: str, request: Request):
    """SSE, pub/sub-backed.

    "calculating" messages carry their own payload and are forwarded as
    "counterfactual_calculating" events; results use re-fetch-on-change.
    """
    r = request.app.state.redis

    async def generator():
        last_raw: str | None = None
        async with subscription(r, "counterfactual_update") as q:
            raw = await r.get(f"match:{fixture_id}:counterfactual:latest")
            if raw is not None:
                last_raw = raw
                yield {"event": "counterfactual_update", "data": raw}
            else:
                yield {
                    "event": "waiting",
                    "data": json.dumps({"message": "waiting for counterfactual data"}),
                }

            while True:
                if await request.is_disconnected():
                    break
                msg = await next_message(q, 10.0)
                if msg is None:
                    continue

                try:
                    payload = json.loads(msg)
                except Exception:
                    continue

                if str(payload.get("fixture_id")) != str(fixture_id):
                    continue

                if payload.get("status") == "calculating":
                    # Own payload IS the info — no key to re-fetch.
                    yield {
                        "event": "counterfactual_calculating",
                        "data": json.dumps(payload),
                    }
                    continue

                raw = await r.get(f"match:{fixture_id}:counterfactual:latest")
                if raw is not None and raw != last_raw:
                    last_raw = raw
                    yield {"event": "counterfactual_update", "data": raw}

    return EventSourceResponse(generator(), ping=15)


@router.api_route(
    "/{fixture_id}/counterfactual/trigger",
    methods=["GET", "POST"],
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_counterfactual(fixture_id: str, request: Request):
    """Force one counterfactual cycle for an uncovered goal / red card.

    Token-gated and idempotent: covered events return {"status": "skipped"}.
    """
    r = request.app.state.redis
    state_raw = await r.get(f"match:{fixture_id}:state")

    if not state_raw:
        raise HTTPException(404, f"No match state for fixture {fixture_id}")

    state = MatchState.model_validate_json(state_raw)

    async def _on_calc_start(info: dict) -> None:
        await r.publish(
            "counterfactual_update",
            json.dumps(
                {
                    "fixture_id": state.fixture_id,
                    "status": "calculating",
                    "minute": info["minute"],
                    "event_type": info["event_type"],
                    "event_team": info["event_team"],
                }
            ),
        )

    loop = asyncio.get_running_loop()
    result = await counterfactual_agent.update(state, loop, on_start=_on_calc_start)

    if result is None:
        events_with_types = [(ev.elapsed, ev.type) for ev in state.events[-5:]]
        return {
            "status": "skipped",
            "reason": "No uncovered trigger event found (already analysed, or none exists yet)",
            "recent_events": events_with_types,
        }

    ttl = _ttl_for(state.status_short)
    result_json = json.dumps(result)

    pipe = r.pipeline(transaction=True)
    pipe.setex(f"match:{fixture_id}:counterfactual:latest", ttl, result_json)
    pipe.lpush(f"match:{fixture_id}:counterfactual:feed", result_json)
    pipe.ltrim(f"match:{fixture_id}:counterfactual:feed", 0, FEED_MAX)
    pipe.expire(f"match:{fixture_id}:counterfactual:feed", ttl)
    await pipe.execute()
    await r.publish(
        "counterfactual_update", json.dumps({"fixture_id": result["fixture_id"]})
    )

    return {"status": "written", "result": result}
