import asyncio
import json
import logging
import math

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from agents import counterfactual_agent
from api.routes._security import require_trigger_token
from api.schemas.event_types import COMPLETED_STATUSES
from api.schemas.schema import MatchState
from ml.in_play import inplay_wdl
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import build_prior_table, elo_to_wdl
from ml.wc_2026_config import TEAM_BY_NAME

router = APIRouter()
log = logging.getLogger(__name__)

TTL_LIVE = 3_600
TTL_COMPLETED = 2_592_000
FEED_MAX = 19


def _ttl_for(status_short: str) -> int:
    return TTL_COMPLETED if status_short in COMPLETED_STATUSES else TTL_LIVE


def _ci(p: float, n: int = 500):
    margin = 1.96 * math.sqrt(p * (1 - p) / n)
    return round(max(0.0, p - margin), 4), round(min(1.0, p + margin), 4)


async def _pre_match_wdl(state: MatchState, odds_table):
    priors = build_prior_table(odds_table)
    key = (state.home_name, state.away_name)
    rev_key = (state.away_name, state.home_name)

    if key in priors:
        p_win, p_draw, p_loss = priors[key]
        source = "market_odds" if odds_table.get(key) else "elo"
    elif rev_key in priors:
        p_l, p_d, p_w = priors[rev_key]
        p_win, p_draw, p_loss = p_w, p_d, p_l
        source = "market_odds" if odds_table.get(rev_key) else "elo"
    else:
        h = TEAM_BY_NAME.get(state.home_name)
        a = TEAM_BY_NAME.get(state.away_name)
        p_win, p_draw, p_loss = (
            elo_to_wdl(h.elo, a.elo) if h and a else (0.40, 0.25, 0.35)
        )
        source = "elo"
    return p_win, p_draw, p_loss, source


@router.get("/{fixture_id}/prediction")
async def get_match_prediction(fixture_id: str, request: Request):
    """Per-match pre-tournament W/D/L probabilities from Betfair/Elo prior,
    plus current tournament path implications from the MC simulation."""
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
            "home_win": {
                "p": round(p_win, 4),
                "ci_lo": _ci(p_win)[0],
                "ci_hi": _ci(p_win)[1],
            },
            "draw": {
                "p": round(p_draw, 4),
                "ci_lo": _ci(p_draw)[0],
                "ci_hi": _ci(p_draw)[1],
            },
            "away_win": {
                "p": round(p_loss, 4),
                "ci_lo": _ci(p_loss)[0],
                "ci_hi": _ci(p_loss)[1],
            },
        },
        "source": source,
        "home_tournament": home_tournament,
        "away_tournament": away_tournament,
    }


@router.get("/{fixture_id}/live-prob")
async def get_live_prob(fixture_id: str, request: Request):
    """In-play W/D/L for the CURRENT match minute/score/reds, backed by
    ml/in_play.py — the same model the counterfactual agent conditions on.

    """
    r = request.app.state.redis
    state_raw = await r.get(f"match:{fixture_id}:state")
    if not state_raw:
        raise HTTPException(404, f"No match state for fixture {fixture_id}")

    state = MatchState.model_validate_json(state_raw)

    odds_client = get_oddsapi_client()
    odds_table = await odds_client.get_all_odds()
    p_win, p_draw, p_loss, source = await _pre_match_wdl(state, odds_table)

    red_h = sum(
        1
        for ev in state.events
        if ev.type in ("red", "yellow_red") and ev.team_name == state.home_name
    )
    red_a = sum(
        1
        for ev in state.events
        if ev.type in ("red", "yellow_red") and ev.team_name == state.away_name
    )

    minute = state.elapsed or 0
    wdl = inplay_wdl(
        (p_win, p_draw, p_loss),
        minute,
        state.home_score,
        state.away_score,
        red_h,
        red_a,
    )

    return {
        "fixture_id": int(fixture_id),
        "elapsed": minute,
        "status_short": state.status_short,
        "home_win": round(wdl[0], 4),
        "draw": round(wdl[1], 4),
        "away_win": round(wdl[2], 4),
        "pre_match_source": source,
    }


async def _pending_response(r, fixture_id: str) -> dict:

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
    """Full counterfactual history for this match — up to 20 entries,
    newest first."""
    r = request.app.state.redis
    feed_raw = await r.lrange(f"match:{fixture_id}:counterfactual:feed", 0, FEED_MAX)
    if not feed_raw:
        return await _pending_response(r, fixture_id)
    return {"fixture_id": int(fixture_id), "entries": [json.loads(e) for e in feed_raw]}


@router.get("/{fixture_id}/counterfactual/stream")
async def counterfactual_stream(fixture_id: str, request: Request):

    r = request.app.state.redis

    async def generator():
        pubsub = r.pubsub()
        await pubsub.subscribe("counterfactual_update")
        last_raw: str | None = None
        try:
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
                try:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=10.0
                    )
                except Exception:
                    msg = None

                if msg is None:
                    continue

                try:
                    payload = json.loads(msg["data"])
                except Exception:
                    continue

                if str(payload.get("fixture_id")) != str(fixture_id):
                    continue

                if payload.get("status") == "calculating":

                    yield {
                        "event": "counterfactual_calculating",
                        "data": json.dumps(payload),
                    }
                    continue

                raw = await r.get(f"match:{fixture_id}:counterfactual:latest")
                if raw is not None and raw != last_raw:
                    last_raw = raw
                    yield {"event": "counterfactual_update", "data": raw}
        finally:
            await pubsub.unsubscribe("counterfactual_update")
            await pubsub.aclose()

    return EventSourceResponse(generator(), ping=15)


@router.api_route(
    "/{fixture_id}/counterfactual/trigger",
    methods=["GET", "POST"],
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_counterfactual(fixture_id: str, request: Request):
    """Force one counterfactual cycle when the fixture has new uncovered events."""
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
