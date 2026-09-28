"""
Team form endpoint.

    GET /matches/{fixture_id}/team-form

One SUNION(active, completed) plus one MGET; form is sorted by kickoff_time.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas.event_types import COMPLETED_STATUSES
from api.schemas.schema import MatchState

router = APIRouter()


@router.get("/{fixture_id}/team-form")
async def team_form(fixture_id: str, request: Request):
    r = request.app.state.redis

    raw = await r.get(f"match:{fixture_id}:state")
    if not raw:
        raise HTTPException(404, "Fixture not found")

    state = MatchState.model_validate_json(raw)
    home_name = state.home_name
    away_name = state.away_name

    all_ids = await r.sunion("matches:active", "matches:completed")
    keys = [f"match:{fid}:state" for fid in all_ids if str(fid) != str(fixture_id)]
    raw_values = await r.mget(keys) if keys else []

    home_form = []
    away_form = []

    for match_raw in raw_values:
        if not match_raw:
            continue
        try:
            m = MatchState.model_validate_json(match_raw)
        except Exception:
            continue

        if m.status_short not in COMPLETED_STATUSES:
            continue

        hs, as_ = m.home_score, m.away_score

        def result_for(team: str) -> str:
            is_home = team == m.home_name
            if is_home:
                if hs > as_:
                    return "W"
                if as_ > hs:
                    return "L"
                return "D"
            else:
                if as_ > hs:
                    return "W"
                if hs > as_:
                    return "L"
                return "D"

        sort_key = m.kickoff_time.isoformat() if m.kickoff_time else ""

        if m.home_name == home_name or m.away_name == home_name:
            home_form.append({
                "opponent": m.away_name if m.home_name == home_name else m.home_name,
                "home_score": hs,
                "away_score": as_,
                "result": result_for(home_name),
                "was_home": m.home_name == home_name,
                "date": sort_key or None,
            })

        if m.home_name == away_name or m.away_name == away_name:
            away_form.append({
                "opponent": m.away_name if m.home_name == away_name else m.home_name,
                "home_score": hs,
                "away_score": as_,
                "result": result_for(away_name),
                "was_home": m.home_name == away_name,
                "date": sort_key or None,
            })

    home_form.sort(key=lambda x: x.get("date") or "", reverse=True)
    away_form.sort(key=lambda x: x.get("date") or "", reverse=True)

    return {
        "home": {"team": home_name, "form": home_form[:5]},
        "away": {"team": away_name, "form": away_form[:5]},
    }
