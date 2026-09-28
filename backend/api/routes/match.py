"""
Match state endpoints.

    GET /matches/                active + completed fixture IDs
    GET /matches/summary         every active + completed state (one SUNION
                                 + one MGET)
    GET /matches/{fixture_id}    single fixture state

/summary must be registered before /{fixture_id}.
"""

import json

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/")
async def list_matches(request: Request):
    r = request.app.state.redis
    active = await r.smembers("matches:active")
    completed = await r.smembers("matches:completed")
    return {"fixtures": sorted(active | completed)}


@router.get("/summary")
async def match_summary(request: Request):
    """All active and recently completed fixtures in one round trip."""
    r = request.app.state.redis
    fixture_ids = await r.sunion("matches:active", "matches:completed")
    if not fixture_ids:
        return {"fixtures": []}

    keys = [f"match:{fid}:state" for fid in fixture_ids]
    raw_values = await r.mget(keys)

    fixtures = []
    for raw in raw_values:
        if raw is None:
            continue
        try:
            fixtures.append(json.loads(raw))
        except Exception:
            continue

    return {"fixtures": fixtures}


@router.get("/{fixture_id}")
async def get_match(fixture_id: str, request: Request):
    r = request.app.state.redis
    raw = await r.get(f"match:{fixture_id}:state")
    if not raw:
        raise HTTPException(
            status_code=404,
            detail=f"Fixture '{fixture_id}' not found. Is the hybrid producer running?",
        )

    return json.loads(raw)
