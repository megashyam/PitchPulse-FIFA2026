"""
Tactical fingerprint endpoints.

    GET      /matches/{id}/tactical          latest tactical fingerprint match
    GET/POST /matches/{id}/tactical/trigger  force a fresh match (token-gated)

compute_and_cache() is shared with api/workers/tactical_worker.py, which
retries every fixture periodically, so matches appear once indexing
(ml/tactical_indexer.ensure_indexed) finishes.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from agents import tactical_agent
from agents.weaviate_client import get_weaviate_client, TACTICAL_PROFILES
from api.routes._security import require_trigger_token
from api.schemas.event_types import COMPLETED_STATUSES, LIVE_STATUSES
from api.schemas.schema import MatchState

router = APIRouter()
log = logging.getLogger(__name__)

CACHE_TTL = 600  # 10 min — fingerprints drift slowly within a match
CACHE_TTL_COMPLETED = 30 * 86_400  # a finished match never changes


def _indexed_count(wv) -> int:
    return wv.get_count(TACTICAL_PROFILES) if wv.ready else 0


async def compute_and_cache(r, fixture_id: str, state: MatchState, loop) -> dict:
    """Compute and cache fingerprint matches for both teams."""
    home_match, away_match = await asyncio.gather(
        tactical_agent.match_team(state, "home", loop),
        tactical_agent.match_team(state, "away", loop),
    )

    if home_match is None and away_match is None:
        # Distinguish WHY, so the frontend can show something more useful
        # than a permanent "run the indexer yourself" instruction.
        wv = get_weaviate_client()
        indexed_count = await asyncio.to_thread(_indexed_count, wv)
        reason = (
            "weaviate_unavailable"
            if not wv.ready
            else (
                "indexing_in_progress"
                if indexed_count == 0
                else "no_match_above_threshold"
            )
        )
        return {
            "status": "skipped",
            "reason": reason,
            "indexed_count": indexed_count,
        }

    result = {
        "fixture_id": int(fixture_id),
        "home_name": state.home_name,
        "away_name": state.away_name,
        "home": home_match,
        "away": away_match,
        "source": "TacticalProfiles · style band + nearest possession",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    ttl = CACHE_TTL_COMPLETED if state.status_short in COMPLETED_STATUSES else CACHE_TTL
    await r.setex(f"match:{fixture_id}:tactical", ttl, json.dumps(result))
    log.info(f"[{fixture_id}] Tactical fingerprint cached")
    return {"status": "written", "result": result}


@router.get("/{fixture_id}/tactical")
async def get_tactical(fixture_id: str, request: Request):
    """Cached tactical fingerprint match for this fixture.

    Not-started matches return 200 {"status": "not_started"}.
    """
    r = request.app.state.redis
    raw = await r.get(f"match:{fixture_id}:tactical")
    if not raw:
        state_raw = await r.get(f"match:{fixture_id}:state")
        if state_raw:
            try:
                state = MatchState.model_validate_json(state_raw)
                if (
                    state.status_short not in LIVE_STATUSES
                    and state.status_short not in COMPLETED_STATUSES
                ):
                    return {
                        "fixture_id": int(fixture_id),
                        "status": "not_started",
                        "message": "Tactical fingerprints begin at kickoff.",
                    }
            except Exception:
                pass
        wv = get_weaviate_client()
        indexed_count = await asyncio.to_thread(_indexed_count, wv)
        message = (
            "TacticalProfiles is still being indexed in the background "
            "(this takes a few minutes on first run) — check back shortly."
            if indexed_count == 0
            else "No tactical fingerprint yet — the background worker refreshes "
            "this every ~2 minutes."
        )
        # 200, not 404: this is the normal state right after kickoff (worker
        # hasn't had its first successful tick yet) or during first-run
        # indexing — both self-resolve, and a 404 always shows as a red
        # console error regardless of how the frontend handles it.
        return {
            "fixture_id": int(fixture_id),
            "status": "pending",
            "message": message,
        }
    return json.loads(raw)


@router.api_route(
    "/{fixture_id}/tactical/trigger",
    methods=["GET", "POST"],
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_tactical(fixture_id: str, request: Request):
    """Force an immediate tactical refresh for this fixture."""
    r = request.app.state.redis
    state_raw = await r.get(f"match:{fixture_id}:state")
    if not state_raw:
        raise HTTPException(
            404, f"No match state for fixture {fixture_id}. Is the producer running?"
        )

    state = MatchState.model_validate_json(state_raw)
    loop = asyncio.get_running_loop()
    return await compute_and_cache(r, fixture_id, state, loop)
