"""
Match intelligence endpoints.

    GET      /matches/{id}/intel          latest feed (last 30 entries)
    GET      /matches/{id}/intel/stream   SSE (pub/sub-backed)
    GET/POST /matches/{id}/intel/trigger  debug: force one cycle (token-gated)
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from agents import match_intel_agent
from api.routes._security import require_trigger_token
from api.routes._sse import pubsub_sse
from api.schemas.event_types import COMPLETED_STATUSES, LIVE_STATUSES, SIGNIFICANT_TYPES
from api.schemas.schema import MatchState

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/{fixture_id}/intel")
async def get_intel_feed(fixture_id: str, request: Request):
    """Last 30 narrative entries for this fixture, newest first.

    Entries ahead of the current match minute are dropped (stale replay).
    Not-started matches return 200 {"status": "not_started"}.
    """
    r = request.app.state.redis
    feed_raw = await r.lrange(f"match:{fixture_id}:intel:feed", 0, 29)

    if not feed_raw:
        state_raw = await r.get(f"match:{fixture_id}:state")
        match_status = None
        if state_raw:
            try:
                state = MatchState.model_validate_json(state_raw)
                match_status = state.status_short
                if (
                    state.status_short not in LIVE_STATUSES
                    and state.status_short not in COMPLETED_STATUSES
                ):
                    return {
                        "fixture_id": int(fixture_id),
                        "status": "not_started",
                        "match_status": match_status,
                        "entries": [],
                        "message": "AI narration begins at kickoff.",
                    }
            except Exception:
                pass
        # Live or completed but nothing generated yet: narration needs ~5
        # match minutes or a goal / red card. 200, not 404.
        return {
            "fixture_id": int(fixture_id),
            "status": "pending",
            "match_status": match_status,
            "entries": [],
            "message": "No AI narration yet — appears at minute 5 or after a goal/red card.",
        }

    current_elapsed = 999  # permissive default if state unavailable
    state_raw = await r.get(f"match:{fixture_id}:state")
    if state_raw:
        try:
            state = MatchState.model_validate_json(state_raw)
            # Completed matches: never filter by elapsed. A finished match's
            # elapsed often reads 0/null, which would wrongly drop every
            # real entry (all at minute ~90) as "stale" — the exact bug
            # behind intel showing empty for FT matches despite valid data.
            if state.status_short in COMPLETED_STATUSES:
                current_elapsed = 999
            else:
                current_elapsed = state.elapsed or 0
        except Exception:
            pass

    entries = []
    for e in feed_raw:
        try:
            entry = json.loads(e)
            if entry.get("minute", 0) <= current_elapsed + 5:
                entries.append(entry)
        except Exception:
            pass

    if not entries:
        return {
            "fixture_id": int(fixture_id),
            "status": "pending",
            "entries": [],
            "message": (
                f"Intel feed exists but all entries are currently stale "
                f"(minute > {current_elapsed}) — will repopulate as the "
                "match progresses."
            ),
        }

    return {
        "fixture_id": int(fixture_id),
        "entries": entries,
        "updated_at": entries[0].get("updated_at") if entries else None,
    }


async def _emit_ok_factory(r, fixture_id: str):
    async def _emit_ok(raw: str) -> bool:
        try:
            entry = json.loads(raw)
            entry_minute = entry.get("minute", 0)
            state_raw = await r.get(f"match:{fixture_id}:state")
            current_elapsed = 999
            if state_raw:
                state = MatchState.model_validate_json(state_raw)
                current_elapsed = state.elapsed or 0
            return entry_minute <= current_elapsed + 5
        except Exception:
            return True

    return _emit_ok


@router.get("/{fixture_id}/intel/stream")
async def intel_stream(fixture_id: str, request: Request):
    """SSE, pub/sub-backed, with the same staleness guard as the REST route."""
    r = request.app.state.redis
    emit_ok = await _emit_ok_factory(r, fixture_id)

    async def generator():
        async for event in pubsub_sse(
            redis_client=r,
            channel="intel_update",
            key=f"match:{fixture_id}:intel:latest",
            event_name="intel_update",
            is_disconnected=request.is_disconnected,
            match_fixture_id=fixture_id,
            emit_ok=emit_ok,
        ):
            yield event

    return EventSourceResponse(generator(), ping=15)


@router.api_route(
    "/{fixture_id}/intel/trigger",
    methods=["GET", "POST"],
    dependencies=[Depends(require_trigger_token)],
)
async def trigger_intel(fixture_id: str, request: Request):
    """Run one full intel cycle for this fixture, as intel_worker does.

    Covers event reactions, live colour and the FT wrap-up. Token-gated;
    returns per-section status.
    """
    r = request.app.state.redis
    state_raw = await r.get(f"match:{fixture_id}:state")

    if not state_raw:
        raise HTTPException(
            404,
            detail=f"No match state for fixture {fixture_id}. Is the producer running?",
        )

    try:
        state = MatchState.model_validate_json(state_raw)
    except Exception as exc:
        return {"status": "error", "stage": "parse_match_state", "error": str(exc)}

    current_elapsed = state.elapsed or 0
    completed = state.status_short in COMPLETED_STATUSES
    is_live = state.status_short in ("1H", "2H", "ET", "P")

    existing_raw = await r.lrange(f"match:{fixture_id}:intel:feed", 0, 29)
    existing = []
    for raw in existing_raw:
        try:
            existing.append(json.loads(raw))
        except Exception:
            pass
    have_event_sigs = {e.get("event_sig") for e in existing if e.get("event_sig")}

    new_entries: list[dict] = []
    section_status: dict[str, str] = {}

    # ── Section 1: per-event history for uncovered goals/reds ─────────────
    sig_events = [
        ev
        for ev in sorted(state.events, key=lambda e: e.elapsed)
        if ev.type in SIGNIFICANT_TYPES and ev.elapsed <= current_elapsed + 2
    ]
    event_errors = []
    for ev in sig_events:
        sig = f"{ev.elapsed}:{ev.type}:{ev.team_id}"
        if sig in have_event_sigs:
            continue
        try:
            new_entries.append(await match_intel_agent.analyze_event(state, ev))
        except Exception as exc:
            event_errors.append(f"{ev.elapsed}'  {ev.type}: {exc}")
    section_status["event_history"] = (
        f"{len([e for e in new_entries])} new / {len(sig_events)} total significant events"
        + (f" — {len(event_errors)} FAILED: {event_errors}" if event_errors else "")
    )

    # ── Section 2: live tactical/xG colour ─────────────────────────────────
    if is_live:
        momentum_raw = await r.get(f"match:{fixture_id}:momentum")
        momentum = json.loads(momentum_raw) if momentum_raw else None
        match_intel_agent.clear_state(state.fixture_id)
        try:
            colour = await match_intel_agent.update(state, momentum)
            if colour:
                new_entries.append(colour)
                section_status["colour"] = "generated"
            else:
                section_status["colour"] = "skipped (score below threshold)"
        except Exception as exc:
            section_status["colour"] = f"FAILED: {exc}"
    else:
        section_status["colour"] = "skipped (match not live)"

    # ── Section 3: FT wrap-up if completed with no narration at all ───────
    if completed and not new_entries and not have_event_sigs:
        try:
            new_entries.append(
                await match_intel_agent.analyze_full_time_summary(state)
            )
            section_status["ft_summary"] = "generated"
        except Exception as exc:
            section_status["ft_summary"] = f"FAILED: {exc}"
    else:
        section_status["ft_summary"] = "skipped (not applicable)"

    if not new_entries:
        return {
            "status": "no_new_entries",
            "sections": section_status,
            "elapsed": state.elapsed,
            "status_short": state.status_short,
        }

    for entry in new_entries:
        entry_json = json.dumps(entry)
        await r.setex(f"match:{fixture_id}:intel:latest", 3600, entry_json)
        await r.lpush(f"match:{fixture_id}:intel:feed", entry_json)
    await r.ltrim(f"match:{fixture_id}:intel:feed", 0, 29)
    await r.expire(f"match:{fixture_id}:intel:feed", 3600)
    await r.publish("intel_update", json.dumps({"fixture_id": state.fixture_id}))

    return {
        "status": "written",
        "entries_written": len(new_entries),
        "sections": section_status,
        "entries": new_entries,
    }
