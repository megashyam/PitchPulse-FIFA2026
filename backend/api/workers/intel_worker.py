"""
Match intelligence background worker.

Every 30 seconds, runs active fixtures through the match intel agent and
writes per-event history to Redis. Completed fixtures without a feed are
backfilled a few per tick.

Design:
    - Feed rewrites are a MULTI/EXEC transaction.
    - Colour entries dedupe on min:{minute}:{narration_type}.
    - Per-fixture agent memory is released when a fixture leaves
      matches:active.

Redis writes:
    match:{id}:intel:latest   JSON entry (newest)             TTL 3600s
    match:{id}:intel:feed     list (history, newest-first)    TTL 3600s

Pub/sub:
    channel intel_update      {fixture_id, minute}
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis

from agents import match_intel_agent
from agents.llm_queue import Priority, set_llm_priority
from api.schemas.event_types import COMPLETED_STATUSES, SIGNIFICANT_TYPES
from api.schemas.schema import MatchState
from monitoring.metrics import WORKER_ERRORS, WORKER_TICK_DURATION, tick_done

log = logging.getLogger(__name__)
INTERVAL = 30.0
FEED_CAP = 30
TTL_LIVE = 3_600  # 1 hour — match in progress
TTL_COMPLETED = 2_592_000  # 30 days — match history
BACKFILL_PER_TICK = 2  # completed fixtures without a feed, per tick

_known_fixtures: set[str] = set()
_backfilled: set[str] = set()  # completed fixtures already attempted this process


def _ttl_for(status_short: str) -> int:
    return TTL_COMPLETED if status_short in COMPLETED_STATUSES else TTL_LIVE


async def run(redis_client: aioredis.Redis) -> None:
    log.info("Intel worker started — every 30s (per-event history)")
    loop = asyncio.get_running_loop()
    while True:
        try:
            with WORKER_TICK_DURATION.labels("intel").time():
                await _update_all(redis_client, loop)
            tick_done("intel")
        except asyncio.CancelledError:
            log.info("Intel worker cancelled")
            raise
        except Exception as exc:
            WORKER_ERRORS.labels("intel").inc()
            log.error(f"Intel worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _update_all(r: aioredis.Redis, loop: asyncio.AbstractEventLoop) -> None:
    """Process matches:active, plus completed fixtures that have no feed yet."""
    active_ids = set(await r.smembers("matches:active"))
    completed_ids = set(await r.smembers("matches:completed"))

    # Release agent memory for fixtures that left the active set.
    for stale in list(_known_fixtures - active_ids):
        _known_fixtures.discard(stale)
        try:
            match_intel_agent.clear_state(int(stale))
        except (TypeError, ValueError):
            pass
    _known_fixtures.update(active_ids)

    # Backfill is throttled so a restart after the tournament doesn't queue
    # every finished match's narration at once.
    fixtures_to_process = set(active_ids)
    backfill = 0
    for cid in sorted(completed_ids - active_ids - _backfilled):
        if backfill >= BACKFILL_PER_TICK:
            break
        _backfilled.add(cid)
        if not await r.exists(f"match:{cid}:intel:feed"):
            fixtures_to_process.add(cid)
            backfill += 1

    fixture_ids = list(fixtures_to_process)
    if not fixture_ids:
        return
    results = await asyncio.gather(
        *[_update_fixture(r, fid, loop) for fid in fixture_ids],
        return_exceptions=True,
    )
    for fid, res in zip(fixture_ids, results):
        if isinstance(res, Exception):
            log.error(f"[{fid}] intel _update_fixture raised: {res}", exc_info=res)


def _load_feed(entries_raw):
    out = []
    for raw in entries_raw:
        try:
            out.append(json.loads(raw))
        except Exception:
            pass
    return out


def _colour_key(entry: dict) -> str:
    return f"min:{entry.get('minute')}:{entry.get('narration_type')}"


async def _update_fixture(
    r: aioredis.Redis,
    fid: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    state_raw = await r.get(f"match:{fid}:state")
    if not state_raw:
        return
    try:
        state = MatchState.model_validate_json(state_raw)
    except Exception as exc:
        log.warning(f"[{fid}] MatchState parse error: {exc}")
        return

    current_elapsed = state.elapsed or 0
    completed = state.status_short in COMPLETED_STATUSES
    is_live = state.status_short in ("1H", "2H", "ET", "P")
    # Task-local: each fixture runs in its own gather() task.
    set_llm_priority(Priority.BACKGROUND if completed else Priority.LIVE)

    # ── Existing feed + replay-restart guard ──────────────────────────────
    existing = _load_feed(await r.lrange(f"match:{fid}:intel:feed", 0, FEED_CAP - 1))

    newest_cached_minute = max((e.get("minute", 0) for e in existing), default=0)
    if newest_cached_minute > current_elapsed + 5 and not completed:
        log.info(
            f"[{fid}] Replay restart — cached {newest_cached_minute}' > "
            f"elapsed {current_elapsed}'. Flushing feed."
        )
        await r.delete(f"match:{fid}:intel:latest")
        await r.delete(f"match:{fid}:intel:feed")
        match_intel_agent.clear_state(state.fixture_id)
        existing = []

    have_event_sigs = {e.get("event_sig") for e in existing if e.get("event_sig")}
    have_colour_keys = {_colour_key(e) for e in existing if not e.get("event_sig")}

    momentum_raw = await r.get(f"match:{fid}:momentum")
    momentum = json.loads(momentum_raw) if momentum_raw else None

    new_entries: list[dict] = []

    # ── 1. Per-event history: one entry per goal / red not yet analysed ────
    sig_events = [
        ev
        for ev in sorted(state.events, key=lambda e: e.elapsed)
        if ev.type in SIGNIFICANT_TYPES and ev.elapsed <= current_elapsed + 2
    ]
    for ev in sig_events:
        sig = f"{ev.elapsed}:{ev.type}:{ev.team_id}"
        if sig in have_event_sigs:
            continue
        try:
            new_entries.append(await match_intel_agent.analyze_event(state, ev, loop))
            have_event_sigs.add(sig)
        except Exception as exc:
            log.warning(f"[{fid}] analyze_event failed @{ev.elapsed}': {exc}")

    # ── 2. Live tactical / xG colour for the current clock ────────────────
    if is_live:
        try:
            result = await match_intel_agent.update(state, momentum, loop)
        except Exception as exc:
            log.warning(f"[{fid}] intel update() failed: {exc}")
            result = None
        if (
            result
            and result.get("narration_type") != "event_reaction"
            and _colour_key(result) not in have_colour_keys
            and result.get("minute", 0) <= current_elapsed + 2
        ):
            new_entries.append(result)
            have_colour_keys.add(_colour_key(result))

    # ── 3. Full-time wrap-up for completed matches with no event narration ─
    # A finished match that produced no goals/red cards (e.g. a 0-0) would
    # otherwise have an empty feed forever. Generate a single full-match
    # summary so the Live Intelligence panel always has SOMETHING for a
    # completed match (image: "No AI narration for this match yet"). Keyed by
    # a stable ft_summary sig so it's produced at most once.
    if completed:
        ft_sig = f"ft_summary:{state.home_score}:{state.away_score}"
        already_have_ft = any(e.get("event_sig") == ft_sig for e in existing)
        # Only bother if there are no event-reaction entries either — if the
        # match had goals, those already tell the story.
        has_event_narration = bool(have_event_sigs) or any(
            e.get("event_sig") for e in existing
        )
        if not already_have_ft and not has_event_narration and not new_entries:
            try:
                new_entries.append(
                    await match_intel_agent.analyze_full_time_summary(state, loop)
                )
            except Exception as exc:
                log.warning(f"[{fid}] FT summary generation failed: {exc}")

    if not new_entries:
        return
    merged: dict[str, dict] = {}
    for e in existing + new_entries:
        key = e.get("event_sig") or _colour_key(e)
        merged[key] = e  # last write wins
    ordered = sorted(merged.values(), key=lambda e: e.get("minute", 0), reverse=True)[
        :FEED_CAP
    ]

    ttl = _ttl_for(state.status_short)

    # transaction=True → readers see either the old feed or the new one,
    # never the empty intermediate.
    pipe = r.pipeline(transaction=True)
    pipe.delete(f"match:{fid}:intel:feed")
    for e in ordered:
        pipe.rpush(f"match:{fid}:intel:feed", json.dumps(e))
    pipe.expire(f"match:{fid}:intel:feed", ttl)
    pipe.setex(f"match:{fid}:intel:latest", ttl, json.dumps(ordered[0]))
    await pipe.execute()

    await r.publish(
        "intel_update",
        json.dumps(
            {"fixture_id": state.fixture_id, "minute": ordered[0].get("minute", 0)}
        ),
    )
    log.info(
        f"[{fid}] Intel: +{len(new_entries)} entrie(s), feed={len(ordered)} "
        f"(completed={completed})"
    )
