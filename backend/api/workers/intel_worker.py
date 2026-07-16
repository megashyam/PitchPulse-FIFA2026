"""
Match intelligence background worker.

This worker continuously enriches live fixtures with AI-generated match
analysis by combining canonical MatchState data, event streams, and derived
match context.

The worker orchestrates the match intelligence agent pipeline by:
    - monitoring active and recently completed fixtures
    - generating event-driven narratives for goals, cards, and substitutions
    - producing periodic live match insights
    - generating full-time summaries when required
    - maintaining a Redis-backed historical intelligence feed

Architecture:
    Match producer
        ↓
    Redis MatchState
        ↓
    Intel worker
        ↓
    Match Intelligence Agent
        ↓
    Redis intel feed + Pub/Sub updates

Redis storage:
    match:{id}:intel:latest
        Latest generated intelligence entry

    match:{id}:intel:feed
        Chronological fixture intelligence history

The worker is designed for long-running async execution with:
    - idempotent updates
    - replay detection
    - duplicate event prevention
    - graceful failure recovery
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis

from agents import match_intel_agent
from api.schemas.event_types import COMPLETED_STATUSES, SIGNIFICANT_TYPES
from api.schemas.schema import MatchState

log = logging.getLogger(__name__)
INTERVAL = 30.0
FEED_CAP = 30
TTL_LIVE = 3_600
TTL_COMPLETED = 2_592_000

_known_fixtures: set[str] = set()


def _ttl_for(status_short: str) -> int:
    return TTL_COMPLETED if status_short in COMPLETED_STATUSES else TTL_LIVE


async def run(redis_client: aioredis.Redis) -> None:
    """
    Start the background match intelligence processing loop.

    The worker periodically scans available fixtures and triggers intelligence
    generation for active matches and incomplete completed-match backfills.

    Failures are isolated per cycle so transient agent, Redis, or parsing
    failures do not terminate the long-running worker.
    """
    log.info("Intel worker started — every 30s (per-event history)")
    loop = asyncio.get_running_loop()
    while True:
        try:
            await _update_all(redis_client, loop)
        except asyncio.CancelledError:
            log.info("Intel worker cancelled")
            return
        except Exception as exc:
            log.error(f"Intel worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def _update_all(r: aioredis.Redis, loop: asyncio.AbstractEventLoop) -> None:
    """
    Discover fixtures requiring intelligence updates.

    The scheduler scans:
        - active fixtures for continuous live analysis
        - completed fixtures missing intelligence feeds for backfill

    Uses Redis indexes to avoid unnecessary processing and maintains agent
    state cleanup for fixtures that leave active play.

    Multiple fixtures are processed concurrently using asyncio gather to allow
    scalable multi-match intelligence generation.
    """
    active_ids = set(await r.smembers("matches:active"))
    completed_ids = await r.smembers("matches:completed")

    # Drop agent state for fixtures that are no longer active.
    for stale in list(_known_fixtures - active_ids):
        _known_fixtures.discard(stale)
        try:
            match_intel_agent.clear_state(int(stale))
        except (TypeError, ValueError):
            pass
    _known_fixtures.update(active_ids)

    fixtures_to_process = set(active_ids)
    for cid in completed_ids:
        if not await r.exists(f"match:{cid}:intel:feed"):
            fixtures_to_process.add(cid)

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
    """
    Generate and persist intelligence updates for a single fixture.

    Processing pipeline:

        1. Load canonical MatchState from Redis.
        2. Detect replay/restart scenarios.
        3. Recover existing intelligence history.
        4. Analyze significant match events.
        5. Generate live tactical/momentum narratives.
        6. Produce full-time summaries when required.
        7. Merge, deduplicate, and persist the intelligence timeline.

    The function is idempotent: previously generated insights are detected
    using event signatures and narration keys to prevent duplicate output.
    """
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

    if completed:
        ft_sig = f"ft_summary:{state.home_score}:{state.away_score}"
        already_have_ft = any(e.get("event_sig") == ft_sig for e in existing)

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
