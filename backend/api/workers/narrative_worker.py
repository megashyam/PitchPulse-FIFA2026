"""
Narrative background worker.

Surge detection every 60s, a tournament-wide trending snapshot and top-N
arc synthesis. Scorer baselines persist in Redis, so a restart needs no
fresh warm-up.
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis

from agents import narrative_arc_agent
from agents.llm_queue import Priority, llm_priority
from agents.narrative_spike_detector import NarrativeSpike, get_detector
from api.routes.narrative_comments import store_comment_samples
from monitoring.metrics import WORKER_ERRORS, WORKER_TICK_DURATION, tick_done

log = logging.getLogger(__name__)
INTERVAL = 60.0
# LLM arcs for the top N trending stories each tick; the on-demand arc
# endpoint (api/routes/narrative.py) covers the rest.
ARC_TOP_N = 6
ARC_CACHE_TTL = 1_800  # an unchanged surge keeps its arc for 30 min
STATE_KEY = "narrative:scorer:state"
STATE_TTL = 3_600
STATE_EVERY = 5  # ticks


async def run(redis_client: aioredis.Redis) -> None:
    log.info("Narrative worker started — surge detection every 60s")
    loop = asyncio.get_running_loop()
    detector = get_detector()
    await load_state(redis_client, detector)

    while True:
        try:
            with WORKER_TICK_DURATION.labels("narrative").time():
                await _tick(redis_client, loop, detector)
            tick_done("narrative")
        except asyncio.CancelledError:
            log.info("Narrative worker cancelled")
            raise
        except Exception as exc:
            WORKER_ERRORS.labels("narrative").inc()
            log.error(f"Narrative worker error: {exc}", exc_info=True)
        await asyncio.sleep(INTERVAL)


async def load_state(r: aioredis.Redis, detector) -> bool:
    try:
        raw = await r.get(STATE_KEY)
        if not raw:
            return False
        detector.load_state(json.loads(raw))
        log.info(
            f"Narrative scorer state restored (warm-up left: "
            f"{detector.warmup_remaining()} ticks)"
        )
        return True
    except Exception as exc:
        log.warning(f"Narrative scorer state restore failed: {exc}")
        return False


async def save_state(r: aioredis.Redis, detector) -> None:
    try:
        await r.setex(STATE_KEY, STATE_TTL, json.dumps(detector.export_state()))
    except Exception as exc:
        log.debug(f"Narrative scorer state save failed: {exc}")


async def _store_comment_samples_for_all_topics(r: aioredis.Redis, detector) -> None:
    stored_count = 0
    for topic in detector.topics:
        try:
            samples = detector.get_last_samples(topic)
            if samples:
                await store_comment_samples(r, topic, samples)
                stored_count += len(samples)
        except Exception as e:
            log.debug(f"Comment sample storage failed for '{topic}': {e}")

    if stored_count:
        log.info(
            f"Narrative worker: stored {stored_count} comment samples across topics"
        )


async def _add_arcs_to_top_trending(
    r: aioredis.Redis, snapshot: list, loop: asyncio.AbstractEventLoop
) -> None:
    """Arcs for the top trending rows.

    Rows with surging sources get an LLM arc at background priority, cached
    by (topic, surging set); quiet rows get the template.
    """
    for row in snapshot[:ARC_TOP_N]:
        try:
            spike = NarrativeSpike(
                spike_id=row["spike_id"],
                topic=row["topic"],
                tick=row["tick"],
                severity=row["severity"],
                sources=row["sources"],
                source_names=row["source_names"],
                summary=row["summary"],
                timestamp=row["timestamp"],
                data_sources=row.get("data_sources"),
                z_scores=row.get("z_scores"),
            )
            surging = sorted(row["source_names"] or [])
            if not surging:
                row["arc"] = narrative_arc_agent.template_arc(spike)
                continue
            key = f"narrative:arc:cache:{row['topic']}:{','.join(surging)}"
            cached = await r.get(key)
            if cached:
                row["arc"] = cached
                continue
            with llm_priority(Priority.BACKGROUND):
                row["arc"] = await narrative_arc_agent.synthesise(spike, loop)
            await r.setex(key, ARC_CACHE_TTL, row["arc"])
        except Exception as exc:
            topic_name = row.get("topic")
            log.debug(f"trending arc synthesis failed for {topic_name}: {exc}")


async def _tick(r: aioredis.Redis, loop: asyncio.AbstractEventLoop, detector) -> None:
    spikes = await detector.tick(loop)
    if detector._tick_count % STATE_EVERY == 0:
        await save_state(r, detector)

    await _store_comment_samples_for_all_topics(r, detector)

    try:
        snapshot = detector.trending(top_n=12)
        await _add_arcs_to_top_trending(r, snapshot, loop)
        await r.setex("narrative:trending:latest", 3_600, json.dumps(snapshot))
    except Exception as exc:
        log.debug(f"trending snapshot failed: {exc}")

    if not spikes:
        return

    for spike in spikes:
        try:
            arc = await narrative_arc_agent.synthesise(spike, loop)
            spike.arc = arc
        except Exception as e:
            log.warning(f"Arc synthesis failed for {spike.spike_id}: {e}")
            spike.arc = None

        spike_dict = spike.to_dict()
        spike_json = json.dumps(spike_dict)

        await r.setex(f"narrative:spike:{spike.spike_id}", 86_400, spike_json)
        await r.lpush("narrative:spikes:feed", spike_json)
        await r.ltrim("narrative:spikes:feed", 0, 49)
        await r.expire("narrative:spikes:feed", 86_400)
        await r.setex("narrative:stream:latest", 3_600, spike_json)

        notif = json.dumps(
            {
                "spike_id": spike.spike_id,
                "topic": spike.topic,
                "severity": spike.severity,
            }
        )
        await r.publish("narrative_spike", notif)

        log.info(
            f"Narrative spike stored: {spike.topic} id={spike.spike_id} "
            f"severity={spike.severity:.2f}"
        )
