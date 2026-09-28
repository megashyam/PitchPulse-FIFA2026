"""
Live match state producer.

Real WC 2026 match state from ESPN's public API, with the committed snapshot
(data/wc2026) as the permanent fallback.

Each poll:
    1. Scoreboard (one request): every fixture's score, status and clock.
       If ESPN is unreachable, fixtures come from the snapshot.
    2. Detail per fixture:
           snapshotted + completed      snapshot file (no request)
           live                         ESPN summary every poll
           completed, not in snapshot   ESPN summary once, then cached
       Detail: team stats, key events (goals/cards/subs with players),
       confirmed lineups, shots with model xG (ml/shot_xg.py).
    3. Persist MatchState; publish "match_update" only on real change.

Redis keys:
    match:{id}:state     MatchState JSON       TTL 12h (NS) / 1h, refreshed per poll
    match:{id}:lineups   confirmed lineups     TTL 12h
    match:{id}:shots     shot list with xG     TTL 12h
    match:{id}:ft_at     first FT sighting (active-set grace when kickoff is unknown)
    matches:active / matches:completed

Env:
    REDIS_URL, POLL_INTERVAL (30), ESPN_LEAGUE (fifa.world),
    ESPN_SEASON (2026), FT_ACTIVE_GRACE_S (6h)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import httpx
import redis.asyncio as aioredis

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from api.schemas.event_types import COMPLETED_STATUSES, LIVE_STATUSES
from api.schemas.schema import MatchState
from feeds import espn, snapshot
from monitoring.metrics import WORKER_ERRORS, tick_done

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
FT_ACTIVE_GRACE_S = int(os.getenv("FT_ACTIVE_GRACE_S", str(6 * 3600)))
FT_AT_TTL = 40 * 24 * 3600
MATCH_SPAN_S = 3 * 3600  # kickoff to final whistle, incl. ET and pens
DETAIL_TTL = 12 * 3600
NOT_PLAYED = {"NS", "PST", "CANC"}


# ── Redis persistence ──────────────────────────────────────────────────────


async def _worker_visible(
    r: aioredis.Redis, fid: int, status: str, kickoff: Optional[datetime] = None
) -> bool:
    """True while a fixture belongs in matches:active.

    NS and live fixtures stay active; FT ones for FT_ACTIVE_GRACE_S,
    measured from kickoff + MATCH_SPAN_S when the kickoff is known.
    """
    if status not in COMPLETED_STATUSES:
        return True
    if kickoff is not None:
        return time.time() - kickoff.timestamp() < MATCH_SPAN_S + FT_ACTIVE_GRACE_S
    key = f"match:{fid}:ft_at"
    ts = await r.get(key)
    if ts is None:
        await r.set(key, str(time.time()), ex=FT_AT_TTL)
        return True
    try:
        return (time.time() - float(ts)) < FT_ACTIVE_GRACE_S
    except ValueError:
        return True


async def persist(
    r: aioredis.Redis,
    state: MatchState,
    *,
    changed: bool,
    worker_visible: bool,
) -> None:
    ttl = 43_200 if state.status_short == "NS" else 3_600
    pipe = r.pipeline(transaction=True)
    pipe.setex(f"match:{state.fixture_id}:state", ttl, state.model_dump_json())
    if worker_visible:
        pipe.sadd("matches:active", str(state.fixture_id))
        pipe.srem("matches:completed", str(state.fixture_id))
    else:
        pipe.sadd("matches:completed", str(state.fixture_id))
        pipe.srem("matches:active", str(state.fixture_id))
    await pipe.execute()

    if changed:
        await r.publish("match_update", json.dumps({"fixture_id": state.fixture_id}))
        log.info(
            f"  [{state.fixture_id}] {state.status_short} {str(state.elapsed or '-'):>3}' "
            f"{state.home_name} {state.home_score}-{state.away_score} {state.away_name}"
        )


async def expire_stale(r: aioredis.Redis, seen_ids: set[int]) -> None:
    for set_name in ("matches:active", "matches:completed"):
        for fid_str in await r.smembers(set_name):
            try:
                known = int(fid_str) in seen_ids
            except ValueError:
                known = False
            if not known:
                await r.srem(set_name, fid_str)


# ── State assembly ─────────────────────────────────────────────────────────


def build_state(f: dict, detail: Optional[dict]) -> MatchState:
    status = f["status"]
    played = status not in NOT_PLAYED
    return MatchState(
        fixture_id=f["fixture_id"],
        league_id=1,
        season=2026,
        round=f["round"],
        venue=f.get("venue", ""),
        status_short=status,
        status_long=espn.STATUS_LONG.get(status, status),
        elapsed=f.get("elapsed") if played else None,
        elapsed_extra=f.get("elapsed_extra") if played else None,
        kickoff_time=f.get("kickoff"),
        home_id=1,
        home_name=f["home_name"],
        home_logo=f.get("home_logo", ""),
        home_score=f.get("home_score") or 0,
        home_pens=f.get("home_pens"),
        away_id=2,
        away_name=f["away_name"],
        away_logo=f.get("away_logo", ""),
        away_score=f.get("away_score") or 0,
        away_pens=f.get("away_pens"),
        **(
            {
                "home_stats": detail["home_stats"],
                "away_stats": detail["away_stats"],
                "events": detail["events"],
                "stats_source": "espn",
            }
            if detail and played
            else {"stats_source": "unavailable"}
        ),
    )


class Producer:
    def __init__(self, r: aioredis.Redis) -> None:
        self.r = r
        self.last_payload: dict[int, str] = {}
        self.detail_cache: dict[int, dict] = {}  # completed, non-snapshot
        self.last_live_detail: dict[int, dict] = {}
        self.last_aux: dict[tuple[int, str], tuple[str, float]] = {}

    async def fixtures(self, client: httpx.AsyncClient) -> list[dict]:
        try:
            events = await espn.fetch_scoreboard(client)
            fx = [f for e in events if (f := espn.parse_event(e))]
            if fx:
                return fx
            log.warning("ESPN scoreboard returned no fixtures — using snapshot")
        except Exception as e:
            log.warning(f"ESPN scoreboard failed ({e}) — using snapshot")
        return snapshot.fixtures()

    async def detail(self, client: httpx.AsyncClient, f: dict) -> Optional[dict]:
        fid, status = f["fixture_id"], f["status"]
        if status in NOT_PLAYED:
            return None
        if status in COMPLETED_STATUSES:
            d = await asyncio.to_thread(snapshot.match_detail, fid) or self.detail_cache.get(fid)
            if d is not None:
                return d
        try:
            raw = await espn.fetch_summary(client, fid)
            d = await asyncio.to_thread(espn.parse_summary, raw, f)
        except Exception as e:
            log.warning(f"  [{fid}] ESPN summary failed: {e}")
            # Keep the last good live detail rather than blanking stats.
            return self.last_live_detail.get(fid)
        if status in COMPLETED_STATUSES:
            self.detail_cache[fid] = d
        elif status in LIVE_STATUSES:
            self.last_live_detail[fid] = d
        return d

    async def _set_aux(self, fid: int, kind: str, value) -> None:
        """Write only on change, or hourly to keep the TTL alive."""
        payload = json.dumps(value)
        prev = self.last_aux.get((fid, kind))
        if prev and prev[0] == payload and time.time() - prev[1] < 3600:
            return
        self.last_aux[(fid, kind)] = (payload, time.time())
        await self.r.setex(f"match:{fid}:{kind}", DETAIL_TTL, payload)

    async def tick(self, client: httpx.AsyncClient) -> set[int]:
        seen: set[int] = set()
        for f in await self.fixtures(client):
            fid = f["fixture_id"]
            seen.add(fid)
            try:
                d = await self.detail(client, f)
                state = build_state(f, d)
                await self._persist_if_changed(state)
                if d:
                    if d.get("lineups"):
                        await self._set_aux(fid, "lineups", {**d["lineups"], "source": "espn"})
                    await self._set_aux(fid, "shots", d.get("shots", []))
            except Exception as e:
                WORKER_ERRORS.labels("match_producer").inc()
                log.error(f"Fixture {fid} failed: {e}", exc_info=True)
        if seen:
            await expire_stale(self.r, seen)
        for gone in set(self.last_payload) - seen:
            self.last_payload.pop(gone, None)
        return seen

    async def _persist_if_changed(self, state: MatchState) -> None:
        payload = state.model_dump_json(exclude={"updated_at"})
        changed = self.last_payload.get(state.fixture_id) != payload
        self.last_payload[state.fixture_id] = payload
        visible = await _worker_visible(
            self.r, state.fixture_id, state.status_short, state.kickoff_time
        )
        await persist(self.r, state, changed=changed, worker_visible=visible)


async def _loop(r: aioredis.Redis) -> None:
    producer = Producer(r)
    # Default UA on purpose: ESPN's CDN rejects custom User-Agents.
    async with httpx.AsyncClient(follow_redirects=True) as client:
        log.info(f"Match producer running — ESPN {espn.LEAGUE} {espn.SEASON}, poll={POLL_INTERVAL}s")
        while True:
            await producer.tick(client)
            tick_done("match_producer")
            await asyncio.sleep(POLL_INTERVAL)


async def run(redis_client: aioredis.Redis) -> None:
    """Supervised entry point (api/supervisor.py restarts it on a crash)."""
    try:
        await _loop(redis_client)
    except asyncio.CancelledError:
        log.info("Match producer cancelled")
        raise
    except Exception:
        WORKER_ERRORS.labels("match_producer").inc()
        raise


async def main() -> None:
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    await _loop(r)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
