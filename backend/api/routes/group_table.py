"""
Group standings endpoint.

    GET /matches/{fixture_id}/group-table

One SUNION(matches:active, matches:completed) plus one MGET per request.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request

from api.schemas.event_types import COMPLETED_STATUSES
from api.schemas.schema import MatchState

router = APIRouter()


def _extract_group(round_str: str) -> str:
    """Extract group letter from 'Group Stage - Group A' → 'A'"""
    m = re.search(r"Group\s+([A-L])", round_str or "", re.IGNORECASE)
    return m.group(1).upper() if m else ""


@router.get("/{fixture_id}/group-table")
async def group_table(fixture_id: str, request: Request):
    r = request.app.state.redis

    raw = await r.get(f"match:{fixture_id}:state")
    if not raw:
        raise HTTPException(404, "Fixture not found")

    state = MatchState.model_validate_json(raw)
    group = _extract_group(state.round)
    if not group:
        raise HTTPException(
            404, f"Could not determine group from round: {state.round!r}"
        )

    all_ids = await r.sunion("matches:active", "matches:completed")
    keys = [f"match:{fid}:state" for fid in all_ids]
    raw_values = await r.mget(keys) if keys else []

    standings: dict[str, dict] = {}

    for match_raw in raw_values:
        if not match_raw:
            continue
        try:
            m = MatchState.model_validate_json(match_raw)
        except Exception:
            continue

        if _extract_group(m.round) != group:
            continue

        for name in (m.home_name, m.away_name):
            if name not in standings:
                standings[name] = {
                    "name": name, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0,
                }

        if m.status_short not in COMPLETED_STATUSES:
            continue

        hs, as_ = m.home_score, m.away_score
        standings[m.home_name]["p"] += 1
        standings[m.away_name]["p"] += 1
        standings[m.home_name]["gf"] += hs
        standings[m.home_name]["ga"] += as_
        standings[m.away_name]["gf"] += as_
        standings[m.away_name]["ga"] += hs

        if hs > as_:
            standings[m.home_name]["w"] += 1
            standings[m.away_name]["l"] += 1
        elif as_ > hs:
            standings[m.away_name]["w"] += 1
            standings[m.home_name]["l"] += 1
        else:
            standings[m.home_name]["d"] += 1
            standings[m.away_name]["d"] += 1

    if not standings:
        return {"group": group, "table": []}

    table = []
    for t in standings.values():
        t["pts"] = 3 * t["w"] + t["d"]
        t["gd"] = t["gf"] - t["ga"]
        table.append(t)

    table.sort(key=lambda t: (-t["pts"], -t["gd"], -t["gf"], t["name"]))

    for i, t in enumerate(table):
        t["pos"] = i + 1

    return {"group": group, "table": table}
