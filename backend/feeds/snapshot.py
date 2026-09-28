"""
Read access to the committed WC 2026 snapshot (feeds/build_snapshot.py).
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from typing import Optional

from api.schemas.schema import MatchEvent, TeamStats
from ml.wc_2026_config import DATA_DIR, FIXTURES


@lru_cache(maxsize=256)
def match_detail(fixture_id: int) -> Optional[dict]:
    """Stats, events, lineups and shots for a snapshotted match.

    Same shape as feeds.espn.parse_summary.
    """
    p = DATA_DIR / "matches" / f"{fixture_id}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "home_stats": TeamStats(**d["home_stats"]),
        "away_stats": TeamStats(**d["away_stats"]),
        "events": [MatchEvent(**e) for e in d["events"]],
        "lineups": d.get("lineups"),
        "shots": d.get("shots", []),
    }


def fixtures() -> list[dict]:
    """Snapshot fixtures in feeds.espn.parse_event() shape."""
    out = []
    for f in FIXTURES:
        rec = dict(f)
        rec["kickoff"] = datetime.fromisoformat(f["date"].replace("Z", "+00:00")) if f.get("date") else None
        out.append(rec)
    return out
