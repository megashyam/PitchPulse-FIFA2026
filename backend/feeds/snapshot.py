"""
Read access to the committed WC 2026 snapshot (feeds/build_snapshot.py).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from schemas.schema import Fixture, MatchDetail
from ml.wc_2026_config import DATA_DIR, FIXTURES


@lru_cache(maxsize=256)
def match_detail(fixture_id: int) -> Optional[MatchDetail]:
    """Stats, events, lineups and shots for a snapshotted match."""
    p = DATA_DIR / "matches" / f"{fixture_id}.json"
    if not p.exists():
        return None
    return MatchDetail.model_validate_json(p.read_text(encoding="utf-8"))


def fixtures() -> list[Fixture]:
    return [Fixture.model_validate(f) for f in FIXTURES]
