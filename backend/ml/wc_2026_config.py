"""
WC 2026 tournament configuration.

Loaded from the committed snapshot (data/wc2026/tournament.json, built by
feeds/build_snapshot.py from ESPN, martj42 and FIFA Annex C).

    WC2026_TEAMS       48 teams with group and point-in-time Elo (as of the
                       opening match; see ml/elo_ratings.py)
    GROUPS             letter → teams in the group
    FIXTURES           all 104 fixtures (group + knockout, with FIFA match_no)
    GROUP_FIXTURES     the 72 group matches, real home/away and neutral flag
    THIRD_PLACE_TABLE  Annex C: sorted qualifying-third groups → {"1A": "E", ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "wc2026"


@dataclass
class TeamConfig:
    name: str
    group: str  # 'A' .. 'L'
    elo: float  # World Football Elo before the opening match
    fifa_rank: int = 0  # Elo rank within the field, for display / sorting
    espn_id: str = ""


def _load() -> dict:
    return json.loads((DATA_DIR / "tournament.json").read_text(encoding="utf-8"))


_SNAPSHOT = _load()

WC2026_TEAMS: List[TeamConfig] = [
    TeamConfig(t["name"], t["group"], float(t["elo"]), espn_id=t.get("espn_id", ""))
    for t in _SNAPSHOT["teams"]
]
for _rank, _t in enumerate(sorted(WC2026_TEAMS, key=lambda t: -t.elo), start=1):
    _t.fifa_rank = _rank

TEAM_BY_NAME: Dict[str, TeamConfig] = {t.name: t for t in WC2026_TEAMS}

GROUPS: Dict[str, List[TeamConfig]] = {}
for _t in WC2026_TEAMS:
    GROUPS.setdefault(_t.group, []).append(_t)
GROUPS = dict(sorted(GROUPS.items()))

FIXTURES: List[dict] = _SNAPSHOT["fixtures"]
FIXTURE_BY_ID: Dict[int, dict] = {f["fixture_id"]: f for f in FIXTURES}
GROUP_FIXTURES: List[dict] = [f for f in FIXTURES if f["stage"] == "group"]
KO_FIXTURE_BY_MATCH_NO: Dict[int, dict] = {
    f["match_no"]: f for f in FIXTURES if f.get("match_no")
}
FINAL_STANDINGS: Dict[str, List[dict]] = _SNAPSHOT["groups"]

THIRD_PLACE_TABLE: Dict[str, Dict[str, str]] = json.loads(
    (DATA_DIR / "third_place_table.json").read_text(encoding="utf-8")
)["table"]
