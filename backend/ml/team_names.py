"""
Team name aliases.

Canonical names are ESPN's displayName (used by the feed and the snapshot).
Other spellings (martj42 results, StatsBomb, Opta commentary, simulator
names) resolve through canonical().
"""

from __future__ import annotations

from typing import Dict, Set

_ALIASES: Dict[str, str] = {
    # martj42 / commentary / StatsBomb / legacy spellings → ESPN
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "Czech Republic": "Czechia",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Holland": "Netherlands",
}


def canonical(name: str) -> str:
    return _ALIASES.get(name, name)


def to_sim(name: str) -> str:
    """Any spelling → simulator name; unknown names pass through unchanged."""
    return canonical(name)


def is_sim_team(name: str) -> bool:
    return canonical(name) in _sim_names()


def _sim_names() -> Set[str]:
    # Lazy: the snapshot builder imports this module before a snapshot exists.
    from ml.wc_2026_config import WC2026_TEAMS

    return {t.name for t in WC2026_TEAMS}


def __getattr__(attr: str):
    if attr == "SIM_NAMES":
        return _sim_names()
    raise AttributeError(attr)
