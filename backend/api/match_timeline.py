"""
Match state as of a given event, reconstructed from MatchState.events.

Shared by the counterfactual and intel agents so backfilled events are
modelled at the score they happened at, not the final score.

Events are ordered by (minute, stoppage, list position), so same-minute and
stoppage-time events resolve correctly.
"""

from __future__ import annotations

from typing import NamedTuple

from api.schemas.event_types import RED_TYPES
from api.schemas.schema import MatchEvent, MatchState


class Snapshot(NamedTuple):
    home_score: int
    away_score: int
    red_home: int
    red_away: int


def _ordered(state: MatchState) -> list[MatchEvent]:
    return [
        e
        for _, e in sorted(
            enumerate(state.events), key=lambda p: (p[1].elapsed, p[1].extra or 0, p[0])
        )
    ]


def _apply(s: list[int], e: MatchEvent) -> None:
    home = e.team_id == 1
    if e.type in ("goal", "penalty_goal"):
        s[0 if home else 1] += 1
    elif e.type == "own_goal":  # team is the conceding side
        s[1 if home else 0] += 1
    elif e.type in RED_TYPES and e.source == "espn":
        s[2 if home else 3] += 1


def around(state: MatchState, event: MatchEvent) -> tuple[Snapshot, Snapshot]:
    """(state just before `event`, state just after it).

    Matches by identity first: two same-minute goals by one side compare equal.
    """
    ordered = _ordered(state)
    for match in (lambda e: e is event, lambda e: e == event):
        s = [0, 0, 0, 0]
        for e in ordered:
            if match(e):
                before = Snapshot(*s)
                _apply(s, e)
                return before, Snapshot(*s)
            _apply(s, e)
    raise ValueError("event not in state")


def at_minute(state: MatchState, minute: int, extra: int = 99) -> Snapshot:
    """State after every event up to and including minute+extra."""
    s = [0, 0, 0, 0]
    for e in _ordered(state):
        if (e.elapsed, e.extra or 0) > (minute, extra):
            break
        _apply(s, e)
    return Snapshot(*s)
