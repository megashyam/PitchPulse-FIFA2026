"""
Deterministic grounding check for match-intel narration.

`violations(text, facts, state, rag_docs, ref_minutes)` lists every claim in
`text` that neither the facts block the model was given nor `MatchState`
supports:

  teams       a real WC team that isn't playing and isn't in a precedent doc
  players     a WC 2026 squad player not named in the facts or match events
  scorelines  "a-b" that isn't in the facts or the match's score progression
  percentages within 1 point of a fact number, or of a difference of two
  decimals    fact numbers at the precision written ("1.2" matches 1.23)
  minutes     "76'" / "minute 76" not in the facts (45/90/120 always allowed)
  remaining   "N minutes remaining" inconsistent with the event/current minute

Arithmetic on given numbers (a "12-point swing", a "0.49 xG gap") is allowed;
anything else is a fabrication. Counting words ("10 men") isn't checked.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from itertools import combinations
from typing import Iterable, List, Optional

from api.match_timeline import around
from api.schemas.event_types import GOAL_TYPES
from api.schemas.schema import MatchState
from ml.wc_2026_config import DATA_DIR, TEAM_BY_NAME

_NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")
_SCORE = re.compile(r"(?<![\d.\-–])(\d{1,2})\s*[-–—]\s*(\d{1,2})(?!\.?\d|[%\-–])")
_PCT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")
_DEC = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d%]|\s*%)")
_MINUTE = re.compile(r"(?<![\d.+])(\d{1,3})\s*['′’](?!\w)|\bminute\s+(\d{1,3})\b", re.I)
_REMAIN = re.compile(r"(\d{1,3})\s+(?:more\s+)?minutes?\s+(?:remaining|left|to play|to go)", re.I)
_HALF_MARKS = {45, 90, 105, 120}


@lru_cache(maxsize=1)
def _roster() -> dict:
    """Surface form -> full name for every player in the snapshot.

    Surnames count when distinctive: >=5 chars, unique, not a team name.
    """
    names: set = set()
    for p in (DATA_DIR / "matches").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        for side in (d.get("lineups") or {}).values():
            for key in ("startingXI", "substitutes"):
                names.update(pl["name"] for pl in side.get(key) or [] if pl.get("name"))
        names.update(e["player_name"] for e in d.get("events", []) if e.get("player_name"))
    surnames: dict = {}
    for n in names:
        parts = n.split()
        if len(parts) > 1:
            surnames.setdefault(parts[-1], set()).add(n)
    forms = {n: n for n in names}
    for s, full in surnames.items():
        if len(full) == 1 and len(s) >= 5 and s not in TEAM_BY_NAME and s not in forms:
            forms[s] = next(iter(full))
    return forms


def _mentions(text: str, form: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(form)}(?!\w)", text) is not None


def _fact_numbers(facts: str) -> List[float]:
    return [float(x) for x in _NUM.findall(facts)]


def _supported(value: float, nums: List[float], tol: float) -> bool:
    if any(abs(value - n) <= tol for n in nums):
        return True
    return any(abs(value - abs(a - b)) <= tol for a, b in combinations(nums, 2))


def _match_scores(state: MatchState) -> set:
    scores = {(0, 0), (state.home_score, state.away_score)}
    for ev in state.events:
        if ev.type in GOAL_TYPES:
            try:
                before, after = around(state, ev)
            except ValueError:
                continue
            scores |= {before[:2], after[:2]}
    return scores | {(b, a) for a, b in scores}


def _allowed_teams(state: MatchState, rag_docs: Iterable[str]) -> set:
    allowed = {state.home_name, state.away_name}
    for doc in rag_docs:
        allowed.update(name for name in TEAM_BY_NAME if name in doc)
    return allowed


def violations(
    text: str,
    facts: str,
    state: MatchState,
    rag_docs: Iterable[str] = (),
    ref_minutes: Iterable[Optional[int]] = (),
) -> List[str]:
    rag_docs = list(rag_docs)
    out: List[str] = []

    allowed = _allowed_teams(state, rag_docs)
    out += [f"team {n} not in play or precedent" for n in TEAM_BY_NAME if n not in allowed and _mentions(text, n)]

    in_match = {e.player_name for e in state.events if e.player_name}
    for form, full in _roster().items():
        if full in in_match or _mentions(facts, form) or _mentions(facts, full.split()[-1]):
            continue
        if _mentions(text, form):
            out.append(f"player {full} not in facts")

    nums = _fact_numbers(facts)
    fact_scores = {(int(a), int(b)) for a, b in _SCORE.findall(facts)}
    ok_scores = _match_scores(state) | fact_scores | {(b, a) for a, b in fact_scores}
    for a, b in _SCORE.findall(text):
        if (int(a), int(b)) not in ok_scores:
            out.append(f"scoreline {a}-{b} not in match")

    pct_spans = []
    for m in _PCT.finditer(text):
        pct_spans.append(m.span(1))
        if not _supported(float(m.group(1)), nums, 1.0):
            out.append(f"{m.group(1)}% not in facts")

    for m in _DEC.finditer(text):
        if any(s <= m.start() < e for s, e in pct_spans):
            continue
        places = len(m.group(1).split(".")[1])
        if not _supported(float(m.group(1)), nums, 0.5 * 10**-places + 1e-9):
            out.append(f"{m.group(1)} not in facts")

    refs = [r for r in ref_minutes if r is not None]
    ok_minutes = {int(n) for n in nums if n == int(n)} | _HALF_MARKS | set(refs)
    for m in _MINUTE.finditer(text):
        minute = int(m.group(1) or m.group(2))
        if minute not in ok_minutes:
            out.append(f"minute {minute} not in facts")

    for m in _REMAIN.finditer(text):
        n = int(m.group(1))
        if not any(abs(n - (end - r)) <= 3 for r in refs for end in (90, 120) if end >= r):
            out.append(f"{n} minutes remaining inconsistent with minute {max(refs, default='?')}")

    return out
