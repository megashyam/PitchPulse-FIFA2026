"""
Match intelligence agent.

Per 30s cycle:
    1. Score narratability.
    2. Pick the narration type and build the RAG query.
    3. Run agents.intel_graph: retrieve top-5 docs → generate (Ollama) →
       grounding check → one retry → template fallback.
    4. Update per-match state to prevent repetition.

Colour types (priority order); goals and red cards are narrated per event
by analyze_event():
    xg_divergence   a team's xG well above its goals
    tactical        every 5 match minutes, or on a momentum shift

Timing is match-minute based, so cadence is independent of replay speed.
"""

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from agents.grounding import _allowed_teams, team_violations
from agents.intel_graph import NarrationSpec, narrate
from agents.intel_prompts import (
    _build_prompt,
    _build_template,
    _event_prompt,
    _event_template,
    _ft_summary_prompt,
    _ft_summary_template,
    _xg_gap,
)
from agents.weaviate_client import NARRATIVE_ARCS, TACTICAL_PROFILES
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import prior_for
from api.match_timeline import around, at_minute
from api.tournament_state import side_probs
from schemas.event_types import COMPLETED_STATUSES, GOAL_TYPES, IN_PLAY, RED_TYPES
from schemas.feeds import IntelEntry, Momentum, WinProb, WinProbShift
from schemas.schema import MatchState

from config import (
    LLM_MIN_SCORE,
    MIN_INTERVAL_S,
    NARRATE_MIN_SCORE,
    PERIODIC_MINS,
    XG_GAP_HIGH,
    XG_GAP_LOW,
    XG_GAP_NARRATE,
)

log = logging.getLogger(__name__)


@dataclass
class MatchIntelState:
    # match minute of the last narrative (not wall-clock)
    last_narrated_minute: int = 0
    # context hash behind the last narrative
    last_context_hash: str = ""
    # recent momentum scores, for the delta term
    momentum_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # wall-clock time of the last narrative
    last_narrative_time: float = 0.0


_intel_states: Dict[int, MatchIntelState] = {}





def _context_hash(state: MatchState, momentum: Optional[Momentum], extra: str = "") -> str:
    """Hash of key match facts; `extra` makes forced periodic ticks unique."""
    minute = state.elapsed or 0
    scoreline = f"{state.home_score}-{state.away_score}"
    mom = ""
    if momentum:
        m = round(momentum["home"]["momentum_score"] * 20) / 20
        mom = f"{m:.2f}"
    last_ev = ""
    if state.events:
        ev = state.events[-1]
        last_ev = f"{ev.elapsed}:{ev.type}"
    key = f"{minute // 5}:{scoreline}:{mom}:{last_ev}:{extra}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _score(
    state: MatchState,
    intel_state: MatchIntelState,
    momentum: Optional[Momentum],
) -> float:
    # unchanged context hash scores 0
    if _context_hash(state, momentum) == intel_state.last_context_hash:
        return 0.0

    # MIN_INTERVAL_S between narratives
    if time.time() - intel_state.last_narrative_time < MIN_INTERVAL_S:
        return 0.0

    minute = state.elapsed or 0
    score = 0.0

    # momentum change vs oldest history entry, capped at 0.30
    if momentum and len(intel_state.momentum_history) >= 2:
        delta = abs(
            momentum["home"]["momentum_score"] - intel_state.momentum_history[0]
        )
        score += min(0.30, delta * 2.5)

    # xG minus score: +0.20 above 0.8, +0.10 above 0.4
    gap = _xg_gap(state)[3]
    if gap > XG_GAP_HIGH:
        score += 0.20
    elif gap > XG_GAP_LOW:
        score += 0.10

    # +0.10 at 40-46', 85-95' and extra time
    if (40 <= minute <= 46) or (85 <= minute <= 95) or state.status_short == "ET":
        score += 0.10

    return round(score, 3)


def event_query(state: MatchState, ev) -> Tuple[str, Optional[str]]:
    """RAG query and NarrativeArcs event_type filter for one goal / red card.

    Uses the score just after the event, not the current score.
    """
    _, after = around(state, ev)
    query = (
        f"{ev.type} minute {ev.elapsed} "
        f"score {after.home_score}-{after.away_score} "
        f"WC {state.home_name} {state.away_name} "
        f"tournament bracket implications"
    )
    ev_filter = "goal" if ev.type in GOAL_TYPES else "red_card" if ev.type in RED_TYPES else None
    return query, ev_filter


def _narration_type_and_query(
    state: MatchState,
    momentum: Optional[Momentum],
) -> Tuple[str, str]:
    minute = state.elapsed or 0

    # xG divergence above 0.6
    diverging, xg, actual, gap = _xg_gap(state)
    if gap > XG_GAP_NARRATE:
        query = (
            f"{diverging} xG {xg:.1f} goals {actual} "
            f"underperformance WC minute {minute} statistical pressure overdue goal"
        )
        return "xg_divergence", query

    # tactical
    if momentum:
        dom = (
            state.home_name
            if momentum["home"]["momentum_score"] > 0.5
            else state.away_name
        )
    else:
        dom = state.home_name
    poss = state.home_stats.possession or 50.0

    query = (
        f"{dom} possession {poss:.0f}% "
        f"WC tactical dominance high press minute {minute}"
    )
    return "tactical", query


async def colour_update(
    state: MatchState,
    momentum: Optional[Momentum],
) -> Optional[IntelEntry]:
    """xG-divergence / tactical colour for the current clock, or None."""
    fid = state.fixture_id
    intel_state = _intel_states.setdefault(fid, MatchIntelState())

    if momentum:
        intel_state.momentum_history.append(momentum["home"]["momentum_score"])

    is_live = state.status_short in IN_PLAY
    elapsed = state.elapsed or 0

    score = _score(state, intel_state, momentum)

    # first narrative once the match is live and past minute 5
    force_baseline = is_live and intel_state.last_narrated_minute == 0 and elapsed >= 5

    # every PERIODIC_MINS match minutes since the last narrative; match
    # minutes, so replay speed does not matter
    minutes_since = elapsed - intel_state.last_narrated_minute
    force_periodic = (
        is_live
        and intel_state.last_narrated_minute > 0
        and minutes_since >= PERIODIC_MINS
    )

    if score <= NARRATE_MIN_SCORE and not force_baseline and not force_periodic:
        return None

    narration_type, query_text = _narration_type_and_query(state, momentum)

    # narrate() in agents.intel_graph: retrieve, generate, ground, retry, template fallback
    # LLM for periodic ticks and score > 0.50; template for other tactical fills
    use_llm = force_periodic or score > LLM_MIN_SCORE
    rag_collection = TACTICAL_PROFILES if narration_type == "tactical" else NARRATIVE_ARCS
    wp = await _win_prob_now(state) if use_llm else None

    out = await narrate(NarrationSpec(
        kind="colour",
        state=state,
        query=query_text,
        collection=rag_collection,
        build_prompt=lambda docs: _build_prompt(state, momentum, docs, wp),
        template=lambda: _build_template(state, momentum, narration_type),
        ref_minutes=(elapsed,),
        use_llm=use_llm,
    ))
    narrative, via, rag_docs = out["narrative"], out["via"], out["rag_docs"]

    if not narrative:
        return None

    # periodic ticks add elapsed to the hash seed so identical match state still changes
    # the hash (SSE diff check)
    extra = str(elapsed) if force_periodic else ""
    intel_state.last_context_hash = _context_hash(state, momentum, extra=extra)
    intel_state.last_narrative_time = time.time()
    intel_state.last_narrated_minute = elapsed

    log.info(
        f"[{fid}] Intel: {narration_type} | score={score:.2f} | "
        f"via={via} | rag={len(rag_docs)} ({rag_collection}) | "
        f"elapsed={elapsed}' | periodic={force_periodic}"
    )

    return {
        "fixture_id": fid,
        "minute": elapsed,
        "narration_type": narration_type,
        "narrative": narrative,
        "score": round(score, 3),
        "rag_docs_used": len(rag_docs),
        "rag_collection": rag_collection,
        "via": via,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _pre_match_wdl(
    state: MatchState,
) -> Tuple[Tuple[float, float, float], Optional[Tuple[float, float]]]:
    """(pre-match W/D/L, market totals or None) — the prior behind GET /live-prob."""
    odds_table = None
    try:
        odds_table = await get_oddsapi_client().get_all_odds()
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] odds lookup failed: {exc}")
    return prior_for(state.home_name, state.away_name, state.fixture_id, odds_table)[:2]






async def _win_prob_now(state: MatchState) -> Optional[WinProb]:
    """Current in-play win probability (same model as GET /live-prob)."""
    try:
        pre_wdl, totals = await _pre_match_wdl(state)
        minute = state.elapsed or 0
        now = at_minute(state, minute, state.elapsed_extra or 99)
        now = now._replace(home_score=state.home_score, away_score=state.away_score)
        wdl = side_probs(
            pre_wdl, state.fixture_id, minute, now, extra=state.elapsed_extra, totals=totals
        )
        return {"home": wdl[0], "draw": wdl[1], "away": wdl[2]}
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] win-prob calc failed: {exc}")
        return None


async def _win_prob_swing(state: MatchState, ev) -> Optional[WinProbShift]:
    """Win-probability swing, before vs after, for the side the event helped."""
    try:
        pre_wdl, totals = await _pre_match_wdl(state)
        before, after = around(state, ev)
        fid = state.fixture_id
        wdl_before = side_probs(pre_wdl, fid, ev.elapsed, before, extra=ev.extra, totals=totals)
        wdl_after = side_probs(pre_wdl, fid, ev.elapsed, after, extra=ev.extra, totals=totals)

        if ev.type in RED_TYPES:
            # red card: the carded team's opponent benefits
            beneficiary_is_home = ev.team_id != 1
        else:
            # around() credits the opponent on an own goal; the side whose score rose is
            # the beneficiary
            beneficiary_is_home = after.home_score > before.home_score

        p_before = wdl_before[0] if beneficiary_is_home else wdl_before[2]
        p_after = wdl_after[0] if beneficiary_is_home else wdl_after[2]
        team = state.home_name if beneficiary_is_home else state.away_name
        return {"team": team, "p_before": p_before, "p_after": p_after}
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] win-prob swing failed: {exc}")
        return None


def _grounding_violation(narrative: str, state: MatchState, rag_docs: List[str]) -> bool:
    """True if the narrative names a WC team not playing and not in the RAG docs.

    Team-only guard used by the RAGAS eval; production uses agents.grounding.
    """
    return bool(team_violations(narrative, _allowed_teams(state, rag_docs)))






def event_spec(state: MatchState, ev, wp: Optional[WinProbShift], use_llm: bool = True) -> NarrationSpec:
    """Graph inputs for one goal / red-card narration (shared with the RAGAS eval)."""
    completed = state.status_short in COMPLETED_STATUSES
    query, ev_filter = event_query(state, ev)
    return NarrationSpec(
        kind="event",
        state=state,
        query=query,
        collection=NARRATIVE_ARCS,
        event_filter=ev_filter,
        build_prompt=lambda docs: _event_prompt(state, ev, docs, completed, wp),
        template=lambda: _event_template(state, ev, completed),
        ref_minutes=(ev.elapsed, state.elapsed),
        use_llm=use_llm,
    )


async def analyze_event(
    state: MatchState,
    ev,
    use_llm: bool = True,
) -> IntelEntry:
    """One intel entry for a specific goal / red card, stamped at its minute."""
    wp = await _win_prob_swing(state, ev)
    out = await narrate(event_spec(state, ev, wp, use_llm))

    return {
        "fixture_id": state.fixture_id,
        "minute": ev.elapsed,
        "narration_type": "event_reaction",
        "narrative": out["narrative"],
        "score": 0.4,
        "rag_docs_used": len(out["rag_docs"]),
        "rag_collection": NARRATIVE_ARCS,
        "via": out["via"],
        "event_sig": f"{ev.elapsed}:{ev.type}:{ev.team_id}",
        "win_prob_shift": wp,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }






async def analyze_full_time_summary(
    state: MatchState,
    use_llm: bool = True,
) -> IntelEntry:
    """Full-time wrap-up for a completed match, generated at most once.

    Keyed at minute 90 with a stable event_sig; covers matches with no goals.
    """
    query = (
        f"full time {state.home_name} {state.away_name} "
        f"{state.home_score}-{state.away_score} World Cup match summary "
        f"xG possession tournament"
    )

    out = await narrate(NarrationSpec(
        kind="ft_summary",
        state=state,
        query=query,
        collection=NARRATIVE_ARCS,
        build_prompt=lambda docs: _ft_summary_prompt(state, docs),
        template=lambda: _ft_summary_template(state),
        ref_minutes=(state.elapsed or 90,),
        use_llm=use_llm,
    ))
    narrative, via = out["narrative"], out["via"]

    return {
        "fixture_id": state.fixture_id,
        "minute": max(90, state.elapsed or 90),
        "narration_type": "tactical",
        "narrative": narrative,
        "score": 0.5,
        "rag_docs_used": len(out["rag_docs"]),
        "rag_collection": NARRATIVE_ARCS,
        "via": via,
        "event_sig": f"ft_summary:{state.home_score}:{state.away_score}",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def clear_state(fixture_id: int) -> None:
    """Drop per-match state once a fixture leaves matches:active."""
    if fixture_id in _intel_states:
        del _intel_states[fixture_id]
        log.info(f"[{fixture_id}] Intel state cleared")
