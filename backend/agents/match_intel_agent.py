"""
Live match intelligence generation pipeline.

Transforms raw match telemetry into context-aware AI narratives by combining:

1. Event detection:
   Identifies high-impact moments such as goals, cards, and tactical shifts.

2. Statistical analysis:
   Scores narrative importance using momentum changes, xG divergence,
   match state, and time sensitivity.

3. Retrieval-Augmented Generation (RAG):
   Uses semantic search over historical football knowledge to provide relevant
   context before generating analyst-style commentary.

4. LLM generation:
   Produces concise live narratives with template fallbacks when inference
   services are unavailable.

Architecture:
- Match-level state is maintained to prevent duplicate narratives.
- Context hashing avoids repeated analysis of unchanged game states.
- Sentence-transformer embeddings are generated asynchronously using a
  dedicated executor to avoid blocking live workers.
- Weaviate hybrid search combines semantic and lexical retrieval.
- LLM output is validated with deterministic template fallbacks.

The pipeline supports:
- Live event reactions
- Tactical momentum updates
- xG-based statistical narratives
- Full-time match summaries
"""

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from agents.ollama_client import generate_with_source
from agents.weaviate_client import (
    get_weaviate_client,
    NARRATIVE_ARCS,
    TACTICAL_PROFILES,
)
from ml.embedding_model import get_embed_model as _get_embed_model
from ml.executors import EMBED_EXECUTOR
from ml.in_play import inplay_wdl
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import build_prior_table, elo_to_wdl
from ml.wc_2026_config import TEAM_BY_NAME
from api.schemas.schema import MatchState

log = logging.getLogger(__name__)


@dataclass
class MatchIntelState:
    """
    Per-match memory used to maintain narrative continuity.

    Tracks generated events, previous context, momentum history, and timing
    constraints to prevent duplicate or low-value narratives.
    """

    covered_events: set = field(default_factory=set)
    last_narrated_minute: int = 0
    last_context_hash: str = ""
    momentum_history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_narrative_time: float = 0.0
    MIN_INTERVAL_SECS: float = 25.0
    PERIODIC_INTERVAL_MINS: int = 5


_intel_states: Dict[int, MatchIntelState] = {}


def _uncovered_key_events(
    state: MatchState,
    intel_state: MatchIntelState,
) -> list:
    """Return goal/red-card events since last_narrated_minute that haven't been covered."""
    out = []
    for ev in state.events:
        if ev.elapsed <= intel_state.last_narrated_minute:
            continue
        sig = f"{ev.elapsed}:{ev.type}:{ev.team_id}"
        if sig in intel_state.covered_events:
            continue
        if ev.type in ("goal", "own_goal", "penalty_goal", "red", "yellow_red"):
            out.append(ev)
    return out


def _context_hash(state: MatchState, momentum: Optional[dict], extra: str = "") -> str:
    """
    Generate a compact fingerprint of the current match state.

    Used as a deduplication mechanism. If scoreline, momentum, and recent
    events have not changed, the pipeline skips unnecessary generation.
    """
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
    momentum: Optional[dict],
) -> float:
    """
    Calculate narrative priority score.

    Higher scores indicate moments worth generating an AI insight for.

    Signals:
    - Goals and red cards
    - Momentum swings
    - xG vs scoreline mismatch
    - High-pressure match phases

    Returns zero when the current context has already been analyzed.
    """
    minute = state.elapsed or 0
    score = 0.0

    for ev in _uncovered_key_events(state, intel_state):
        if ev.type in ("goal", "own_goal", "penalty_goal"):
            score += 0.40
        elif ev.type in ("red", "yellow_red"):
            score += 0.35

    if momentum and len(intel_state.momentum_history) >= 2:
        delta = abs(
            momentum["home"]["momentum_score"] - intel_state.momentum_history[0]
        )
        score += min(0.30, delta * 2.5)

    h_div = abs(state.home_stats.expected_goals - state.home_score)
    a_div = abs(state.away_stats.expected_goals - state.away_score)
    max_div = max(h_div, a_div)
    if max_div > 0.8:
        score += 0.20
    elif max_div > 0.4:
        score += 0.10

    if (40 <= minute <= 46) or (85 <= minute <= 95) or state.status_short == "ET":
        score += 0.10

    if _context_hash(state, momentum) == intel_state.last_context_hash:
        return 0.0

    has_key_event = bool(_uncovered_key_events(state, intel_state))
    if not has_key_event:
        if (
            time.time() - intel_state.last_narrative_time
            < intel_state.MIN_INTERVAL_SECS
        ):
            return 0.0

    return round(score, 3)


def _narration_type_and_query(
    state: MatchState,
    intel_state: MatchIntelState,
    momentum: Optional[dict],
) -> Tuple[str, str]:
    """
    Classify the narrative type and construct the RAG retrieval query.

    Priority:
    1. Event reaction (goals/cards)
    2. Statistical anomaly (xG divergence)
    3. Tactical momentum analysis

    Returns:
        Narrative category and semantic search query.
    """
    minute = state.elapsed or 0

    key_evs = _uncovered_key_events(state, intel_state)
    if key_evs:
        ev = key_evs[-1]
        query = (
            f"{ev.type} minute {ev.elapsed} "
            f"score {state.home_score}-{state.away_score} "
            f"WC {state.home_name} {state.away_name} "
            f"tournament bracket implications"
        )
        return "event_reaction", query

    h_div = abs(state.home_stats.expected_goals - state.home_score)
    a_div = abs(state.away_stats.expected_goals - state.away_score)
    if max(h_div, a_div) > 0.6:
        diverging = state.home_name if h_div >= a_div else state.away_name
        xg = (
            state.home_stats.expected_goals
            if h_div >= a_div
            else state.away_stats.expected_goals
        )
        actual = state.home_score if h_div >= a_div else state.away_score
        query = (
            f"{diverging} xG {xg:.1f} goals {actual} "
            f"underperformance WC minute {minute} statistical pressure overdue goal"
        )
        return "xg_divergence", query

    if momentum:
        dom = (
            state.home_name
            if momentum["home"]["momentum_score"] > 0.5
            else state.away_name
        )
        poss = momentum["home"]["ewma_possession"]
        press = momentum["home"]["ewma_pressure"]
    else:
        dom, poss, press = state.home_name, 50.0, 0.15

    query = (
        f"{dom} possession {poss:.0f}% shot pressure {press:.3f} "
        f"WC tactical dominance high press minute {minute}"
    )
    return "tactical", query


def _build_prompt(
    state: MatchState,
    momentum: Optional[dict],
    rag_docs: List[str],
    wp: Optional[dict] = None,
) -> str:
    """
    Build the LLM instruction prompt using live match context.

    Combines:
    - Scoreline
    - xG
    - possession
    - momentum metrics
    - retrieved historical context

    The prompt is designed to produce concise analyst-style commentary rather
    than generic match summaries.
    """
    minute = state.elapsed or 0
    score_line = (
        f"{state.home_name} {state.home_score}–{state.away_score} {state.away_name}"
    )

    rag_section = ""
    if rag_docs:
        rag_section = (
            "\n\nHistorical WC precedent — cite ONLY these exact matches if "
            "relevant; do not name any other match, player, minute, or "
            "scoreline not listed here:\n"
            + "\n".join(f"• {d[:200]}" for d in rag_docs[:3])
        )

    if momentum:
        h = momentum["home"]
        a = momentum["away"]
        mom_line = (
            f"Home momentum {h['momentum_score']:.0%}, "
            f"goal prob {h['goal_prob_5min']:.1%}. "
            f"Away momentum {a['momentum_score']:.0%}, "
            f"goal prob {a['goal_prob_5min']:.1%}."
        )
    else:
        mom_line = ""

    wp_line = ""
    if wp:
        wp_line = (
            f" Live win probability: {state.home_name} {wp['home']:.0%}, "
            f"draw {wp['draw']:.0%}, {state.away_name} {wp['away']:.0%}."
        )

    return (
        f"[INST] You are a football intelligence analyst providing live WC 2026 "
        f"analysis. The match is happening RIGHT NOW. "
        f"Minute {minute}. {score_line}. "
        f"xG: {state.home_stats.expected_goals:.2f} vs {state.away_stats.expected_goals:.2f}. "
        f"Possession: {state.home_stats.possession:.0f}% vs {state.away_stats.possession:.0f}%. "
        f"{mom_line}{wp_line}"
        f"{rag_section}\n\n"
        f"Write 2-3 sharp sentences. "
        f"1. The Reality: state what's happening on the pitch right now. "
        f"2. The Signal: back it up with a specific number from above (xG, "
        f"momentum, or win probability — prefer win probability when given). "
        f"3. The Stakes: what to watch for in the next 5 minutes. "
        f"Rules: only state facts supported by the data above — never invent "
        f"a historical match, player, or scoreline not explicitly given. "
        f"Avoid clichés like 'seismic', 'detonated', 'haunting'. Be precise, "
        f"not melodramatic. Use active verbs; no robotic summary. [/INST]"
    )


def _build_template(
    state: MatchState,
    momentum: Optional[dict],
    narration_type: str,
) -> str:
    minute = state.elapsed or 0
    score_line = f"{state.home_score}–{state.away_score}"

    if narration_type == "event_reaction":

        for ev in reversed(state.events):
            if ev.type in ("goal", "own_goal", "penalty_goal"):
                return (
                    f"{ev.team_name} score at {ev.elapsed}' to make it {score_line}. "
                    f"xG at time of goal: {state.home_stats.expected_goals:.2f} "
                    f"({state.home_name}) vs {state.away_stats.expected_goals:.2f} "
                    f"({state.away_name}). "
                    f"Possession at {minute}': {state.home_stats.possession:.0f}% "
                    f"{state.home_name}, {state.away_stats.possession:.0f}% {state.away_name}."
                )
            if ev.type in ("red", "yellow_red"):
                return (
                    f"{ev.team_name} reduced to 10 men at {ev.elapsed}'. "
                    f"Score {score_line} at {minute}'. "
                    f"Numerical advantage could be decisive with "
                    f"{90 - minute} minutes remaining."
                )

    if narration_type == "xg_divergence":
        h_div = abs(state.home_stats.expected_goals - state.home_score)
        a_div = abs(state.away_stats.expected_goals - state.away_score)
        team = state.home_name if h_div >= a_div else state.away_name
        xg = (
            state.home_stats.expected_goals
            if h_div >= a_div
            else state.away_stats.expected_goals
        )
        actual = state.home_score if h_div >= a_div else state.away_score
        return (
            f"{team} carrying {xg:.2f} xG against {actual} actual goals at {minute}'. "
            f"The {abs(xg - actual):.2f} xG gap reflects sustained attacking pressure "
            f"the scoreline has not yet captured."
        )

    if not momentum:
        return (
            f"Match at {minute}' — {state.home_name} {score_line} {state.away_name}. "
            f"xG: {state.home_stats.expected_goals:.2f} vs "
            f"{state.away_stats.expected_goals:.2f}."
        )

    h_mom = momentum["home"]["momentum_score"]
    dom = state.home_name if h_mom > 0.5 else state.away_name
    sub = state.away_name if h_mom > 0.5 else state.home_name
    dom_score = state.home_score if h_mom > 0.5 else state.away_score
    sub_score = state.away_score if h_mom > 0.5 else state.home_score
    m = momentum["home" if h_mom > 0.5 else "away"]

    return (
        f"{dom} holding {m['momentum_score']:.0%} momentum at {minute}' "
        f"({dom} {dom_score}–{sub_score} {sub}). "
        f"Goal probability {m['goal_prob_5min']:.1%} in next 5 minutes. "
        f"Shot pressure EWMA {m['ewma_pressure']:.3f}, "
        f"pass accuracy {m['ewma_pass_acc']:.0f}%."
    )


async def update(
    state: MatchState,
    momentum: Optional[dict],
    loop: asyncio.AbstractEventLoop,
) -> Optional[dict]:
    """
    Generate the next live match intelligence update.

    Pipeline:
    1. Update match memory and momentum history.
    2. Determine whether the current state warrants narration.
    3. Generate embedding for semantic retrieval.
    4. Retrieve relevant historical football context.
    5. Generate LLM narrative or fallback template.
    6. Persist state to prevent duplicate insights.

    Returns:
        Structured intelligence object or None if no update is required.
    """
    fid = state.fixture_id
    if fid not in _intel_states:
        _intel_states[fid] = MatchIntelState()
    intel_state = _intel_states[fid]

    if momentum:
        intel_state.momentum_history.append(momentum["home"]["momentum_score"])

    is_live = state.status_short in ("1H", "2H", "ET", "P")
    elapsed = state.elapsed or 0

    score = _score(state, intel_state, momentum)
    has_key_event = bool(_uncovered_key_events(state, intel_state))

    force_baseline = is_live and intel_state.last_narrated_minute == 0 and elapsed >= 5

    minutes_since = elapsed - intel_state.last_narrated_minute
    force_periodic = (
        is_live
        and intel_state.last_narrated_minute > 0
        and minutes_since >= intel_state.PERIODIC_INTERVAL_MINS
    )

    if (
        not has_key_event
        and score <= 0.20
        and not force_baseline
        and not force_periodic
    ):
        return None

    narration_type, query_text = _narration_type_and_query(state, intel_state, momentum)

    model = _get_embed_model()
    query_vector: List[float] = await loop.run_in_executor(
        EMBED_EXECUTOR,
        lambda: model.encode(query_text, normalize_embeddings=True).tolist(),
    )

    wv = get_weaviate_client()
    rag_collection = (
        TACTICAL_PROFILES if narration_type == "tactical" else NARRATIVE_ARCS
    )
    event_filter = None
    if narration_type == "event_reaction":
        key_evs = _uncovered_key_events(state, intel_state)
        if key_evs:
            ev = key_evs[-1]
            if ev.type in ("goal", "own_goal", "penalty_goal"):
                event_filter = "goal"
            elif ev.type in ("red", "yellow_red"):
                event_filter = "red_card"

    rag_docs = wv.hybrid_search(
        query_vector=query_vector,
        query_text=query_text,
        top_k=5,
        event_filter=event_filter,
        collection=rag_collection,
    )

    use_llm = has_key_event or force_periodic or score > 0.50
    if use_llm:
        wp = await _win_prob_now(state)
        prompt = _build_prompt(state, momentum, rag_docs, wp)
        narrative, via = await generate_with_source(prompt)
        if narrative and _grounding_violation(narrative, state, rag_docs):
            log.warning(
                f"[{state.fixture_id}] colour narrative failed grounding check — discarding"
            )
            narrative = ""
        if not narrative:
            narrative = _build_template(state, momentum, narration_type)
            via = "template"
    else:
        narrative = _build_template(state, momentum, narration_type)
        via = "template"

    if not narrative:
        return None

    extra = str(elapsed) if force_periodic else ""
    intel_state.last_context_hash = _context_hash(state, momentum, extra=extra)
    intel_state.last_narrative_time = time.time()
    intel_state.last_narrated_minute = elapsed

    for ev in state.events:
        if ev.elapsed <= elapsed:
            intel_state.covered_events.add(f"{ev.elapsed}:{ev.type}:{ev.team_id}")

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


GOAL_TYPES = ("goal", "own_goal", "penalty_goal")
RED_TYPES = ("red", "yellow_red")
SIGNIFICANT_TYPES = GOAL_TYPES + RED_TYPES


def _score_at(state: MatchState, upto_minute: int) -> Tuple[int, int]:
    """Running (home, away) score at/through a given match minute."""
    hs = as_ = 0
    for e in sorted(state.events, key=lambda x: x.elapsed):
        if e.elapsed > upto_minute:
            break
        if e.type in ("goal", "penalty_goal"):
            if e.team_name == state.home_name:
                hs += 1
            else:
                as_ += 1
        elif e.type == "own_goal":
            if e.team_name == state.home_name:
                as_ += 1
            else:
                hs += 1
    return hs, as_


async def _pre_match_wdl(state: MatchState) -> Tuple[float, float, float]:

    try:
        odds_client = get_oddsapi_client()
        odds_table = await odds_client.get_all_odds()
        priors = build_prior_table(odds_table)
        key = (state.home_name, state.away_name)
        rev_key = (state.away_name, state.home_name)
        if key in priors:
            return priors[key]
        if rev_key in priors:
            p_l, p_d, p_w = priors[rev_key]
            return p_w, p_d, p_l
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] pre-match odds lookup failed: {exc}")
    h = TEAM_BY_NAME.get(state.home_name)
    a = TEAM_BY_NAME.get(state.away_name)
    return elo_to_wdl(h.elo, a.elo) if h and a else (0.40, 0.25, 0.35)


def _reds_at(state: MatchState, upto_minute: int) -> Tuple[int, int]:
    rh = sum(
        1
        for e in state.events
        if e.type in RED_TYPES
        and e.elapsed <= upto_minute
        and e.team_name == state.home_name
    )
    ra = sum(
        1
        for e in state.events
        if e.type in RED_TYPES
        and e.elapsed <= upto_minute
        and e.team_name == state.away_name
    )
    return rh, ra


async def _win_prob_now(state: MatchState) -> Optional[dict]:

    try:
        pre_wdl = await _pre_match_wdl(state)
        minute = state.elapsed or 0
        rh, ra = _reds_at(state, minute)
        wdl = inplay_wdl(pre_wdl, minute, state.home_score, state.away_score, rh, ra)
        return {"home": wdl[0], "draw": wdl[1], "away": wdl[2]}
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] win-prob calc failed: {exc}")
        return None


async def _win_prob_swing(state: MatchState, ev) -> Optional[dict]:

    try:
        pre_wdl = await _pre_match_wdl(state)
        hs_before, as_before = _score_at(state, ev.elapsed - 1)
        hs_after, as_after = _score_at(state, ev.elapsed)
        rh_before, ra_before = _reds_at(state, ev.elapsed - 1)
        rh_after, ra_after = _reds_at(state, ev.elapsed)

        wdl_before = inplay_wdl(
            pre_wdl, ev.elapsed, hs_before, as_before, rh_before, ra_before
        )
        wdl_after = inplay_wdl(
            pre_wdl, ev.elapsed, hs_after, as_after, rh_after, ra_after
        )

        if ev.type in RED_TYPES:
            beneficiary_is_home = ev.team_name != state.home_name
        else:

            beneficiary_is_home = hs_after > hs_before

        p_before = wdl_before[0] if beneficiary_is_home else wdl_before[2]
        p_after = wdl_after[0] if beneficiary_is_home else wdl_after[2]
        team = state.home_name if beneficiary_is_home else state.away_name
        return {"team": team, "p_before": p_before, "p_after": p_after}
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] win-prob swing failed: {exc}")
        return None


def _allowed_teams(state: MatchState, rag_docs: List[str]) -> set:

    allowed = {state.home_name, state.away_name}
    for doc in rag_docs:
        for name in TEAM_BY_NAME:
            if name in doc:
                allowed.add(name)
    return allowed


def _grounding_violation(
    narrative: str, state: MatchState, rag_docs: List[str]
) -> bool:

    allowed = _allowed_teams(state, rag_docs)
    for name in TEAM_BY_NAME:
        if name in allowed:
            continue
        if name in narrative:
            return True
    return False


def _event_template(state: MatchState, ev, completed: bool) -> str:
    hs, as_ = _score_at(state, ev.elapsed)
    totals = (
        f"Match totals: xG {state.home_stats.expected_goals:.2f} "
        f"({state.home_name}) vs {state.away_stats.expected_goals:.2f} "
        f"({state.away_name}), possession {state.home_stats.possession:.0f}%/"
        f"{state.away_stats.possession:.0f}%."
    )
    if ev.type in GOAL_TYPES:
        return f"{ev.team_name} score at {ev.elapsed}' to make it {hs}\u2013{as_}. {totals}"
    remaining = max(0, 90 - ev.elapsed)
    tail = (
        "Down to ten for the rest of the match."
        if completed
        else f"{remaining} minutes to play a man down."
    )
    return (
        f"{ev.team_name} reduced to 10 men at {ev.elapsed}' "
        f"(score {hs}\u2013{as_}). {tail}"
    )


def _event_prompt(
    state: MatchState,
    ev,
    rag_docs: List[str],
    completed: bool,
    wp: Optional[dict] = None,
) -> str:
    hs, as_ = _score_at(state, ev.elapsed)
    kind = "red card" if ev.type in RED_TYPES else "goal"
    rag = ""
    if rag_docs:
        rag = (
            "\n\nHistorical WC precedent \u2014 cite ONLY these exact matches if "
            "relevant; do not name any other match, player, minute, or "
            "scoreline not listed here:\n"
            + "\n".join(f"\u2022 {d[:180]}" for d in rag_docs[:2])
        )
    wp_line = ""
    if wp:
        wp_line = (
            f" Win probability shift: {wp['team']} "
            f"{wp['p_before']:.0%} \u2192 {wp['p_after']:.0%}."
        )
    tense = (
        "This match has finished."
        if completed
        else f"The match is live at minute {state.elapsed or 0}."
    )
    return (
        f"[INST] You are a football intelligence analyst, not a commentator. "
        f"At minute {ev.elapsed}', a moment shifted the game: {ev.team_name} \u2014 {kind}. "
        f"The new score is {hs}\u2013{as_}. "
        f"Current match totals: xG {state.home_stats.expected_goals:.2f} vs {state.away_stats.expected_goals:.2f}, "
        f"possession {state.home_stats.possession:.0f}% vs {state.away_stats.possession:.0f}%."
        f"{wp_line}"
        f" {tense}{rag}\n\n"
        f"Write 2 sentences on the impact of this moment. Rules: "
        f"(1) State only facts supported by the data above \u2014 never invent a "
        f"historical match, player moment, minute, or scoreline that isn't "
        f"explicitly given. If no precedent is listed, don't reference one. "
        f"(2) If a win-probability shift is given, cite the actual numbers "
        f"instead of vague momentum language. "
        f"(3) Avoid clich\u00e9s like 'seismic', 'detonated', 'haunting', 'catastrophic'. "
        f"Be precise and analytical, not melodramatic. [/INST]"
    )


async def analyze_event(
    state: MatchState,
    ev,
    loop: asyncio.AbstractEventLoop,
    use_llm: bool = True,
) -> dict:
    """
    Generate intelligence for a specific high-impact match event.

    Unlike live rolling updates, this method anchors the narrative to the
    exact event timestamp, allowing complete historical reconstruction of a
    match timeline.

    Used for:
    - Goals
    - Red cards
    - Completed match event feeds
    """
    completed = state.status_short in ("FT", "AET", "PEN")
    query = (
        f"{ev.type} minute {ev.elapsed} score {state.home_score}-{state.away_score} "
        f"WC {state.home_name} {state.away_name} tournament bracket implications"
    )

    rag_docs: List[str] = []
    try:
        model = _get_embed_model()
        qv = await loop.run_in_executor(
            EMBED_EXECUTOR,
            lambda: model.encode(query, normalize_embeddings=True).tolist(),
        )
        wv = get_weaviate_client()
        ev_filter = "goal" if ev.type in GOAL_TYPES else "red_card"
        rag_docs = wv.hybrid_search(
            query_vector=qv,
            query_text=query,
            top_k=5,
            event_filter=ev_filter,
            collection=NARRATIVE_ARCS,
        )
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] event RAG failed: {exc}")

    wp = await _win_prob_swing(state, ev)

    narrative = ""
    via = "template"
    if use_llm:
        try:
            narrative, source = await generate_with_source(
                _event_prompt(state, ev, rag_docs, completed, wp)
            )
        except Exception as exc:
            log.debug(f"[{state.fixture_id}] event LLM failed: {exc}")
            narrative = ""
        if narrative and _grounding_violation(narrative, state, rag_docs):
            log.warning(
                f"[{state.fixture_id}] event narrative failed grounding check — discarding"
            )
            narrative = ""
        if narrative:
            via = source
    if not narrative:
        narrative = _event_template(state, ev, completed)
        via = "template"

    return {
        "fixture_id": state.fixture_id,
        "minute": ev.elapsed,
        "narration_type": "event_reaction",
        "narrative": narrative,
        "score": 0.4,
        "rag_docs_used": len(rag_docs),
        "rag_collection": NARRATIVE_ARCS,
        "via": via,
        "event_sig": f"{ev.elapsed}:{ev.type}:{ev.team_id}",
        "win_prob_shift": wp,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _ft_summary_template(state: MatchState) -> str:
    hs, as_ = state.home_score, state.away_score
    if hs > as_:
        verdict = f"{state.home_name} saw it out {hs}\u2013{as_}"
    elif as_ > hs:
        verdict = f"{state.away_name} took it {as_}\u2013{hs} on the road"
    else:
        verdict = f"honours even at {hs}\u2013{as_}"
    return (
        f"Full time: {verdict}. {state.home_name} finished with "
        f"{state.home_stats.possession:.0f}% possession and "
        f"{state.home_stats.expected_goals:.2f} xG against "
        f"{state.away_stats.expected_goals:.2f} for {state.away_name} \u2014 "
        f"a {'tight, low-chance affair' if (state.home_stats.expected_goals + state.away_stats.expected_goals) < 2 else 'lively, chance-filled contest'} "
        f"by the underlying numbers."
    )


def _ft_summary_prompt(state: MatchState, rag_docs: List[str]) -> str:
    rag = ""
    if rag_docs:
        rag = (
            "\n\nHistorical WC precedent \u2014 cite ONLY these exact matches if "
            "relevant; do not name any other match, player, minute, or "
            "scoreline not listed here:\n"
            + "\n".join(f"\u2022 {d[:180]}" for d in rag_docs[:2])
        )
    return (
        f"[INST] You are a sharp FIFA World Cup analyst writing the post-match "
        f"wrap for {state.home_name} vs {state.away_name}, which finished "
        f"{state.home_score}\u2013{state.away_score}. "
        f"Match totals: xG {state.home_stats.expected_goals:.2f} vs "
        f"{state.away_stats.expected_goals:.2f}, possession "
        f"{state.home_stats.possession:.0f}% vs {state.away_stats.possession:.0f}%, "
        f"pass accuracy {state.home_stats.pass_accuracy:.0f}% vs "
        f"{state.away_stats.pass_accuracy:.0f}%.{rag}\n\n"
        f"Write 3 sentences summarising how this match played out and what the "
        f"underlying numbers say about it (did the result match the xG? who "
        f"controlled it?). Be specific with the numbers above. State only "
        f"facts supported by the data given \u2014 never invent a historical "
        f"match, player moment, or scoreline that isn't explicitly listed. "
        f"No cliches, no 'in conclusion'. [/INST]"
    )


async def analyze_full_time_summary(
    state: MatchState,
    loop: asyncio.AbstractEventLoop,
    use_llm: bool = True,
) -> dict:
    """
    Generate a final match analysis after completion.

    Handles matches without major events (such as 0-0 draws) by producing a
    statistics-driven summary using xG, possession, and historical context.
    """
    query = (
        f"full time {state.home_name} {state.away_name} "
        f"{state.home_score}-{state.away_score} World Cup match summary "
        f"xG possession tournament"
    )

    rag_docs: List[str] = []
    try:
        model = _get_embed_model()
        qv = await loop.run_in_executor(
            EMBED_EXECUTOR,
            lambda: model.encode(query, normalize_embeddings=True).tolist(),
        )
        wv = get_weaviate_client()
        rag_docs = wv.hybrid_search(
            query_vector=qv,
            query_text=query,
            top_k=5,
            collection=NARRATIVE_ARCS,
        )
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] FT summary RAG failed: {exc}")

    narrative = ""
    via = "template"
    if use_llm:
        try:
            narrative, source = await generate_with_source(
                _ft_summary_prompt(state, rag_docs)
            )
        except Exception as exc:
            log.debug(f"[{state.fixture_id}] FT summary LLM failed: {exc}")
            narrative = ""
        if narrative and _grounding_violation(narrative, state, rag_docs):
            log.warning(
                f"[{state.fixture_id}] FT summary failed grounding check — discarding"
            )
            narrative = ""
        if narrative:
            via = source
    if not narrative:
        narrative = _ft_summary_template(state)
        via = "template"

    return {
        "fixture_id": state.fixture_id,
        "minute": max(90, state.elapsed or 90),
        "narration_type": "tactical",
        "narrative": narrative,
        "score": 0.5,
        "rag_docs_used": len(rag_docs),
        "rag_collection": NARRATIVE_ARCS,
        "via": via,
        "event_sig": f"ft_summary:{state.home_score}:{state.away_score}",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def clear_state(fixture_id: int) -> None:
    """
    Remove cached intelligence state for a match.

    Used when replaying fixtures or restarting workers to ensure narratives are
    regenerated from a clean state.
    """
    if fixture_id in _intel_states:
        del _intel_states[fixture_id]
        log.info(f"[{fixture_id}] Intel state cleared")
