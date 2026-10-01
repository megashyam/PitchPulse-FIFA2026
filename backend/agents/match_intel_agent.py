"""
Match intelligence agent.

Per 30s cycle:
    1. Score narratability.
    2. Pick the narration type and build the RAG query.
    3. Run agents.intel_graph: retrieve top-5 docs → generate (Ollama or
       Groq) → grounding check → one retry → template fallback.
    4. Update per-match state to prevent repetition.

Triggers (priority order):
    event_reaction  uncovered goal / red card since the last narrated minute
    xg_divergence   a team's xG well above its goals
    tactical        every 5 match minutes, or on a momentum shift

Timing is match-minute based, so cadence is independent of replay speed.
Goals and red cards bypass the 25s rate limit.
"""

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from agents.grounding import _allowed_teams
from agents.intel_graph import NarrationSpec, narrate
from agents.weaviate_client import NARRATIVE_ARCS, TACTICAL_PROFILES
from ml.in_play import inplay_wdl
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import match_wdl
from ml.wc_2026_config import FIXTURE_BY_ID, TEAM_BY_NAME
from api.match_timeline import around, at_minute
from api.schemas.event_types import GOAL_TYPES, RED_TYPES
from api.schemas.schema import MatchState

log = logging.getLogger(__name__)


# ── Per-fixture state ─────────────────────────────────────────────────────


@dataclass
class MatchIntelState:
    # Signatures of events already narrated: "elapsed:type:team_id"
    covered_events: set = field(default_factory=set)
    # Match minute of the last generated narrative (NOT real-world time)
    last_narrated_minute: int = 0
    # Hash of the context that produced the last narrative — dedup guard
    last_context_hash: str = ""
    # Rolling momentum history for delta calculation
    momentum_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # Wall-clock time of last narrative — 25s guard against rapid-fire
    last_narrative_time: float = 0.0
    MIN_INTERVAL_SECS: float = 25.0
    # How many match minutes between periodic tactical narratives
    PERIODIC_INTERVAL_MINS: int = 5


_intel_states: Dict[int, MatchIntelState] = {}


# ── Helpers ───────────────────────────────────────────────────────────────


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


# ── Score ─────────────────────────────────────────────────────────────────


def _score(
    state: MatchState,
    intel_state: MatchIntelState,
    momentum: Optional[dict],
) -> float:
    minute = state.elapsed or 0
    score = 0.0

    # Key events since last narrated minute
    for ev in _uncovered_key_events(state, intel_state):
        if ev.type in ("goal", "own_goal", "penalty_goal"):
            score += 0.40
        elif ev.type in ("red", "yellow_red"):
            score += 0.35

    # Momentum delta
    if momentum and len(intel_state.momentum_history) >= 2:
        delta = abs(
            momentum["home"]["momentum_score"] - intel_state.momentum_history[0]
        )
        score += min(0.30, delta * 2.5)

    # xG divergence
    h_div = abs(state.home_stats.expected_goals - state.home_score)
    a_div = abs(state.away_stats.expected_goals - state.away_score)
    max_div = max(h_div, a_div)
    if max_div > 0.8:
        score += 0.20
    elif max_div > 0.4:
        score += 0.10

    # Time sensitivity
    if (40 <= minute <= 46) or (85 <= minute <= 95) or state.status_short == "ET":
        score += 0.10

    # Context hash guard — nothing has changed, nothing to say
    if _context_hash(state, momentum) == intel_state.last_context_hash:
        return 0.0

    # Rate limit — goals/reds always bypass, other types respect 25s gap
    has_key_event = bool(_uncovered_key_events(state, intel_state))
    if not has_key_event:
        if (
            time.time() - intel_state.last_narrative_time
            < intel_state.MIN_INTERVAL_SECS
        ):
            return 0.0

    return round(score, 3)


# ── Narration type ────────────────────────────────────────────────────────


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
    intel_state: MatchIntelState,
    momentum: Optional[dict],
) -> Tuple[str, str]:
    minute = state.elapsed or 0

    # Priority 1: uncovered key event
    key_evs = _uncovered_key_events(state, intel_state)
    if key_evs:
        return "event_reaction", event_query(state, key_evs[-1])[0]

    # Priority 2: xG divergence
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

    # Priority 3: tactical
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


# ── Prompt / template ─────────────────────────────────────────────────────


def _build_prompt(
    state: MatchState,
    momentum: Optional[dict],
    rag_docs: List[str],
    wp: Optional[dict] = None,
) -> str:
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
            f"Last 15 minutes: {state.home_name} {h['shots_15min']} shots "
            f"(xG {h['xg_15min']:.2f}), {state.away_name} {a['shots_15min']} shots "
            f"(xG {a['xg_15min']:.2f}). Model chance of a goal in the next 5 "
            f"minutes: {h['goal_prob_5min']:.0%} vs {a['goal_prob_5min']:.0%}."
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
        f"xG (shot-model estimate): {state.home_stats.expected_goals:.2f} vs "
        f"{state.away_stats.expected_goals:.2f}. "
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
        # Find the most recent goal/card to describe
        for ev in reversed(state.events):
            if ev.type in ("goal", "own_goal", "penalty_goal"):
                return (
                    f"{_sentence(_describe(ev))} at {ev.elapsed}'; it is {score_line}. "
                    f"Match xG so far: {state.home_stats.expected_goals:.2f} "
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
                    f"{max(0, (120 if minute > 90 else 90) - minute)} minutes remaining."
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

    # tactical
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
        f"{m['shots_15min']} shots worth {m['xg_15min']:.2f} xG in the last 15 minutes; "
        f"model chance of a goal in the next 5 minutes {m['goal_prob_5min']:.0%}."
    )


# ── Main entry point ──────────────────────────────────────────────────────


async def update(
    state: MatchState,
    momentum: Optional[dict],
) -> Optional[dict]:
    fid = state.fixture_id
    if fid not in _intel_states:
        _intel_states[fid] = MatchIntelState()
    intel_state = _intel_states[fid]

    if momentum:
        intel_state.momentum_history.append(momentum["home"]["momentum_score"])

    is_live = state.status_short in ("1H", "2H", "ET", "P")
    elapsed = state.elapsed or 0

    # ── Decide whether to generate ────────────────────────────────────────
    score = _score(state, intel_state, momentum)
    has_key_event = bool(_uncovered_key_events(state, intel_state))

    # Baseline: first narrative of this cycle once match is underway
    force_baseline = is_live and intel_state.last_narrated_minute == 0 and elapsed >= 5

    # Periodic: every PERIODIC_INTERVAL_MINS match minutes since last narrative.
    # Uses MATCH minutes (not real seconds) so it fires correctly at any replay speed.
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

    # ── Narration type + query ────────────────────────────────────────────
    narration_type, query_text = _narration_type_and_query(state, intel_state, momentum)

    # ── Retrieve → generate → ground → retry/template (agents.intel_graph) ─
    # LLM for key events (always), periodic (always), and high scores;
    # template for low-score tactical fills.
    use_llm = has_key_event or force_periodic or score > 0.50
    rag_collection = TACTICAL_PROFILES if narration_type == "tactical" else NARRATIVE_ARCS
    event_filter = None
    if narration_type == "event_reaction":
        key_evs = _uncovered_key_events(state, intel_state)
        if key_evs:
            event_filter = event_query(state, key_evs[-1])[1]
    wp = await _win_prob_now(state) if use_llm else None

    out = await narrate(NarrationSpec(
        kind="colour",
        state=state,
        query=query_text,
        collection=rag_collection,
        event_filter=event_filter,
        build_prompt=lambda docs: _build_prompt(state, momentum, docs, wp),
        template=lambda: _build_template(state, momentum, narration_type),
        ref_minutes=(elapsed,),
        use_llm=use_llm,
    ))
    narrative, via, rag_docs = out["narrative"], out["via"], out["rag_docs"]

    if not narrative:
        return None

    # ── Update state ──────────────────────────────────────────────────────
    # Use elapsed as the extra seed for periodic ticks so hash is unique
    # even with identical match state (prevents SSE diff-check suppression).
    extra = str(elapsed) if force_periodic else ""
    intel_state.last_context_hash = _context_hash(state, momentum, extra=extra)
    intel_state.last_narrative_time = time.time()
    intel_state.last_narrated_minute = elapsed

    # Cover all events up to now — they won't re-trigger
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




async def _pre_match_wdl(state: MatchState) -> Tuple[float, float, float]:
    """Pre-match W/D/L; the shared prior behind GET /live-prob."""
    odds_table = None
    try:
        odds_table = await get_oddsapi_client().get_all_odds()
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] odds lookup failed: {exc}")
    fx = FIXTURE_BY_ID.get(state.fixture_id) or {}
    return match_wdl(
        state.home_name, state.away_name, host_side=fx.get("host_side"), odds_table=odds_table
    )


def _sentence(text: str) -> str:
    """Upper-case the first letter only; str.capitalize() lowercases names."""
    return text[:1].upper() + text[1:]


def _describe(ev) -> str:
    """"goal by Player (Team)"; own goals name the conceding side correctly."""
    who = f"{ev.player_name} ({ev.team_name})" if ev.player_name else ev.team_name
    if ev.type == "own_goal":
        return f"own goal by {who}"
    if ev.type in RED_TYPES:
        return f"red card for {who}"
    return f"{'penalty ' if ev.type == 'penalty_goal' else ''}goal by {who}"


async def _win_prob_now(state: MatchState) -> Optional[dict]:
    """Current in-play win probability (same model as GET /live-prob)."""
    try:
        pre_wdl = await _pre_match_wdl(state)
        minute = state.elapsed or 0
        now = at_minute(state, minute, state.elapsed_extra or 99)
        wdl = inplay_wdl(
            pre_wdl, minute, state.home_score, state.away_score, now.red_home, now.red_away,
            extra=state.elapsed_extra,
        )
        return {"home": wdl[0], "draw": wdl[1], "away": wdl[2]}
    except Exception as exc:
        log.debug(f"[{state.fixture_id}] win-prob calc failed: {exc}")
        return None


async def _win_prob_swing(state: MatchState, ev) -> Optional[dict]:
    """Win-probability swing, before vs after, for the side the event helped."""
    try:
        pre_wdl = await _pre_match_wdl(state)
        before, after = around(state, ev)
        wdl_before = inplay_wdl(pre_wdl, ev.elapsed, *before, extra=ev.extra)
        wdl_after = inplay_wdl(pre_wdl, ev.elapsed, *after, extra=ev.extra)

        if ev.type in RED_TYPES:
            # a red card hurts the carded player's team, benefiting the opponent
            beneficiary_is_home = ev.team_id != 1
        else:
            # around() credits the opponent on an own goal, so the side whose
            # score rose is the beneficiary in every case.
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
    allowed = _allowed_teams(state, rag_docs)
    for name in TEAM_BY_NAME:
        if name in allowed:
            continue
        if name in narrative:
            return True
    return False


def _event_template(state: MatchState, ev, completed: bool) -> str:
    hs, as_ = around(state, ev)[1][:2]
    totals = (
        f"Match totals: model xG {state.home_stats.expected_goals:.2f} "
        f"({state.home_name}) vs {state.away_stats.expected_goals:.2f} "
        f"({state.away_name}), possession {state.home_stats.possession:.0f}%/"
        f"{state.away_stats.possession:.0f}%."
    )
    if ev.type in GOAL_TYPES:
        return f"{_sentence(_describe(ev))} at {ev.elapsed}' makes it {hs}\u2013{as_}. {totals}"
    remaining = max(0, 90 - ev.elapsed)
    tail = (
        "Down to ten for the rest of the match."
        if completed
        else f"{remaining} minutes to play a man down."
    )
    who = f"{ev.player_name} sent off \u2014 " if ev.player_name else ""
    return (
        f"{who}{ev.team_name} reduced to 10 men at {ev.elapsed}' "
        f"(score {hs}\u2013{as_}). {tail}"
    )


def _event_prompt(
    state: MatchState,
    ev,
    rag_docs: List[str],
    completed: bool,
    wp: Optional[dict] = None,
) -> str:
    hs, as_ = around(state, ev)[1][:2]
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
        f"At minute {ev.elapsed}', a moment shifted the game: {_describe(ev)}. "
        f"The score after it: {state.home_name} {hs}\u2013{as_} {state.away_name}. "
        f"Current match totals: xG (shot-model estimate) "
        f"{state.home_stats.expected_goals:.2f} vs {state.away_stats.expected_goals:.2f}, "
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


def event_spec(state: MatchState, ev, wp: Optional[dict], use_llm: bool = True) -> NarrationSpec:
    """Graph inputs for one goal / red-card narration (shared with the RAGAS eval)."""
    completed = state.status_short in ("FT", "AET", "PEN")
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
) -> dict:
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
        f"Match totals: xG (shot-model estimate) {state.home_stats.expected_goals:.2f} vs "
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
    use_llm: bool = True,
) -> dict:
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
    """Wipe per-match state (intel_worker, on replay restart)."""
    if fixture_id in _intel_states:
        del _intel_states[fixture_id]
        log.info(f"[{fixture_id}] Intel state cleared")
