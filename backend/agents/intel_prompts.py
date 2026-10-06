"""Prompt and template text for agents.match_intel_agent: colour, event and
full-time narration. Pure functions of MatchState; no I/O."""

from typing import List, Optional, Tuple

from agents.rag import precedent_block
from api.match_timeline import around
from api.tournament_state import is_knockout
from schemas.event_types import GOAL_TYPES, RED_TYPES
from schemas.feeds import Momentum, WinProb, WinProbShift
from schemas.schema import MatchState


def _xg_gap(state: MatchState) -> Tuple[str, float, int, float]:
    """(team, xG, goals, |xG - goals|) for the side whose xG is furthest from
    its score; home on a tie."""
    h_div = abs(state.home_stats.expected_goals - state.home_score)
    a_div = abs(state.away_stats.expected_goals - state.away_score)
    if h_div >= a_div:
        return state.home_name, state.home_stats.expected_goals, state.home_score, h_div
    return state.away_name, state.away_stats.expected_goals, state.away_score, a_div


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


def _build_prompt(
    state: MatchState,
    momentum: Optional[Momentum],
    rag_docs: List[str],
    wp: Optional[WinProb] = None,
) -> str:
    minute = state.elapsed or 0
    score_line = (
        f"{state.home_name} {state.home_score}–{state.away_score} {state.away_name}"
    )

    rag_section = precedent_block(rag_docs, 3, 200)

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
    if wp and is_knockout(state.fixture_id):
        wp_line = (
            f" Chance to go through: {state.home_name} {wp['home']:.0%}, "
            f"{state.away_name} {wp['away']:.0%}."
        )
    elif wp:
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
    momentum: Optional[Momentum],
    narration_type: str,
) -> str:
    minute = state.elapsed or 0
    score_line = f"{state.home_score}–{state.away_score}"

    if narration_type == "xg_divergence":
        team, xg, actual, gap = _xg_gap(state)
        return (
            f"{team} carrying {xg:.2f} xG against {actual} actual goals at {minute}'. "
            f"The {gap:.2f} xG gap reflects sustained attacking pressure "
            f"the scoreline has not yet captured."
        )

    # tactical
    if not momentum:
        return (
            f"Match at {minute}' — {state.home_name} {score_line} {state.away_name}. "
            f"xG: {state.home_stats.expected_goals:.2f} vs "
            f"{state.away_stats.expected_goals:.2f}."
        )

    if momentum["home"]["momentum_score"] > 0.5:
        dom, sub, dom_score, sub_score = state.home_name, state.away_name, state.home_score, state.away_score
        m = momentum["home"]
    else:
        dom, sub, dom_score, sub_score = state.away_name, state.home_name, state.away_score, state.home_score
        m = momentum["away"]

    return (
        f"{dom} holding {m['momentum_score']:.0%} momentum at {minute}' "
        f"({dom} {dom_score}–{sub_score} {sub}). "
        f"{m['shots_15min']} shots worth {m['xg_15min']:.2f} xG in the last 15 minutes; "
        f"model chance of a goal in the next 5 minutes {m['goal_prob_5min']:.0%}."
    )


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
    wp: Optional[WinProbShift] = None,
) -> str:
    hs, as_ = around(state, ev)[1][:2]
    rag = precedent_block(rag_docs, 2, 180)
    wp_line = ""
    if wp:
        wp_line = (
            f" {'Chance to go through' if is_knockout(state.fixture_id) else 'Win probability'}"
            f" shift: {wp['team']} "
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
    rag = precedent_block(rag_docs, 2, 180)
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
