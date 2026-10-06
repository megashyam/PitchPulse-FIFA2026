"""Prompt and template text for agents.counterfactual_agent. Pure functions;
no I/O."""

from typing import List, Optional, Sequence, Tuple

from api.tournament_state import is_knockout
from schemas.feeds import Change
from schemas.schema import MatchEvent, MatchState

from config import CF_SIMS


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _no_impact_text(trigger: MatchEvent, conditioned: bool) -> str:
    who = trigger.player_name or trigger.team_name
    what = trigger.type.replace("_", " ")
    reason = (
        "left the score and the numbers on the pitch unchanged, so the bracket "
        "outlook did not move."
        if conditioned
        else "is outside the simulated tournament."
    )
    return f"{who}'s {what} at {trigger.elapsed}' {reason}"


def _prob_label(state: MatchState) -> str:
    return "chance to go through" if is_knockout(state.fixture_id) else "in-match win probability"


def _build_prompt(
    state: MatchState,
    trigger: MatchEvent,
    changes: List[Change],
    path_shift: float,
    swing: Optional[Tuple[float, float]],
) -> str:
    ev_desc = trigger.type.replace("_", " ")
    score = f"{state.home_score}–{state.away_score}"
    swing_line = (
        f"{_prob_label(state).capitalize()} for {trigger.team_name} moved "
        f"{swing[0]:.0%} → {swing[1]:.0%} on this event.\n"
        if swing
        else ""
    )
    lines = "\n".join(
        f"  {name}: {pb:.1%} → {pa:.1%}  ({'+' if d > 0 else ''}{d:.1%})"
        for name, pb, pa, d, _ in changes[:4]
    )
    bracket_context = (
        f"CHAMPION PROBABILITY SHIFTS "
        f"(two {CF_SIMS:,}-sim brackets, common random numbers):\n{lines}\n\n"
        f"Champion-probability mass relocated: {path_shift:.1%}."
    )
    task = (
        "Explain how this event reshapes the tournament outlook. Lead with "
        "the size of the shift and which team it favours or costs most — "
        "don't open by restating the scoreline or event type, the reader "
        "already knows those. Weave in a second team from the shifts list "
        "so the ripple is visible, not just the headline mover, and land on "
        "one non-obvious consequence: a team whose odds moved despite not "
        "playing, or a favourite quietly benefiting from the result. Write "
        "ONE tight paragraph, no more than 80 words — every clause should "
        "add new information."
    )

    return (
        f"[INST] You are the lead tournament analyst for a live World Cup 2026 "
        f"intelligence desk. Your job is to explain, vividly and specifically, "
        f"how one match event reshapes (or fails to reshape) the ENTIRE "
        f"tournament bracket — the kind of insight a smart fan couldn't get "
        f"just from watching the game.\n\n"
        f"MATCH: {state.home_name} {score} {state.away_name} "
        f"· Minute {trigger.elapsed}' · {state.status_short}\n"
        f"EVENT: {trigger.team_name} — {ev_desc}\n"
        f"{swing_line}\n"
        f"{bracket_context}\n\n"
        f"{task}\n\n"
        f"Use the actual percentages given. Avoid the words 'significant', "
        f"'crucial', 'notable', 'pivotal'. Only name teams that appear above — "
        f"do not invent a rival, precedent match, or scoreline not given here. "
        f"[/INST]"
    )


def _template(
    state: MatchState,
    trigger: MatchEvent,
    changes: List[Change],
    path_shift: float,
    after_teams: Sequence,
    swing: Optional[Tuple[float, float]],
) -> str:
    """Number-dense fallback narrative used when no LLM is reachable."""
    ev_desc = trigger.type.replace("_", " ")
    score = f"{state.home_score}–{state.away_score}"

    # this team's in-match win prob before and after the event
    swing_txt = ""
    if swing and abs(swing[1] - swing[0]) > 0.005:
        swing_txt = (
            f" The event swung {trigger.team_name}'s {_prob_label(state)} "
            f"from {_pct(swing[0])} to {_pct(swing[1])}."
        )

    if changes:
        # biggest riser and faller across the bracket
        risers = [c for c in changes if c[3] > 0]
        fallers = [c for c in changes if c[3] < 0]
        top = changes[0]
        name, pb, pa, delta, _ = top
        direction = "climbs" if delta > 0 else "slips"

        lead = (
            f"{path_shift * 100:.1f}% of the tournament's title probability just "
            f"relocated across the bracket. {name} {direction} from "
            f"{_pct(pb)} to {_pct(pa)} to lift the World Cup "
            f"({'+' if delta > 0 else ''}{delta * 100:.1f} points)"
        )

        # second team: the biggest faller
        second = ""
        if delta > 0 and fallers:
            fn, fpb, fpa, fd, _ = fallers[0]
            second = (
                f", while {fn} pays for it, sliding {fpb * 100:.1f}% → "
                f"{fpa * 100:.1f}%"
            )
        elif delta < 0 and risers:
            rn, rpb, rpa, rd, _ = risers[0]
            second = (
                f", while {rn} is the quiet beneficiary, rising {rpb * 100:.1f}% "
                f"→ {rpa * 100:.1f}%"
            )
        elif len(changes) > 1:
            n2, pb2, pa2, _, _ = changes[1]
            second = (
                f", and {n2} moves {pb2 * 100:.1f}% → {pa2 * 100:.1f}% "
                f"in the ripple"
            )

        return (
            f"{lead}{second}. The {ev_desc} in {state.home_name} {score} "
            f"{state.away_name} at {trigger.elapsed}' didn't just change this "
            f"result — it reweighted the whole draw.{swing_txt}"
        )

    # no material bracket change
    top = sorted(after_teams, key=lambda t: t.probs["champion"], reverse=True)[:3]
    leaders = ", ".join(f"{t.name} ({_pct(t.probs['champion'])})" for t in top)
    return (
        f"{trigger.team_name}'s {ev_desc} at {trigger.elapsed}' in "
        f"{state.home_name} {score} {state.away_name} didn't touch the "
        f"championship math — under {max(0.1, path_shift * 100):.1f}% of title "
        f"probability shifted, which the model treats as noise rather than "
        f"signal. That's because the two conditioned brackets only diverge on "
        f"scoreline and red cards, and this event changed neither.{swing_txt} "
        f"The favourites sit exactly where they did before kickoff on this "
        f"storyline: {leaders}. It would take a goal, an equaliser, or a "
        f"sending-off — not a caution or a fresh legs substitution — to move "
        f"any of those numbers again."
    )
