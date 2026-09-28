"""
Bracket counterfactual agent.

For each goal or red card:
    1. Rebuild the match state just before and after the event from the
       event timeline (api/match_timeline.around).
    2. Run two tournament simulations with common random numbers: earlier
       matches pinned to their real results, this match conditioned live on
       the before/after state. The champion-probability delta is the
       event's effect.
    3. Events that change neither score nor players on the pitch, or move
       no team past DELTA_THRESHOLD, get a template entry (no LLM).

Seeds use zlib.crc32 (stable across restarts); simulations run on
ml.executors.CF_SIM_EXECUTOR.
"""

import asyncio
import logging
import os
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

from agents.ollama_client import generate_with_source
from api.match_timeline import around
from api.schemas.event_types import TRIGGER_TYPES
from api.schemas.schema import MatchEvent, MatchState
from ml.executors import CF_SIM_EXECUTOR
from ml.in_play import inplay_wdl
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import match_wdl
from ml.team_names import to_sim
from ml.tournament_sim import LiveMatch, results_from_fixtures, run_simulation
from ml.wc_2026_config import FIXTURE_BY_ID, TEAM_BY_NAME

log = logging.getLogger(__name__)

DELTA_THRESHOLD = 0.003  # min |Δ champion prob| to list a team
CF_SIMS = int(
    os.getenv("CF_SIMS", "20000")
)  # paths per before/after simulation

# The prompt asks for one ~60-80 word paragraph; small models don't reliably
# self-limit, so this token cap (~80-90 words) is the backstop.
CF_NARRATIVE_MAX_TOKENS = 130
CF_NARRATIVE_NUM_CTX = 1280  # prompt + response must both fit this window

@dataclass
class CfState:
    covered: Set[str] = field(default_factory=set)
    last_time: float = 0.0
    MIN_GAP: float = 45.0


_states: Dict[int, CfState] = {}


def _sig(ev: MatchEvent) -> str:
    # ev.extra distinguishes same-minute events (e.g. two stoppage-time
    # subs both at elapsed=45) — without it a team's second trigger event
    # in the same displayed minute collides with the first and is silently
    # treated as already-covered.
    return f"{ev.elapsed}:{ev.extra or 0}:{ev.type}:{ev.team_id}"


def event_sig(elapsed: int, extra: Optional[int], ev_type: str, team_id: int) -> str:
    """Event signature; the worker uses it to rebuild coverage from the feed."""
    return f"{elapsed}:{extra or 0}:{ev_type}:{team_id}"


def seed_covered(fixture_id: int, sigs: Set[str]) -> None:
    """Mark signatures restored from the Redis feed as analysed."""
    cf = _states.setdefault(fixture_id, CfState())
    cf.covered |= sigs


def clear_state(fixture_id: int) -> None:
    _states.pop(fixture_id, None)


def _maybe_reset_on_replay(cf: CfState, current_elapsed: int) -> None:
    if not cf.covered or current_elapsed > 20:
        return
    covered_minutes = []
    for sig in cf.covered:
        try:
            covered_minutes.append(int(sig.split(":")[0]))
        except (ValueError, IndexError):
            pass
    if covered_minutes and max(covered_minutes) > current_elapsed + 20:
        log.info(
            f"Replay restart — covered up to {max(covered_minutes)}', "
            f"now at {current_elapsed}'. Resetting."
        )
        cf.covered.clear()
        cf.last_time = 0.0


def _find_trigger(state: MatchState, cf: CfState) -> Optional[MatchEvent]:
    current_elapsed = state.elapsed or 0
    _maybe_reset_on_replay(cf, current_elapsed)

    for ev in reversed(state.events):
        if ev.type not in TRIGGER_TYPES:
            continue
        if ev.elapsed > current_elapsed:
            continue
        if _sig(ev) in cf.covered:
            continue
        return ev
    return None


def _find_all_triggers(state: MatchState, cf: CfState) -> List[MatchEvent]:
    """Every uncovered trigger event in chronological order (for update_all)."""
    current_elapsed = state.elapsed or 0
    _maybe_reset_on_replay(cf, current_elapsed)

    out: List[MatchEvent] = []
    seen: Set[str] = set()
    for ev in sorted(state.events, key=lambda e: e.elapsed):
        if ev.type not in TRIGGER_TYPES:
            continue
        if ev.elapsed > current_elapsed:
            continue
        sig = _sig(ev)
        if sig in cf.covered or sig in seen:
            continue
        seen.add(sig)
        out.append(ev)
    return out


# ── No-impact events ───────────────────────────────────────────────────────


def _no_impact_entry(
    state: MatchState, trigger: MatchEvent, swing: Tuple[float, float], conditioned: bool
) -> dict:
    who = trigger.player_name or trigger.team_name
    what = trigger.type.replace("_", " ")
    reason = (
        "left the score and the numbers on the pitch unchanged, so the bracket "
        "outlook did not move."
        if conditioned
        else "is outside the simulated tournament."
    )
    return {
        "fixture_id": state.fixture_id,
        "minute": trigger.elapsed,
        "extra": trigger.extra,
        "event_type": trigger.type,
        "event_team": trigger.team_name,
        "event_team_id": trigger.team_id,
        "path_shift_pct": 0.0,
        "top_changes": [],
        "narrative": f"{who}'s {what} at {trigger.elapsed}' {reason}",
        "via": "template",
        "conditioned": conditioned,
        "match_win_prob_before": round(swing[0], 4),
        "match_win_prob_after": round(swing[1], 4),
        "n_sims": 0,
        "elapsed_s": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Divergence between two conditioned brackets ────────────────────────────


def _divergence(
    before_teams, after_teams
) -> Tuple[List[Tuple[str, float, float, float]], float]:
    before = {t.name: t.probs["champion"] for t in before_teams}
    after = {t.name: t.probs["champion"] for t in after_teams}
    changes: List[Tuple[str, float, float, float]] = []
    total_abs = 0.0
    for name, pb in before.items():
        pa = after.get(name, pb)
        d = pa - pb
        total_abs += abs(d)
        if abs(d) >= DELTA_THRESHOLD:
            changes.append((name, pb, pa, d))
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    # Sum of |Δ| double-counts (one team's gain is another's loss); halving
    # gives the champion-probability mass that actually relocated.
    path_shift = min(1.0, total_abs / 2.0)
    return changes, path_shift


# ── Narrative ──────────────────────────────────────────────────────────────


def _team_stage_line(after_teams, sim_name: str) -> Optional[str]:
    """One team's odds at every stage, not just champion."""
    for t in after_teams:
        if t.name == sim_name:
            p = t.probs
            return (
                f"{t.name}: {p.get('champion', 0):.1%} champion, "
                f"{p.get('final', 0):.1%} to reach the final, "
                f"{p.get('sf', 0):.1%} to reach the semifinal"
            )
    return None


def _build_prompt(state, trigger, changes, path_shift, after_teams, swing) -> str:
    ev_desc = trigger.type.replace("_", " ")
    score = f"{state.home_score}–{state.away_score}"
    swing_line = (
        f"In-play win probability for {trigger.team_name} moved "
        f"{swing[0]:.0%} → {swing[1]:.0%} on this event.\n"
        if swing
        else ""
    )
    if changes:
        lines = "\n".join(
            f"  {name}: {pb:.1%} → {pa:.1%}  ({'+' if d > 0 else ''}{d:.1%})"
            for name, pb, pa, d in changes[:4]
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
    else:
        home_line = _team_stage_line(after_teams, to_sim(state.home_name))
        away_line = _team_stage_line(after_teams, to_sim(state.away_name))
        team_lines = "\n".join(line for line in (home_line, away_line) if line)
        top_now = sorted(after_teams, key=lambda t: t.probs["champion"], reverse=True)[
            :3
        ]
        leaderboard = ", ".join(
            f"{t.name} {t.probs['champion']:.1%}" for t in top_now
        )
        bracket_context = (
            f"TOURNAMENT STATE — this event produced IDENTICAL odds to omitting "
            f"it entirely, because the simulator only conditions on scoreline and "
            f"dismissals, not on cautions or substitutions:\n"
            f"{team_lines}\n\n"
            f"Current championship-odds leaderboard: {leaderboard}."
        )
        task = (
            f"Explain why the simulator treated this {ev_desc} as inconsequential "
            f"— ground it in how the model actually works (it conditions on "
            f"scoreline and dismissals, not on cautions or substitutions), not "
            f"just 'nothing changed'. Use the multi-stage odds above (final, "
            f"semifinal, champion — not just one number) to say something real "
            f"about where {trigger.team_name} and their opponent actually stand "
            f"in the tournament right now. Do not end by simply noting that a "
            f"goal or red card would matter — that's obvious; instead close on "
            f"what's specifically at stake for these two teams. Write ONE tight "
            f"paragraph, no more than 60 words."
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


def _allowed_teams(state, changes, after_teams) -> set:
    """Team names the narrative may mention.

    The two sides playing, every team in the computed bracket shift and the
    top of the post-event leaderboard. Any other team is one the model saw
    no number for.
    """
    allowed = {state.home_name, state.away_name}
    allowed.update(name for name, *_ in changes)
    top = sorted(after_teams, key=lambda t: t.probs["champion"], reverse=True)[:8]
    allowed.update(t.name for t in top)
    return allowed


def _grounding_violation(narrative: str, state, changes, after_teams) -> bool:
    """True if the narrative names a WC 2026 team outside _allowed_teams."""
    allowed = _allowed_teams(state, changes, after_teams)
    for name in TEAM_BY_NAME:
        if name in allowed:
            continue
        if name in narrative:
            return True
    return False


def _ordinal_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _template(state, trigger, changes, path_shift, after_teams, swing) -> str:
    """Number-dense fallback narrative used when no LLM is reachable."""
    ev_desc = trigger.type.replace("_", " ")
    score = f"{state.home_score}–{state.away_score}"

    # Swing line (this team's in-match win prob before/after the event).
    swing_txt = ""
    if swing and abs(swing[1] - swing[0]) > 0.005:
        swing_txt = (
            f" The event swung {trigger.team_name}'s in-match win probability "
            f"from {_ordinal_pct(swing[0])} to {_ordinal_pct(swing[1])}."
        )

    if changes:
        # Biggest riser and biggest faller across the whole bracket.
        risers = [c for c in changes if c[3] > 0]
        fallers = [c for c in changes if c[3] < 0]
        top = changes[0]
        name, pb, pa, delta = top
        direction = "climbs" if delta > 0 else "slips"

        lead = (
            f"{path_shift * 100:.1f}% of the tournament's title probability just "
            f"relocated across the bracket. {name} {direction} from "
            f"{_ordinal_pct(pb)} to {_ordinal_pct(pa)} to lift the World Cup "
            f"({'+' if delta > 0 else ''}{delta * 100:.1f} points)"
        )

        # Add a contrasting second team for texture.
        second = ""
        if delta > 0 and fallers:
            fn, fpb, fpa, fd = fallers[0]
            second = (
                f", while {fn} pays for it, sliding {fpb * 100:.1f}% → "
                f"{fpa * 100:.1f}%"
            )
        elif delta < 0 and risers:
            rn, rpb, rpa, rd = risers[0]
            second = (
                f", while {rn} is the quiet beneficiary, rising {rpb * 100:.1f}% "
                f"→ {rpa * 100:.1f}%"
            )
        elif len(changes) > 1:
            n2, pb2, pa2, d2 = changes[1]
            second = (
                f", and {n2} moves {pb2 * 100:.1f}% → {pa2 * 100:.1f}% "
                f"in the ripple"
            )

        return (
            f"{lead}{second}. The {ev_desc} in {state.home_name} {score} "
            f"{state.away_name} at {trigger.elapsed}' didn't just change this "
            f"result — it reweighted the whole draw.{swing_txt}"
        )

    # No material bracket change: explain why.
    top = sorted(after_teams, key=lambda t: t.probs["champion"], reverse=True)[:3]
    leaders = (
        ", ".join(f"{t.name} ({_ordinal_pct(t.probs['champion'])})" for t in top)
        if top
        else "the field"
    )
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


# ── Main entry point ───────────────────────────────────────────────────────


def _stable_seed(fid: int, trigger: MatchEvent) -> int:
    """Deterministic across process restarts (hash() is salt-randomised)."""
    raw = f"{fid}:{trigger.elapsed}:{trigger.type}:{trigger.team_id}".encode()
    return zlib.crc32(raw) % 2_147_483_646 + 1


async def update(
    state: MatchState,
    loop: asyncio.AbstractEventLoop,
    on_start: Optional[Callable[[dict], Awaitable[None]]] = None,
    results: Optional[Dict[int, dict]] = None,
) -> Optional[dict]:
    """Analyse the newest uncovered trigger event, at most once per MIN_GAP.

    Live path. `on_start`, if given, is awaited before the simulations run
    so the worker can publish a "calculating" signal.
    """
    fid = state.fixture_id
    cf = _states.setdefault(fid, CfState())

    if time.time() - cf.last_time < cf.MIN_GAP:
        return None

    trigger = _find_trigger(state, cf)
    if not trigger:
        return None

    cf.last_time = time.time()
    return await _analyze_trigger(state, cf, trigger, loop, on_start=on_start, results=results)


async def update_all(
    state: MatchState,
    loop: asyncio.AbstractEventLoop,
    on_start: Optional[Callable[[dict], Awaitable[None]]] = None,
    max_events: int = 20,
    results: Optional[Dict[int, dict]] = None,
) -> List[dict]:
    """Analyse every uncovered trigger event, oldest first (backfill path).

    Not MIN_GAP-throttled. `max_events` caps one pass; the rest run next tick.
    """
    fid = state.fixture_id
    cf = _states.setdefault(fid, CfState())

    triggers = _find_all_triggers(state, cf)[:max_events]
    if not triggers:
        return []

    results: List[dict] = []
    for trigger in triggers:
        result = await _analyze_trigger(
            state, cf, trigger, loop, on_start=on_start, results=results
        )
        if result is not None:
            results.append(result)
    cf.last_time = time.time()
    return results


async def _analyze_trigger(
    state: MatchState,
    cf: "CfState",
    trigger: MatchEvent,
    loop: asyncio.AbstractEventLoop,
    on_start: Optional[Callable[[dict], Awaitable[None]]] = None,
    results: Optional[Dict[int, dict]] = None,
) -> Optional[dict]:
    """Two CRN-paired conditioned brackets for one event, then a narrative."""
    fid = state.fixture_id

    if on_start is not None:
        try:
            await on_start(
                {
                    "minute": trigger.elapsed,
                    "event_type": trigger.type,
                    "event_team": trigger.team_name,
                }
            )
        except Exception:
            log.warning(f"[{fid}] on_start callback failed", exc_info=True)

    odds_client = get_oddsapi_client()
    odds_table = await odds_client.get_all_odds()  # cached — see odds_api_client

    cf.covered.add(_sig(trigger))

    fx = FIXTURE_BY_ID.get(fid)
    home_c, away_c = to_sim(state.home_name), to_sim(state.away_name)
    conditioned = fx is not None
    pre_wdl = match_wdl(
        home_c, away_c, host_side=fx.get("host_side") if fx else None, odds_table=odds_table
    )

    # Match state just before / just after the event, not the final score.
    before, after = around(state, trigger)
    minute = trigger.elapsed
    after_wdl = inplay_wdl(pre_wdl, minute, *after, extra=trigger.extra)
    before_wdl = inplay_wdl(pre_wdl, minute, *before, extra=trigger.extra)

    acted_home = trigger.team_id == 1
    swing = (
        (before_wdl[0], after_wdl[0]) if acted_home else (before_wdl[2], after_wdl[2])
    )

    # An event that changes neither score nor numbers on the pitch cannot
    # move the bracket — skip both simulations and the LLM call.
    if not conditioned or before == after:
        return _no_impact_entry(state, trigger, swing, conditioned)

    if results is None:
        results = results_from_fixtures(
            before=state.kickoff_time.strftime("%Y-%m-%dT%H:%MZ") if state.kickoff_time else ""
        )
    pinned = {k: v for k, v in results.items() if k != fid}
    seed = _stable_seed(fid, trigger)

    def _sim(snap):
        return run_simulation(
            odds_table=odds_table,
            n_sims=CF_SIMS,
            seed=seed,
            results=pinned,
            live={fid: LiveMatch(minute, *snap, extra=trigger.extra)},
        )

    try:
        after_result, before_result = await asyncio.gather(
            loop.run_in_executor(CF_SIM_EXECUTOR, lambda: _sim(after)),
            loop.run_in_executor(CF_SIM_EXECUTOR, lambda: _sim(before)),
        )
    except Exception as exc:
        log.error(f"[{fid}] CF sim failed: {exc}")
        cf.covered.discard(_sig(trigger))
        return None

    changes, path_shift = _divergence(before_result.teams, after_result.teams)

    # No team moved past DELTA_THRESHOLD: the template says so; skip the LLM.
    narrative, via = "", "template"
    if changes:
        prompt = _build_prompt(state, trigger, changes, path_shift, after_result.teams, swing)
        narrative, via = await generate_with_source(
            prompt,
            timeout=35.0,
            max_tokens=CF_NARRATIVE_MAX_TOKENS,
            num_ctx=CF_NARRATIVE_NUM_CTX,
        )
    if narrative and _grounding_violation(narrative, state, changes, after_result.teams):
        log.warning(f"[{fid}] CF narrative failed grounding check — discarding")
        narrative = ""
    if not narrative:
        narrative = _template(
            state, trigger, changes, path_shift, after_result.teams, swing
        )
        via = "template"

    log.info(
        f"[{fid}] CF: {trigger.type}@{trigger.elapsed}' "
        f"path_shift={path_shift:.2%} changes={len(changes)}"
    )

    return {
        "fixture_id": fid,
        "minute": trigger.elapsed,
        "extra": trigger.extra,  # stoppage-time sub-minute; lets the worker rebuild coverage
        "event_type": trigger.type,
        "event_team": trigger.team_name,
        "event_team_id": trigger.team_id,  # lets the worker rebuild coverage
        "path_shift_pct": round(path_shift, 3),
        "top_changes": [
            {
                "team": name,
                "before": round(pb, 4),
                "after": round(pa, 4),
                "delta": round(d, 4),
            }
            for name, pb, pa, d in changes[:4]
        ],
        "narrative": narrative,
        "via": via,
        "conditioned": conditioned,
        "match_win_prob_before": round(swing[0], 4),
        "match_win_prob_after": round(swing[1], 4),
        "n_sims": after_result.n_sims,
        "elapsed_s": after_result.elapsed_s,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
