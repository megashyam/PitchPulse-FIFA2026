"""
Bracket counterfactual agent.

For each goal or red card:
    1. Rebuild the match state just before and after the event from the
       event timeline (api/match_timeline.around).
    2. Run two tournament simulations with common random numbers: earlier
       matches pinned to their real results, this match conditioned live on
       the before/after state. The champion-probability delta is the
       event's effect; its SE is the paired std(hit_after - hit_before)/√N.
    3. Events that change neither score nor players on the pitch, or move
       no team past max(DELTA_THRESHOLD, 2·SE), get a template entry (no LLM).

Seeds use zlib.crc32 (stable across restarts); simulations run on
ml.executors.CF_SIM_EXECUTOR.
"""

import asyncio
import logging
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from agents.cf_prompts import _build_prompt, _no_impact_text, _template
from agents.grounding import team_violations
from agents.ollama_client import generate_with_source
from api.match_timeline import around
from api.tournament_state import side_probs
from schemas.event_types import TRIGGER_TYPES
from schemas.feeds import CfEntry, Change
from schemas.schema import MatchEvent, MatchState
from ml.executors import CF_SIM_EXECUTOR
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import prior_for
from ml.team_names import canonical
from ml.tournament_sim import LiveMatch, results_from_fixtures, run_simulation
from ml.wc_2026_config import FIXTURE_BY_ID

from config import (
    CF_MAX_EVENTS_PER_PASS,
    CF_MIN_GAP_S,
    CF_NARRATIVE_MAX_TOKENS,
    CF_SIMS,
    DELTA_THRESHOLD,
)

log = logging.getLogger(__name__)


@dataclass
class CfState:
    covered: Set[str] = field(default_factory=set)
    last_time: float = 0.0


_states: Dict[int, CfState] = {}


def _sig(ev: MatchEvent) -> str:
    # ev.extra separates same-minute events (two stoppage-time subs at elapsed=45);
    # without it the second collides with the first and counts as covered
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


def _find_all_triggers(state: MatchState, cf: CfState) -> List[MatchEvent]:
    """Uncovered trigger events, oldest first; update_latest() takes the last, update_backlog() all."""
    current_elapsed = state.elapsed or 0

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


def _entry(
    state: MatchState,
    trigger: MatchEvent,
    swing: Tuple[float, float],
    conditioned: bool,
    narrative: str,
    via: str = "template",
    path_shift: float = 0.0,
    changes: Sequence[Change] = (),
    n_sims: int = 0,
    elapsed_s: float = 0.0,
) -> CfEntry:
    return {
        "fixture_id": state.fixture_id,
        "minute": trigger.elapsed,
        "extra": trigger.extra,  # stoppage-time sub-minute; lets the worker rebuild coverage
        "event_type": trigger.type,
        "event_team": trigger.team_name,
        "event_team_id": trigger.team_id,  # lets the worker rebuild coverage
        "path_shift_pct": round(path_shift, 3),
        "top_changes": [
            {
                "team": c.team,
                "before": round(c.before, 4),
                "after": round(c.after, 4),
                "delta": round(c.delta, 4),
                "delta_se": round(c.se, 4) if c.se is not None else None,
            }
            for c in changes[:4]
        ],
        "narrative": narrative,
        "via": via,
        "conditioned": conditioned,
        "match_win_prob_before": round(swing[0], 4),
        "match_win_prob_after": round(swing[1], 4),
        "n_sims": n_sims,
        "elapsed_s": elapsed_s,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _no_impact_entry(
    state: MatchState, trigger: MatchEvent, swing: Tuple[float, float], conditioned: bool
) -> CfEntry:
    return _entry(state, trigger, swing, conditioned, _no_impact_text(trigger, conditioned))


def _paired_se(before_result, after_result) -> Optional["np.ndarray"]:
    """Per-team SE of the CRN champion delta: std(hit_after − hit_before)/√N."""
    hb, ha = before_result.champion_hits, after_result.champion_hits
    if hb is None or ha is None:
        return None
    d = ha.astype(np.int8) - hb.astype(np.int8)
    return d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])


def _divergence(before_result, after_result) -> Tuple[List[Change], float]:
    """Teams whose champion probability moved by at least
    max(DELTA_THRESHOLD, 2·SE), with the paired SE when hits were kept."""
    se = _paired_se(before_result, after_result)
    after = {t.name: t.probs["champion"] for t in after_result.teams}
    changes: List[Change] = []
    total_abs = 0.0
    for i, t in enumerate(before_result.teams):
        pb = t.probs["champion"]
        pa = after.get(t.name, pb)
        d = pa - pb
        total_abs += abs(d)
        s_i = float(se[i]) if se is not None else None
        if abs(d) >= max(DELTA_THRESHOLD, 2 * (s_i or 0.0)):
            changes.append(Change(t.name, pb, pa, d, s_i))
    changes.sort(key=lambda c: abs(c.delta), reverse=True)
    # sum of |Δ| counts every move twice (one team's gain is another's loss); halved it
    # is the champion-probability mass that moved
    path_shift = min(1.0, total_abs / 2.0)
    return changes, path_shift


def _cf_allowed_teams(state, changes, after_teams) -> set:
    """Team names the narrative may mention.

    The two sides playing, every team in the computed bracket shift and the
    top of the post-event leaderboard. Any other team is one the model saw
    no number for.
    """
    allowed = {state.home_name, state.away_name}
    allowed.update(c.team for c in changes)
    top = sorted(after_teams, key=lambda t: t.probs["champion"], reverse=True)[:8]
    allowed.update(t.name for t in top)
    return allowed


async def _narrate(state, trigger, changes, path_shift, swing, after_teams) -> Tuple[str, str]:
    """(narrative, via). LLM only when some team moved; an ungrounded or empty
    LLM answer falls back to the template."""
    narrative, via = "", "template"
    if changes:
        prompt = _build_prompt(state, trigger, changes, path_shift, swing)
        narrative, via = await generate_with_source(
            prompt,
            timeout=35.0,
            max_tokens=CF_NARRATIVE_MAX_TOKENS,
        )
    if narrative and team_violations(narrative, _cf_allowed_teams(state, changes, after_teams)):
        log.warning(f"[{state.fixture_id}] CF narrative failed grounding check — discarding")
        narrative = ""
    if not narrative:
        return _template(state, trigger, changes, path_shift, after_teams, swing), "template"
    return narrative, via


def _stable_seed(fid: int, trigger: MatchEvent) -> int:
    """Deterministic across process restarts (hash() is salt-randomised)."""
    raw = f"{fid}:{trigger.elapsed}:{trigger.type}:{trigger.team_id}".encode()
    return zlib.crc32(raw) % 2_147_483_646 + 1


async def update_latest(
    state: MatchState,
    loop: asyncio.AbstractEventLoop,
    results: Optional[Dict[int, dict]] = None,
) -> Optional[CfEntry]:
    """Analyse the newest uncovered trigger event, at most once per CF_MIN_GAP_S (live path)."""
    fid = state.fixture_id
    cf = _states.setdefault(fid, CfState())

    if time.time() - cf.last_time < CF_MIN_GAP_S:
        return None

    triggers = _find_all_triggers(state, cf)
    if not triggers:
        return None
    trigger = triggers[-1]

    cf.last_time = time.time()
    return await _analyze_trigger(state, cf, trigger, loop, results=results)


async def update_backlog(
    state: MatchState,
    loop: asyncio.AbstractEventLoop,
    results: Optional[Dict[int, dict]] = None,
) -> List[CfEntry]:
    """Analyse every uncovered trigger event, oldest first (backfill path).

    Not CF_MIN_GAP_S-throttled. CF_MAX_EVENTS_PER_PASS caps one pass; the rest run next tick.
    """
    fid = state.fixture_id
    cf = _states.setdefault(fid, CfState())

    triggers = _find_all_triggers(state, cf)[:CF_MAX_EVENTS_PER_PASS]
    if not triggers:
        return []

    out: List[CfEntry] = []
    for trigger in triggers:
        result = await _analyze_trigger(state, cf, trigger, loop, results=results)
        if result is not None:
            out.append(result)
    cf.last_time = time.time()
    return out


async def _analyze_trigger(
    state: MatchState,
    cf: "CfState",
    trigger: MatchEvent,
    loop: asyncio.AbstractEventLoop,
    results: Optional[Dict[int, dict]] = None,
) -> Optional[CfEntry]:
    """Two CRN-paired conditioned brackets for one event, then a narrative."""
    fid = state.fixture_id

    odds_client = get_oddsapi_client()
    odds_table = await odds_client.get_all_odds()  # cached: see odds_api_client

    cf.covered.add(_sig(trigger))

    conditioned = fid in FIXTURE_BY_ID
    pre_wdl, totals, _ = prior_for(
        canonical(state.home_name), canonical(state.away_name), fid, odds_table
    )

    # match state just before and just after the event, not the final score
    before, after = around(state, trigger)
    minute = trigger.elapsed
    after_wdl = side_probs(pre_wdl, fid, minute, after, extra=trigger.extra, totals=totals)
    before_wdl = side_probs(pre_wdl, fid, minute, before, extra=trigger.extra, totals=totals)

    acted_home = trigger.team_id == 1
    swing = (
        (before_wdl[0], after_wdl[0]) if acted_home else (before_wdl[2], after_wdl[2])
    )

    # an event that changes neither score nor players on the pitch cannot move the
    # bracket: skip both simulations and the LLM
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
            keep_hits=True,
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

    changes, path_shift = _divergence(before_result, after_result)
    # no team past DELTA_THRESHOLD: template, no LLM call
    narrative, via = await _narrate(state, trigger, changes, path_shift, swing, after_result.teams)

    log.info(
        f"[{fid}] CF: {trigger.type}@{trigger.elapsed}' "
        f"path_shift={path_shift:.2%} changes={len(changes)}"
    )
    return _entry(
        state, trigger, swing, conditioned, narrative, via,
        path_shift=path_shift, changes=changes,
        n_sims=after_result.n_sims, elapsed_s=after_result.elapsed_s,
    )
