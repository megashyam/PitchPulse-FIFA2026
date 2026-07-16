"""
ml/real_bracket_override.py
============================
Overrides the Monte Carlo tournament prediction with REAL knockout results
already known from the live match feed, so the Predictor panel doesn't keep
showing a nonzero championship probability for a team that's actually been
eliminated (or fail to reflect a team that's already guaranteed a Final
spot).

Scope / limitation (see tonight's Predictor investigation): this only
touches KNOCKOUT-stage (r32 onward) teams whose real fate is already
recorded in match state. It deliberately does NOT rebuild the group stage
or roster — ml/wc_2026_config.py's team list/groups are placeholder data
that doesn't match the real worldcup26.ir schedule (12/48 real teams aren't
even in the simulator's roster). If the real KO bracket hasn't started yet
(no r32/r16/qf/sf/final fixture in the feed), compute_real_overrides()
returns None and the caller should use the plain Monte Carlo result as-is.

For the handful of teams still genuinely alive once only 1-2 real matches
remain undecided, this computes an EXACT probability tree (not another
Monte Carlo sample) from the same odds/Elo priors used everywhere else in
the app — with that few matches left, exact is both simpler and more
precise than resampling.
"""

import logging
from typing import Dict, List, Optional, Set

import redis.asyncio as aioredis

from api.schemas.event_types import COMPLETED_STATUSES
from api.schemas.schema import MatchState
from ml.odds_api_client import get_oddsapi_client
from ml.prior_builder import build_prior_table, elo_to_wdl, ko_prob
from ml.team_names import is_sim_team, to_sim
from ml.wc_2026_config import TEAM_BY_NAME

log = logging.getLogger(__name__)

_STAGE_ORDER = ["r32", "r16", "qf", "sf", "final"]
_ALL_KO_STAGES = _STAGE_ORDER + ["champion"]

_ROUND_LABELS = {
    "r32": "r32",
    "round of 32": "r32",
    "r16": "r16",
    "round of 16": "r16",
    "qf": "qf",
    "quarter-final": "qf",
    "quarterfinal": "qf",
    "sf": "sf",
    "semi-final": "sf",
    "semifinal": "sf",
    "final": "final",
}


def _round_stage(round_str: str) -> Optional[str]:
    return _ROUND_LABELS.get((round_str or "").strip().lower())


async def compute_real_overrides(
    r: aioredis.Redis,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Team name -> {stage: probability} overrides derived from real results,
    or None if the real KO bracket doesn't exist in the feed yet."""
    keys = await r.keys("match:*:state")
    if not keys:
        return None
    raw_values = await r.mget(keys)

    ko_matches: List[tuple] = []
    for raw in raw_values:
        if not raw:
            continue
        try:
            m = MatchState.model_validate_json(raw)
        except Exception:
            continue
        stage = _round_stage(m.round)
        if stage:
            ko_matches.append((stage, m))

    if not ko_matches:
        return None

    # PASS 1: which real stages each team appears in — independent of who
    # won any individual match. Merely being scheduled into a stage's
    # fixture proves a team reached it, which is the most robust signal we
    # have (versus inferring "reached" purely from a computed winner).
    reached: Dict[str, Set[str]] = {}
    completed: List[tuple] = []
    pending_match: Optional[MatchState] = None
    ko_matches.sort(key=lambda x: _STAGE_ORDER.index(x[0]))

    for stage, m in ko_matches:
        home, away = to_sim(m.home_name), to_sim(m.away_name)
        reached.setdefault(home, set()).add(stage)
        reached.setdefault(away, set()).add(stage)
        if m.status_short in COMPLETED_STATUSES:
            completed.append((stage, m))
        elif pending_match is None:
            # Genuinely undecided real match. If more than one KO-round
            # fixture is undecided at once, only the further-along teams'
            # numbers are guaranteed exact — earlier open rounds fall back
            # to the plain simulation for their stage-of-uncertainty.
            pending_match = m

    # PASS 2: determine the LOSER of every completed match. A decisive
    # scoreline settles it directly. A level scoreline (extra time draw
    # resolved by penalties — the feed doesn't expose shootout scorers, and
    # status_short stays "FT" either way, confirmed via a real Germany 1-1
    # Paraguay R32 match) is settled by checking which of the two teams
    # goes on to appear in a LATER real round's fixture per pass 1.
    stage_idx = {s: i for i, s in enumerate(_STAGE_ORDER)}
    eliminated_at: Dict[str, str] = {}

    for stage, m in completed:
        home, away = to_sim(m.home_name), to_sim(m.away_name)
        if m.home_score != m.away_score:
            winner = home if m.home_score > m.away_score else away
        else:
            cur_idx = stage_idx[stage]
            home_later = any(
                stage_idx.get(s, -1) > cur_idx for s in reached.get(home, ())
            )
            away_later = any(
                stage_idx.get(s, -1) > cur_idx for s in reached.get(away, ())
            )
            if home_later and not away_later:
                winner = home
            elif away_later and not home_later:
                winner = away
            else:
                log.debug(
                    f"Could not resolve winner of drawn KO match "
                    f"{home} vs {away} ({stage}) from later-round schedule"
                )
                continue

        loser = away if winner == home else home
        eliminated_at[loser] = stage

        # Credit the winner with the NEXT stage explicitly — the schedule
        # may not have that fixture published yet (confirmed: a team that
        # just won its semifinal has no "Final" fixture to be found by pass
        # 1 until the feed actually creates one), so advancement can't rely
        # on schedule appearance alone once a team is ahead of the feed.
        next_idx = stage_idx[stage] + 1
        next_stage = (
            _STAGE_ORDER[next_idx] if next_idx < len(_STAGE_ORDER) else "champion"
        )
        reached.setdefault(winner, set()).add(next_stage)

    overrides: Dict[str, Dict[str, float]] = {}

    def _set_capped(team: str, floor_stage: str) -> None:
        team_ovr = overrides.setdefault(team, {})
        stop_idx = _ALL_KO_STAGES.index(floor_stage) + 1
        for i, s in enumerate(_ALL_KO_STAGES):
            team_ovr[s] = 1.0 if i < stop_idx else 0.0

    for team in reached:
        if is_sim_team(team) and team in eliminated_at:
            _set_capped(team, eliminated_at[team])

    # Every simulator team that never appears in any real KO match at all
    # did not qualify for r32 in reality.
    ko_participants = set(reached.keys())
    for team_name in TEAM_BY_NAME:
        if team_name in ko_participants:
            continue
        overrides[team_name] = {s: 0.0 for s in _ALL_KO_STAGES}
        overrides[team_name]["group_exit"] = 1.0

    alive = [t for t in reached if is_sim_team(t) and t not in eliminated_at]
    if pending_match is not None and len(alive) >= 2:
        try:
            await _apply_alive_tree(overrides, alive, pending_match, reached)
        except Exception as exc:
            log.warning(f"Real-bracket alive-team probability calc failed: {exc}")

    return overrides or None


async def _apply_alive_tree(
    overrides: Dict[str, Dict[str, float]],
    alive: List[str],
    pending_match: MatchState,
    reached: Dict[str, Set[str]],
) -> None:
    """Exact probability tree for the teams still alive once only 1-2 real
    matches remain — one pending semifinal-or-later fixture plus, for the
    side that's already through, the still-hypothetical Final."""
    odds_client = get_oddsapi_client()
    odds_table = await odds_client.get_all_odds()
    priors = build_prior_table(odds_table)

    def _wdl(a: str, b: str):
        if (a, b) in priors:
            return priors[(a, b)]
        if (b, a) in priors:
            pw, pd, pl = priors[(b, a)]
            return pl, pd, pw
        ta, tb = TEAM_BY_NAME.get(a), TEAM_BY_NAME.get(b)
        if ta and tb:
            return elo_to_wdl(ta.elo, tb.elo)
        return (0.40, 0.25, 0.35)

    pending_home = to_sim(pending_match.home_name)
    pending_away = to_sim(pending_match.away_name)
    pending_pair = [t for t in (pending_home, pending_away) if t in alive]
    confirmed_finalist = [t for t in alive if "final" in reached.get(t, set())]

    if len(confirmed_finalist) != 1 or len(pending_pair) != 2:
        # More than one open match, or the pending fixture isn't (yet) the
        # last one standing between the field and the Final — leave the
        # plain simulation's numbers for these teams rather than risk a
        # wrong exact-tree assumption. Logged (not just silent) because this
        # means `alive` teams are shown with the plain Monte Carlo number
        # even though we know some of them are through to a later real
        # round — no per-team flag exists in `overrides` to mark that
        # distinction for the frontend.
        log.info(
            f"Real-bracket exact-tree skipped — {len(confirmed_finalist)} confirmed "
            f"finalist(s), {len(pending_pair)} pending-match team(s) in alive={alive}; "
            f"falling back to plain sim numbers for these teams."
        )
        return

    finalist = confirmed_finalist[0]
    cand_a, cand_b = pending_pair

    p_a_adv, p_b_adv = ko_prob(*_wdl(cand_a, cand_b))
    p_a_champ_if_adv, p_fin_champ_if_a = ko_prob(*_wdl(cand_a, finalist))
    p_b_champ_if_adv, p_fin_champ_if_b = ko_prob(*_wdl(cand_b, finalist))

    overrides.setdefault(cand_a, {})["final"] = round(p_a_adv, 6)
    overrides.setdefault(cand_a, {})["champion"] = round(p_a_adv * p_a_champ_if_adv, 6)
    overrides.setdefault(cand_b, {})["final"] = round(p_b_adv, 6)
    overrides.setdefault(cand_b, {})["champion"] = round(p_b_adv * p_b_champ_if_adv, 6)
    overrides.setdefault(finalist, {})["final"] = 1.0
    overrides.setdefault(finalist, {})["champion"] = round(
        p_a_adv * p_fin_champ_if_a + p_b_adv * p_fin_champ_if_b, 6
    )
