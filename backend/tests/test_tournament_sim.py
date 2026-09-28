"""
Tournament simulator tests run through the production entrypoint.

Calls run_simulation(odds_table=<live snapshot>) so priors come from the live
market when available, then verifies invariants that must hold for any valid
Monte Carlo run:
    - probability conservation
    - bracket slot counts
    - deterministic seeding
    - common-random-number alignment

The invariants are input-independent, so they hold whether priors come from
live odds or the Elo fallback.
"""

import numpy as np
import pytest

from ml.tournament_sim import run_simulation, STAGES
from ml.wc2026_format import R32_SLOTS
from ml.wc_2026_config import WC2026_TEAMS, GROUPS

N = 8000
SEED = 1234


@pytest.fixture(scope="module")
def live_result(request):
    """Sim via the odds-driven entrypoint; Elo fallback only if odds are empty."""
    odds = None
    try:
        import asyncio
        from ml.odds_api_client import get_oddsapi_client

        odds = (
            asyncio.get_event_loop().run_until_complete(
                get_oddsapi_client().get_all_odds()
            )
            or None
        )
    except Exception:
        odds = None
    res = run_simulation(odds_table=odds, n_sims=N, seed=SEED)
    res._used_live_odds = bool(odds)
    return res


def test_config_shape():
    assert len(WC2026_TEAMS) == 48
    assert len(GROUPS) == 12 and all(len(v) == 4 for v in GROUPS.values())
    assert len(R32_SLOTS) == 16


@pytest.mark.integration
@pytest.mark.odds
def test_championship_probs_sum_to_one(live_result):
    assert sum(t.probs["champion"] for t in live_result.teams) == pytest.approx(
        1.0, abs=1e-9
    )


@pytest.mark.integration
@pytest.mark.odds
def test_stage_conservation(live_result):
    expected = {"r32": 32, "r16": 16, "qf": 8, "sf": 4, "final": 2, "champion": 1}
    for stage, k in expected.items():
        assert sum(t.probs[stage] for t in live_result.teams) == pytest.approx(
            float(k), abs=1e-9
        ), stage


@pytest.mark.integration
@pytest.mark.odds
def test_group_exit_complement(live_result):
    assert sum(t.probs["group_exit"] for t in live_result.teams) == pytest.approx(
        16.0, abs=1e-9
    )
    for t in live_result.teams:
        assert t.probs["group_exit"] == pytest.approx(1.0 - t.probs["r32"], abs=1e-9)


@pytest.mark.integration
@pytest.mark.odds
def test_stage_monotonicity(live_result):
    order = ["r32", "r16", "qf", "sf", "final", "champion"]
    for t in live_result.teams:
        for a, b in zip(order, order[1:]):
            assert t.probs[b] <= t.probs[a] + 1e-12, t.name


@pytest.mark.integration
@pytest.mark.odds
def test_group_position_probs_sum(live_result):
    for key in ("group_first", "group_second", "group_third", "group_fourth"):
        assert sum(t.probs[key] for t in live_result.teams) == pytest.approx(
            12.0, abs=1e-9
        ), key
    for t in live_result.teams:
        s = sum(
            t.probs[k]
            for k in ("group_first", "group_second", "group_third", "group_fourth")
        )
        assert s == pytest.approx(1.0, abs=1e-9), t.name


@pytest.mark.integration
@pytest.mark.odds
def test_confidence_intervals_bracket_estimate(live_result):
    for t in live_result.teams:
        for stage in STAGES:
            lo, hi = t.ci_95[stage]
            assert lo - 1e-9 <= t.probs[stage] <= hi + 1e-9
            assert 0.0 <= lo <= hi <= 1.0


@pytest.mark.integration
def test_seed_determinism_live_path():
    """Same seed + same live odds snapshot → identical output."""
    import asyncio
    from ml.odds_api_client import get_oddsapi_client

    try:
        odds = (
            asyncio.get_event_loop().run_until_complete(
                get_oddsapi_client().get_all_odds()
            )
            or None
        )
    except Exception:
        odds = None
    a = run_simulation(odds_table=odds, n_sims=2000, seed=99)
    b = run_simulation(odds_table=odds, n_sims=2000, seed=99)
    for ta, tb in zip(a.teams, b.teams):
        assert ta.probs == tb.probs


@pytest.mark.integration
def test_elo_override_moves_probability(live_result):
    name = WC2026_TEAMS[0].name
    base = run_simulation(n_sims=N, seed=7)
    boosted = run_simulation(n_sims=N, seed=7, elo_overrides={name: 150.0})
    assert boosted.team(name).probs["champion"] > base.team(name).probs["champion"]


@pytest.mark.integration
def test_crn_shared_seed_streams_align():
    a = run_simulation(n_sims=2000, seed=42, elo_overrides={})
    b = run_simulation(n_sims=2000, seed=42, elo_overrides={WC2026_TEAMS[0].name: 0.0})
    for ta, tb in zip(a.teams, b.teams):
        assert ta.probs == tb.probs


@pytest.mark.integration
@pytest.mark.odds
def test_stronger_team_higher_champion_prob(live_result):
    elos = np.array([t.elo for t in live_result.teams])
    champ = np.array([t.probs["champion"] for t in live_result.teams])
    corr = np.corrcoef(np.argsort(np.argsort(elos)), np.argsort(np.argsort(champ)))[
        0, 1
    ]
    assert corr > 0.6  # looser under live odds, which can outrank base Elo
