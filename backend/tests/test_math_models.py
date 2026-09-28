"""
Probability model tests exercised on real inputs.

Validates the models against production data rather than hand-picked numbers:
    - Shin de-vigging on live market odds
    - Elo-to-W/D/L conversion using StatsBomb-derived Elo ratings
    - Poisson in-play probabilities, including calibration to the prior at kickoff

Deterministic edge cases (numerical overflow, full-time collapse) are kept as
fast local checks, since they must hold for any valid input.
"""

import math

import pytest

from ml.prior_builder import (
    elo_expected,
    elo_to_wdl,
    oddsapi_to_wdl,
    ko_prob,
    shin_devig,
)
from ml.in_play import inplay_wdl, elo_deltas, _poisson_pmf
from eval.eval_anomaly_threshold import aggregate
from ml.backtest_elo_wdl import ELO_START, HOME_ADV, load_matches, update_elo

# --------------------------------------------------------------- real Elo priors


@pytest.fixture(scope="module")
def real_elo():
    """Elo ratings learned from the full real StatsBomb WC history."""
    matches = load_matches()
    if not matches:
        pytest.skip("StatsBomb match list unreachable")
    elo: dict[str, float] = {}
    for m in matches:
        if m.get("home_score") is None:
            continue
        update_elo(
            elo,
            m["home_team"]["home_team_name"],
            m["away_team"]["away_team_name"],
            int(m["home_score"]),
            int(m["away_score"]),
        )
    return elo


@pytest.mark.integration
@pytest.mark.statsbomb
def test_elo_to_wdl_valid_on_every_real_pairing(real_elo):
    teams = list(real_elo)
    assert len(teams) > 8
    for h in teams[:12]:
        for a in teams[:12]:
            if h == a:
                continue
            w, d, loss = elo_to_wdl(real_elo[h] + HOME_ADV, real_elo[a])
            assert w + d + loss == pytest.approx(1.0, abs=1e-9)
            assert min(w, d, loss) >= 0.01 - 1e-12
            # prior_builder.elo_to_wdl clips the draw to [0.03, 0.30]
            assert 0.03 - 1e-9 <= d <= 0.30 + 1e-9


@pytest.mark.integration
@pytest.mark.statsbomb
def test_elo_expected_matches_real_strength_gap(real_elo):
    ranked = sorted(real_elo, key=real_elo.get, reverse=True)
    strong, weak = ranked[0], ranked[-1]
    assert elo_expected(real_elo[strong], real_elo[weak]) > 0.5
    # standard Elo identity holds on real ratings
    e = elo_expected(real_elo[strong], real_elo[weak])
    assert elo_expected(real_elo[weak], real_elo[strong]) == pytest.approx(
        1 - e, abs=1e-9
    )


@pytest.mark.integration
@pytest.mark.statsbomb
def test_inplay_valid_across_real_match_states(statsbomb_events, real_elo):
    d = statsbomb_events
    pre = elo_to_wdl(
        real_elo.get(d["home"], ELO_START) + HOME_ADV,
        real_elo.get(d["away"], ELO_START),
    )
    for minute in (0, 15, 44, 45, 70, 89):
        for hs, aw in [(0, 0), (d["home_score"], d["away_score"]), (2, 2)]:
            for rh, ra in [(0, 0), (1, 0), (0, 1)]:
                w, dr, loss = inplay_wdl(pre, minute, hs, aw, rh, ra)
                assert w + dr + loss == pytest.approx(1.0, abs=1e-9)
                assert min(w, dr, loss) >= 0.0
    # full-time state must equal the real result deterministically
    hs, as_ = d["home_score"], d["away_score"]
    exp = (
        (1.0, 0.0, 0.0)
        if hs > as_
        else (0.0, 1.0, 0.0) if hs == as_ else (0.0, 0.0, 1.0)
    )
    # once the expected added time is exhausted the match is decided
    assert inplay_wdl(pre, 90, hs, as_, extra=10) == exp


# --------------------------------------------------------------- live market odds


@pytest.mark.integration
@pytest.mark.odds
def test_shin_devig_on_live_market(live_odds):
    """De-vig every real 3-way market: must sum to 1 and strip the overround."""
    checked = 0
    for (home, away), (oh, od, oa) in live_odds.items():
        if not all(x and x > 1.0 for x in (oh, od, oa)):
            continue
        w, d, loss = oddsapi_to_wdl(oh, od, oa)
        assert w + d + loss == pytest.approx(1.0, abs=1e-9), (home, away)
        raw = 1 / oh + 1 / od + 1 / oa
        if raw > 1.0:  # a real book always has margin
            assert w <= 1 / oh + 1e-9
        assert all(map(math.isfinite, (w, d, loss)))
        checked += 1
    if checked == 0:
        pytest.skip("No well-formed 3-way markets in the live snapshot")


@pytest.mark.integration
@pytest.mark.odds
def test_ko_prob_valid_from_live_market(live_odds):
    for (_h, _a), (oh, od, oa) in live_odds.items():
        if not all(x and x > 1.0 for x in (oh, od, oa)):
            continue
        w, d, loss = oddsapi_to_wdl(oh, od, oa)
        a, b = ko_prob(w, d, loss)
        assert a + b == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= a <= 1.0
        return
    pytest.skip("No usable market for KO conversion")


# --------------------------------------------------------------- always-true corners


def test_poisson_pmf_normalises():
    assert sum(_poisson_pmf(1.3, k) for k in range(60)) == pytest.approx(1.0, abs=1e-9)


def test_sigmoid_style_bounds_and_ft_collapse():
    pre = (0.45, 0.27, 0.28)
    assert inplay_wdl(pre, 90, 2, 1, extra=10) == (1.0, 0.0, 0.0)
    dh, da = elo_deltas(pre, inplay_wdl(pre, 85, 2, 0))
    assert dh > 0 > da and -80.0 <= dh <= 80.0


def test_stoppage_time_is_not_full_time():
    """90'+0 and 90'+4 still have added time to play."""
    pre = (0.45, 0.27, 0.28)
    at90, at94, over = (inplay_wdl(pre, 90, 1, 1, extra=x) for x in (0, 4, 10))
    assert 0.0 < at94[0] < at90[0] and 0.0 < at94[2] < at90[2]
    assert over == (0.0, 1.0, 0.0)


# --------------------------------------------------------------- in-play calibration


@pytest.mark.parametrize("gap", [0, 50, 100, 200, 300, 400])
@pytest.mark.parametrize("flip", [False, True])
def test_inplay_reproduces_prior_at_kickoff(gap, flip):
    """0-0 at minute 0 carries no new information, so it must return the prior."""
    a, b = (1500.0, 1500.0 + gap) if flip else (1500.0 + gap, 1500.0)
    pre = elo_to_wdl(a, b)
    state = inplay_wdl(pre, 0, 0, 0)
    assert state == pytest.approx(pre, abs=0.02)
    dh, da = elo_deltas(pre, state)
    assert abs(dh) < 1.0 and abs(da) < 1.0


def test_inplay_lead_monotone_and_red_card_direction():
    pre = elo_to_wdl(1600.0, 1500.0)
    # a home lead gets more decisive as the clock runs down
    p_hw = [inplay_wdl(pre, m, 1, 0)[0] for m in (10, 30, 50, 70, 89)]
    assert p_hw == sorted(p_hw)
    # a home red card at 0-0 hurts home and helps away
    base = inplay_wdl(pre, 30, 0, 0)
    red = inplay_wdl(pre, 30, 0, 0, red_home=1)
    assert red[0] < base[0] and red[2] > base[2]


# --------------------------------------------------------------- Shin de-vig


@pytest.mark.parametrize(
    "odds", [(1.5, 4.2, 7.0), (2.5, 3.2, 3.0), (1.2, 6.5, 15.0), (1.9, 3.6, 4.5)]
)
def test_shin_sums_to_one_and_z_bounded(odds):
    probs, z = shin_devig(odds)
    overround = sum(1.0 / o for o in odds) - 1.0
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)  # before any normalisation
    assert 0.0 < z <= overround


def test_shin_favourite_longshot_bias():
    """Shin removes proportionally more margin from the longshot."""
    odds = (1.3, 5.5, 11.0)
    probs, _ = shin_devig(odds)
    raw = [1.0 / o for o in odds]
    fav_cut = 1.0 - probs[0] / raw[0]
    long_cut = 1.0 - probs[2] / raw[2]
    assert long_cut > fav_cut > 0.0
    # relative to proportional normalisation the favourite gains, longshot loses
    booksum = sum(raw)
    assert probs[0] > raw[0] / booksum
    assert probs[2] < raw[2] / booksum


def test_shin_equals_normalisation_without_overround():
    odds = (2.0, 4.0, 4.0)  # booksum exactly 1
    probs, z = shin_devig(odds)
    assert z == 0.0
    assert list(probs) == pytest.approx([0.5, 0.25, 0.25], abs=1e-12)
    assert oddsapi_to_wdl(*odds) == pytest.approx((0.5, 0.25, 0.25), abs=1e-12)


# --------------------------------------------------------------- anomaly eval


def test_false_alarm_rate_independent_of_seed_count():
    """A fixed per-day false-alert count gives the same FA/topic-day however
    many topic-days are pooled."""
    day = {"detected": [], "latency": [], "alerts": 4, "true_alerts": 1, "false_by_type": {}}
    rates = [aggregate([day] * n)["false_alerts_per_topic_day"] for n in (1, 5, 10, 50)]
    assert rates == pytest.approx([3.0] * 4)
