"""
Offline tests for the WC 2026 data layer.

No network: ESPN parsing runs on trimmed fixtures in tests/fixtures,
everything else on the committed data/wc2026 snapshot.

    python -m pytest tests/test_wc2026_data.py -v
"""

from __future__ import annotations

import asyncio
import json
from itertools import combinations
from pathlib import Path

import pytest

from api.match_timeline import around, at_minute
from api.schemas.schema import MatchEvent, MatchState
from api.tournament_state import red_counts
from feeds import espn
from ml import shot_xg
from ml.tournament_sim import results_from_fixtures, run_simulation
from ml.wc2026_format import KO_TREE, R32_ORDER, R32_SLOTS, third_slot_of
from ml.wc_2026_config import (
    FIXTURES,
    GROUP_FIXTURES,
    GROUPS,
    KO_FIXTURE_BY_MATCH_NO,
    THIRD_PLACE_TABLE,
    WC2026_TEAMS,
)

FIX = Path(__file__).parent / "fixtures"


def _events():
    return json.loads((FIX / "espn_scoreboard_events.json").read_text(encoding="utf-8"))


def _summary():
    return json.loads((FIX / "espn_summary_760513.json").read_text(encoding="utf-8"))


# ── Snapshot integrity ───────────────────────────────────────────────────────


def test_snapshot_is_the_real_field():
    assert len(WC2026_TEAMS) == 48
    assert len(GROUPS) == 12 and all(len(g) == 4 for g in GROUPS.values())
    assert len(FIXTURES) == 104 and len(GROUP_FIXTURES) == 72
    assert sorted(KO_FIXTURE_BY_MATCH_NO) == list(range(73, 105))
    names = {t.name for t in WC2026_TEAMS}
    assert {"Spain", "Argentina", "Curaçao", "Cape Verde", "Jordan"} <= names
    assert all(1300 < t.elo < 2400 for t in WC2026_TEAMS)


def test_group_fixtures_are_round_robins():
    by_group: dict[str, set] = {}
    for f in GROUP_FIXTURES:
        by_group.setdefault(f["group"], set()).add(frozenset((f["home_name"], f["away_name"])))
    for g, pairs in by_group.items():
        teams = {t.name for t in GROUPS[g]}
        assert pairs == {frozenset(p) for p in combinations(teams, 2)}


def test_annex_c_complete_and_eligible():
    assert len(THIRD_PLACE_TABLE) == 495
    for combo, row in THIRD_PLACE_TABLE.items():
        assert sorted(row.values()) == list(combo)
        for winner, g in row.items():
            assert g in third_slot_of(winner)[1]


def test_bracket_order_reproduces_tree():
    ms = list(R32_ORDER)
    assert sorted(ms) == sorted(R32_SLOTS)
    parent = {frozenset(ch): m for m, ch in KO_TREE.items()}
    while len(ms) > 1:
        ms = [parent[frozenset(ms[i : i + 2])] for i in range(0, len(ms), 2)]
    assert ms == [104]


# ── ESPN parsing ─────────────────────────────────────────────────────────────


def test_parse_event_real_fields():
    f = next(espn.parse_event(e) for e in _events() if e["id"] == "760513")
    assert f["stage"] == "qf" and f["status"] == "AET"
    assert (f["home_name"], f["away_name"]) == ("Argentina", "Switzerland")
    assert (f["home_score"], f["away_score"], f["winner"]) == (3, 1, "Argentina")
    assert f["neutral"] and f["host_side"] is None


def test_parse_status_stoppage_and_extra_time():
    assert espn.parse_status(
        {"displayClock": "90'+4'", "type": {"name": "STATUS_SECOND_HALF"}}
    ) == ("2H", 90, 4)
    assert espn.parse_status({"type": {"name": "STATUS_FINAL_PEN"}, "displayClock": "120'"})[0] == "PEN"
    assert espn.parse_status({"type": {"name": "STATUS_SCHEDULED"}}) == ("NS", None, None)


def test_parse_summary_real_stats_events_lineups():
    f = next(espn.parse_event(e) for e in _events() if e["id"] == "760513")
    d = espn.parse_summary(_summary(), f)
    assert d["home_stats"].possession == 59 and d["home_stats"].shots_total == 22
    assert d["home_stats"].expected_goals > d["away_stats"].expected_goals > 0
    goals = [e for e in d["events"] if e.type == "goal"]
    assert goals[0].player_name == "Alexis Mac Allister" and goals[0].elapsed == 10
    assert all(e.source == "espn" for e in d["events"])
    xi = d["lineups"]["home"]["startingXI"]
    assert len(xi) == 11 and xi[0]["line"] == "G"
    assert all(s["side"] in (1, 2) for s in d["shots"])


def test_own_goal_credits_conceding_side():
    f = {"home_espn_id": "1", "away_espn_id": "2", "home_name": "H", "away_name": "A"}
    ke = [{"type": {"text": "Own Goal"}, "team": {"id": "1"}, "clock": {"displayValue": "30'"},
           "participants": [{"athlete": {"displayName": "X"}}]}]
    ev = espn._events(ke, {"1": (1, "H"), "2": (2, "A")})[0]
    assert ev.type == "own_goal" and ev.team_id == 2  # scored by A's player, benefits H
    assert f  # parse_summary contract uses the same id map


def test_shot_xg_both_coordinate_conventions_agree():
    base = dict(minute=10, extra=None, period=1, team_id="", team_name="", player=None,
                header=False, penalty=False, direct_fk=False, cross=False,
                through_ball=False, corner=False, set_piece=False, fast_break=False,
                outcome="missed")
    new = shot_xg.Shot(x=100 - 11 / 1.05, y=50.0, **base)  # 11 m, central
    old = shot_xg.Shot(x=11 / shot_xg.OLD_X_SCALE, y=0.5, **base)
    assert abs(shot_xg.xg(new) - shot_xg.xg(old)) < 1e-6
    assert 0.05 < shot_xg.xg(new) < 0.5
    assert shot_xg.xg(shot_xg.Shot(x=50.0, y=50.0, **base)) < shot_xg.xg(new)


# ── Only real (ESPN) red cards condition models ──────────────────────────────


def _state(events, **kw) -> MatchState:
    return MatchState(fixture_id=1, home_name="H", away_name="A", events=events, **kw)


def test_red_counts_ignore_non_feed_events():
    evs = [
        MatchEvent(elapsed=20, team_id=1, team_name="H", type="red", source="espn"),
        MatchEvent(elapsed=30, team_id=2, team_name="A", type="red", source="synthesised"),
    ]
    assert red_counts(_state(evs)) == (1, 0)


# ── State as of the event, not the final score ───────────────────────────────


def test_around_uses_running_score():
    g1 = MatchEvent(elapsed=10, team_id=1, team_name="H", type="goal")
    evs = [
        g1,
        MatchEvent(elapsed=50, team_id=1, team_name="H", type="goal"),
        MatchEvent(elapsed=70, team_id=1, team_name="H", type="goal"),
    ]
    s = _state(evs, home_score=3, away_score=0, status_short="FT")
    before, after = around(s, g1)
    assert before[:2] == (0, 0) and after[:2] == (1, 0)


def test_around_orders_stoppage_time_and_same_minute():
    a = MatchEvent(elapsed=90, extra=3, team_id=2, team_name="A", type="goal")
    b = MatchEvent(elapsed=90, extra=1, team_id=1, team_name="H", type="goal")
    c = MatchEvent(elapsed=90, extra=1, team_id=1, team_name="H", type="goal")
    s = _state([a, b, c])
    assert around(s, a)[0][:2] == (2, 0)
    assert around(s, c)[0][:2] == (1, 0)
    assert at_minute(s, 90, 1)[:2] == (2, 0)


# ── Simulator on the real tournament ─────────────────────────────────────────


def _sums(res):
    return {s: round(sum(t.probs[s] for t in res.teams), 6) for s in ("r32", "r16", "qf", "sf", "final", "champion")}


def test_stage_probabilities_sum_correctly():
    assert _sums(run_simulation(n_sims=4000, seed=1)) == {
        "r32": 32, "r16": 16, "qf": 8, "sf": 4, "final": 2, "champion": 1
    }


def test_champion_is_certain_after_the_final():
    res = run_simulation(n_sims=500, seed=1, results=results_from_fixtures())
    champ = [t.name for t in res.teams if t.probs["champion"] == 1.0]
    assert champ == [next(f["winner"] for f in FIXTURES if f["stage"] == "final")]
    assert all(t.probs["champion"] in (0.0, 1.0) for t in res.teams)


def test_pinned_group_stage_reproduces_real_round_of_32():
    last_group = max(f["date"] for f in GROUP_FIXTURES)
    res = run_simulation(n_sims=1000, seed=2, results=results_from_fixtures(before=last_group + "~"))
    real = {t for f in FIXTURES if f["stage"] == "r32" for t in (f["home_name"], f["away_name"])}
    assert {t.name for t in res.teams if t.probs["r32"] == 1.0} == real
    assert all(t.probs["r32"] == 0.0 for t in res.teams if t.name not in real)


def test_common_random_numbers_are_deterministic():
    a = run_simulation(n_sims=2000, seed=9)
    b = run_simulation(n_sims=2000, seed=9)
    assert [t.probs for t in a.teams] == [t.probs for t in b.teams]


# ── A not-started fixture persists once, with no detail fetch ────────────────


class _FakeRedis:
    def __init__(self):
        self.kv, self.sets, self.published = {}, {}, []

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = v

    async def setex(self, k, ttl, v):
        self.kv[k] = v

    async def smembers(self, k):
        return set(self.sets.get(k, set()))

    async def srem(self, k, v):
        self.sets.setdefault(k, set()).discard(v)

    async def publish(self, ch, msg):
        self.published.append((ch, msg))

    def pipeline(self, transaction=True):
        outer = self

        class P:
            def __init__(self):
                self.ops = []

            def setex(self, *a):
                self.ops.append(outer.setex(*a))

            def sadd(self, k, v):
                outer.sets.setdefault(k, set()).add(v)

            def srem(self, k, v):
                outer.sets.setdefault(k, set()).discard(v)

            async def execute(self):
                for op in self.ops:
                    await op

        return P()


def test_ns_fixture_publishes_once_and_fetches_nothing():
    from api.workers import match_producer as mp

    ns = {**espn.parse_event(_events()[0]), "status": "NS", "home_score": None, "away_score": None}
    r = _FakeRedis()
    p = mp.Producer(r)
    calls = []

    async def fixtures(_client):
        return [ns]

    async def summary(*_a, **_k):
        calls.append(1)
        raise AssertionError("NS fixture must not fetch a summary")

    p.fixtures = fixtures
    orig = espn.fetch_summary
    espn.fetch_summary = summary
    try:
        asyncio.run(p.tick(None))
        asyncio.run(p.tick(None))
    finally:
        espn.fetch_summary = orig
    assert calls == []
    assert [c for c, _ in r.published] == ["match_update"]
    st = MatchState.model_validate_json(r.kv[f"match:{ns['fixture_id']}:state"])
    assert st.status_short == "NS" and st.stats_source == "unavailable"


@pytest.mark.parametrize("fid", [f["fixture_id"] for f in FIXTURES[:3]])
def test_snapshot_match_detail_loads(fid):
    from feeds import snapshot

    d = snapshot.match_detail(fid)
    assert d and d["home_stats"].shots_total > 0 and d["lineups"]
