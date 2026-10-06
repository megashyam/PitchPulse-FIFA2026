"""
WC 2026 Monte Carlo tournament simulator.

Uses the real field, fixtures and bracket, vectorised across all N
simulations.

Group stage:
    Each of the 72 fixtures (true home/away; host nations get home
    advantage in their own country) gets 90' Dixon-Coles rates fitted to
    its prior (ml.prior_builder.match_wdl, plus match_totals when quoted).
    Scores are drawn by inverse CDF from one uniform per (sim, match, side):
    home goals from their Poisson marginal, away goals from the Dixon-Coles
    conditional, so runs sharing a seed use common random numbers. Played
    matches are pinned to their real score; a live match adds goals for the
    time left.
    Standings (FIFA 2026): points → head-to-head points, goal difference and
    goals among teams level on points → goal difference → goals for → Elo
    rank (proxy for the FIFA ranking) → random. Head-to-head is one pass;
    conduct isn't modelled. Fully played groups take FIFA's official order.
    Thirds: points → goal difference → goals for → Elo rank; once every
    group is played the official best eight qualify.

Round of 32:
    Winners, runners-up and the best eight thirds fill FIFA's slots; thirds
    are placed by the Annex C row for that combination of groups.

Knockouts:
    FIFA bracket order (ml.wc2026_format.R32_ORDER). Advance probability
    includes strength-aware extra time (ml.prior_builder.ko_advance_batch).
    Played ties are pinned to the real winner; a live tie uses its in-play
    distribution.

Output:
    Per-team stage probabilities with Wilson 95% intervals.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.in_play import (
    FULL_TIME,
    _wdl_from_rates,
    apply_reds,
    default_rho,
    fit_match_rates,
    remaining_fraction,
    inplay_wdl,
    rates_from_wdl_batch,
)
from ml.metrics import wilson
from ml.prior_builder import build_ko_matrix, match_totals, match_wdl
from ml.wc2026_format import KO_TREE, R32_ORDER, R32_SLOTS, THIRD_SLOT_WINNERS
from ml.wc_2026_config import (
    FINAL_STANDINGS,
    FIXTURES,
    GROUP_FIXTURES,
    GROUPS,
    KO_FIXTURE_BY_MATCH_NO,
    THIRD_PLACE_TABLE,
    WC2026_TEAMS,
)

from config import N_SIMS

logger = logging.getLogger(__name__)

STAGES = ["group_exit", "r32", "r16", "qf", "sf", "final", "champion"]
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}
COMPLETED = {"FT", "AET", "PEN"}
_MAX_GOALS = 15


@dataclass
class LiveMatch:
    minute: int
    home_score: int
    away_score: int
    red_home: int = 0
    red_away: int = 0
    extra: Optional[int] = None


@dataclass
class TeamResult:
    name: str
    group: str
    elo: float
    fifa_rank: int
    probs: Dict[str, float]  # stage → probability
    ci_95: Dict[str, Tuple[float, float]]  # stage → Wilson (lo, hi)


@dataclass
class SimResult:
    n_sims: int
    elapsed_s: float
    teams: List[TeamResult]
    champion_hits: Optional[np.ndarray] = None  # (N, n_teams) bool, with keep_hits

    def team(self, name: str) -> Optional[TeamResult]:
        return next((t for t in self.teams if t.name == name), None)


def results_from_fixtures(before: Optional[str] = None) -> Dict[int, dict]:
    """Completed results from the snapshot, optionally only before ISO `before`."""
    return {
        f["fixture_id"]: {
            "home_score": f["home_score"],
            "away_score": f["away_score"],
            "winner": f["winner"],
        }
        for f in FIXTURES
        if f["status"] in COMPLETED and (before is None or f["date"] < before)
    }


def _poisson_ppf(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Inverse Poisson CDF per column: u (N, M) uniforms, lam (M,) rates."""
    k = np.arange(_MAX_GOALS + 1)
    lg = np.array([math.lgamma(i + 1) for i in k])
    lam = np.maximum(lam, 0.0)[:, None]
    with np.errstate(divide="ignore"):
        logp = np.where(lam > 0, -lam + k * np.log(np.where(lam > 0, lam, 1.0)) - lg, np.where(k == 0, 0.0, -np.inf))
    cdf = np.cumsum(np.exp(logp), axis=1)
    cdf[:, -1] = 1.0
    out = np.empty(u.shape, dtype=np.int16)
    for m in range(u.shape[1]):
        out[:, m] = np.searchsorted(cdf[m], u[:, m], side="right")
    return out


def _dc_away(
    hg: np.ndarray, ag: np.ndarray, u: np.ndarray, lam: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """Away goals from the Dixon-Coles conditional given home goals ``hg``.
    ``ag`` is the Poisson inverse CDF of the same uniforms ``u`` (N, M); lam
    (M, 2) rates, rho (M,). The conditional differs only in how it splits
    P(away <= 1) between 0 and 1 when home scored 0 or 1, so only those draws
    change."""
    lh, la = lam[:, 0], lam[:, 1]
    p0 = np.exp(-la)
    c0 = np.where(hg == 0, p0 * (1 - lh * la * rho), p0 * (1 + la * rho))
    return np.where((hg <= 1) & (ag <= 1), (u >= c0).astype(ag.dtype), ag)


def _remaining_rates(rates: np.ndarray, live: LiveMatch) -> np.ndarray:
    frac = remaining_fraction(live.minute, live.extra)
    lh, la = rates * frac
    return np.array(apply_reds(lh, la, live.red_home, live.red_away))


def _third_lut() -> np.ndarray:
    """(4096, 8) lookup: qualifying-thirds bitmask → third's group per slot.

    Columns follow THIRD_SLOT_WINNERS; -1 marks an impossible mask.
    """
    lut = np.full((1 << 12, 8), -1, dtype=np.int8)
    letters = sorted(GROUPS)
    for combo, row in THIRD_PLACE_TABLE.items():
        mask = sum(1 << letters.index(g) for g in combo)
        lut[mask] = [letters.index(row[w]) for w in THIRD_SLOT_WINNERS]
    return lut


_PARENT = {frozenset(ch): m for m, ch in KO_TREE.items()}
_THIRD_LUT = _third_lut()


def _prior(f: dict, odds_table, ovr: Dict[str, float]):
    """(W/D/L, market totals or None) for a real fixture."""
    wdl = match_wdl(
        f["home_name"], f["away_name"], host_side=f.get("host_side"),
        odds_table=odds_table, elo_overrides=ovr,
    )
    return wdl, match_totals(f["home_name"], f["away_name"], odds_table=odds_table, elo_overrides=ovr)


class TournamentSimulator:
    def __init__(
        self,
        odds_table=None,
        n_sims: int = N_SIMS,
        seed: Optional[int] = None,
        elo_overrides: Optional[Dict[str, float]] = None,
        results: Optional[Dict[int, dict]] = None,
        live: Optional[Dict[int, LiveMatch]] = None,
        keep_hits: bool = False,
    ):
        self.teams = WC2026_TEAMS
        self.n_teams = len(self.teams)
        self.name_to_idx = {t.name: i for i, t in enumerate(self.teams)}
        self.n_sims = n_sims
        self.seed = seed
        self.elo_overrides = elo_overrides or {}
        self.results = results or {}
        self.live = live or {}
        self.keep_hits = keep_hits
        self.group_letters = sorted(GROUPS)
        self.group_idx = np.array(
            [[self.name_to_idx[t.name] for t in GROUPS[g]] for g in self.group_letters],
            dtype=np.int64,
        )  # (12, 4)
        # Final tiebreak: FIFA ranking (the field's Elo rank stands in), as a key in [0, 1)
        n = self.n_teams
        self.rank_key = np.array([(n - t.fifa_rank) / (n + 1) for t in self.teams])
        self._init_group(odds_table)
        self._init_official()
        self._init_ko(odds_table)

    def _init_group(self, odds_table) -> None:
        """Per-fixture 90' rates and rho, played/live score bases, and the
        home/away incidence matrices H, A (fixture × team)."""
        gf = GROUP_FIXTURES
        self.g_home = np.array([self.name_to_idx[f["home_name"]] for f in gf])
        self.g_away = np.array([self.name_to_idx[f["away_name"]] for f in gf])
        priors = [_prior(f, odds_table, self.elo_overrides) for f in gf]
        self.g_wdl = np.array([wdl for wdl, _ in priors])
        rates = rates_from_wdl_batch(self.g_wdl)  # (72, 2)
        self.g_rho = np.full(len(gf), default_rho())
        for m, (_, totals) in enumerate(priors):
            if totals is not None:
                lh, la, self.g_rho[m] = fit_match_rates(tuple(self.g_wdl[m]), totals)
                rates[m] = (lh, la)
        self.g_base = np.zeros((len(gf), 2), dtype=np.int16)
        self.g_rates = rates.copy()
        for m, f in enumerate(gf):
            fid = f["fixture_id"]
            if fid in self.results:
                r = self.results[fid]
                self.g_base[m] = (r["home_score"], r["away_score"])
                self.g_rates[m] = 0.0
            elif fid in self.live:
                lv = self.live[fid]
                self.g_base[m] = (lv.home_score, lv.away_score)
                self.g_rates[m] = _remaining_rates(rates[m], lv)

        self.H = np.zeros((len(gf), self.n_teams), dtype=np.float32)
        self.A = np.zeros((len(gf), self.n_teams), dtype=np.float32)
        self.H[np.arange(len(gf)), self.g_home] = 1.0
        self.A[np.arange(len(gf)), self.g_away] = 1.0

    def _init_official(self) -> None:
        """Official order for fully played groups / thirds (dominates comp / tkey)."""
        gf = GROUP_FIXTURES
        self.official = np.zeros(self.n_teams)
        self.official_third = np.zeros(self.n_teams)
        played = {f["fixture_id"] for f in gf if f["fixture_id"] in self.results}
        by_group: Dict[str, set] = {}
        for f in gf:
            by_group.setdefault(f["group"], set()).add(f["fixture_id"])
        for g, fids in by_group.items():
            if fids <= played:
                for row in FINAL_STANDINGS[g]:
                    self.official[self.name_to_idx[row["team"]]] += (5 - row["rank"]) * 1e11
        if len(played) == len(gf):
            r32 = {t for f in FIXTURES if f["stage"] == "r32" for t in (f["home_name"], f["away_name"])}
            for rows in FINAL_STANDINGS.values():
                third = rows[2]["team"]
                if third in r32:
                    self.official_third[self.name_to_idx[third]] += 1e10

    def _init_ko(self, odds_table) -> None:
        """Knockout advance matrix, plus pins / live ties keyed by FIFA match number."""
        self.ko_matrix = build_ko_matrix(odds_table, self.elo_overrides)
        self.ko_pinned: Dict[int, int] = {}
        self.ko_live: Dict[int, Tuple[int, int, float]] = {}
        for m, f in KO_FIXTURE_BY_MATCH_NO.items():
            fid = f["fixture_id"]
            if fid in self.results and self.results[fid].get("winner"):
                self.ko_pinned[m] = self.name_to_idx[self.results[fid]["winner"]]
            elif fid in self.live:
                h, a = self.name_to_idx[f["home_name"]], self.name_to_idx[f["away_name"]]
                pre, totals = _prior(f, odds_table, self.elo_overrides)
                self.ko_live[m] = (h, a, live_ko_advance(pre, self.live[fid], totals))

    def run(self) -> SimResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        N, n = self.n_sims, self.n_teams
        M = len(self.g_home)

        u = rng.random((N, M, 2), dtype=np.float32)
        hg = _poisson_ppf(u[:, :, 0], self.g_rates[:, 0])
        ag = _poisson_ppf(u[:, :, 1], self.g_rates[:, 1])
        ag = _dc_away(hg, ag, u[:, :, 1], self.g_rates, self.g_rho)
        hg, ag = self.g_base[:, 0] + hg, self.g_base[:, 1] + ag
        del u
        hgf, agf = hg.astype(np.float32), ag.astype(np.float32)
        hp = np.where(hg > ag, 3.0, np.where(hg == ag, 1.0, 0.0)).astype(np.float32)
        ap = np.where(ag > hg, 3.0, np.where(hg == ag, 1.0, 0.0)).astype(np.float32)
        pts = hp @ self.H + ap @ self.A
        gf = hgf @ self.H + agf @ self.A
        ga = agf @ self.H + hgf @ self.A
        # head-to-head among teams level on points: a match counts when its two teams are level
        tie = (pts @ self.H.T == pts @ self.A.T).astype(np.float32)  # matmul gathers, faster than fancy indexing
        gd_m = (hgf - agf) * tie
        h2h_pts = (hp * tie) @ self.H + (ap * tie) @ self.A
        h2h_gd = gd_m @ self.H - gd_m @ self.A
        h2h_gf = (hgf * tie) @ self.H + (agf * tie) @ self.A
        # mixed-radix keys, exact in float64 (goal tallies < 64); the tail < 1 is
        # the ranking step, then random
        tail = self.rank_key + rng.random((N, n), dtype=np.float32) / (n + 1)
        overall = (gf - ga + 64.0) * 64 + gf
        comp = (((pts.astype(np.float64) * 10 + h2h_pts) * 128 + h2h_gd + 64) * 64 + h2h_gf) * 8192 + overall
        comp += tail + self.official
        tkey = pts * 8192.0 + overall + tail + self.official_third

        gcomp = comp[:, self.group_idx]  # (N, 12, 4)
        order = np.argsort(-gcomp, axis=2)
        ranked = self.group_idx[np.arange(12)[None, :, None], order]  # (N, 12, 4)
        winners, runners, thirds, fourths = (ranked[:, :, k] for k in range(4))

        tcomp = np.take_along_axis(tkey[:, self.group_idx], order[:, :, 2:3], axis=2)[:, :, 0]  # (N, 12)
        best8 = np.argsort(-tcomp, axis=1)[:, :8]
        mask = (1 << best8.astype(np.int64)).sum(axis=1)
        third_groups = _THIRD_LUT[mask]  # (N, 8)
        third_by_winner = np.take_along_axis(thirds, third_groups.astype(np.int64), axis=1)

        def slot(code: str, winner_code: str) -> np.ndarray:
            g = self.group_letters.index(code[-1]) if not code.startswith("3:") else None
            if code.startswith("1"):
                return winners[:, g]
            if code.startswith("2"):
                return runners[:, g]
            return third_by_winner[:, THIRD_SLOT_WINNERS.index(winner_code)]

        reach = np.zeros((N, n), dtype=np.int8)
        rows = np.arange(N)
        r32_teams = np.concatenate([winners, runners, np.take_along_axis(thirds, best8, axis=1)], axis=1)
        reach[rows[:, None], r32_teams] = STAGE_IDX["r32"]

        home = np.stack([slot(R32_SLOTS[m][0], R32_SLOTS[m][0]) for m in R32_ORDER], axis=1)
        away = np.stack([slot(R32_SLOTS[m][1], R32_SLOTS[m][0]) for m in R32_ORDER], axis=1)
        matches = list(R32_ORDER)

        for stage in ("r16", "qf", "sf", "final", "champion"):
            w = self._ko_round(rng, matches, home, away)
            reach[rows[:, None], w] = np.maximum(reach[rows[:, None], w], STAGE_IDX[stage])
            if w.shape[1] == 1:
                break
            matches = [_PARENT[frozenset(matches[i : i + 2])] for i in range(0, len(matches), 2)]
            home, away = w[:, 0::2], w[:, 1::2]

        teams_out = self._team_results(reach, (winners, runners, thirds, fourths))
        elapsed = time.perf_counter() - t0
        logger.info(f"Simulation complete: {N} runs in {elapsed:.2f}s")
        return SimResult(
            n_sims=N,
            elapsed_s=round(elapsed, 3),
            teams=teams_out,
            champion_hits=reach == STAGE_IDX["champion"] if self.keep_hits else None,
        )

    def _team_results(self, reach: np.ndarray, positions) -> List[TeamResult]:
        """Per-team stage and group-finish probabilities with Wilson CIs."""
        N, n = reach.shape
        pos_counts = [np.bincount(x.ravel(), minlength=n) for x in positions]
        out: List[TeamResult] = []
        for t_idx, team in enumerate(self.teams):
            tr = reach[:, t_idx]
            counts = {
                s: int((tr == 0).sum()) if s == "group_exit" else int((tr >= i).sum())
                for s, i in STAGE_IDX.items()
            }
            for key, arr in zip(("group_first", "group_second", "group_third", "group_fourth"), pos_counts):
                counts[key] = int(arr[t_idx])
            out.append(
                TeamResult(
                    name=team.name,
                    group=team.group,
                    elo=team.elo,
                    fifa_rank=team.fifa_rank,
                    probs={k: round(c / N, 6) for k, c in counts.items()},
                    ci_95={k: tuple(round(x, 6) for x in wilson(c, N)) for k, c in counts.items()},
                )
            )
        return out

    def _ko_round(
        self, rng: np.random.Generator, matches: List[int], home: np.ndarray, away: np.ndarray
    ) -> np.ndarray:
        p = self.ko_matrix[home, away]
        for j, m in enumerate(matches):
            if m in self.ko_live:
                h, a, ph = self.ko_live[m]
                col = p[:, j]
                col[(home[:, j] == h) & (away[:, j] == a)] = ph
                col[(home[:, j] == a) & (away[:, j] == h)] = 1.0 - ph
        u = rng.random(home.shape, dtype=np.float32)
        w = np.where(u < p, home, away)
        for j, m in enumerate(matches):
            if m in self.ko_pinned:
                w[:, j] = self.ko_pinned[m]
        return w


def live_ko_advance(pre_wdl, live: LiveMatch, totals=None) -> float:
    """P(home advances) for a knockout tie in progress; ``totals`` is the
    market's (P(over), line) when quoted."""
    if live.minute <= FULL_TIME:
        w, d, _ = inplay_wdl(
            pre_wdl, live.minute, live.home_score, live.away_score,
            live.red_home, live.red_away, extra=live.extra, totals=totals,
        )
        return w + d * _et_advance(pre_wdl, 0, live, totals)
    return _et_advance(pre_wdl, live.minute - FULL_TIME, live, totals)


def _et_advance(pre_wdl, et_minute: int, live: LiveMatch, totals=None) -> float:
    """P(home advances) from level (or current ET score) with `et_minute` of
    30' extra time played, then a 50/50 shootout."""
    lh, la, rho = fit_match_rates(tuple(pre_wdl), totals)
    frac = max(0.0, (30 - et_minute) / 30) / 3.0
    lead = live.home_score - live.away_score if et_minute > 0 else 0
    if frac <= 0:
        return 1.0 if lead > 0 else 0.0 if lead < 0 else 0.5
    w, d, _ = _wdl_from_rates(lh * frac, la * frac, lead, rho)
    return float(w + 0.5 * d)


def run_simulation(
    odds_table=None,
    n_sims: int = N_SIMS,
    seed: Optional[int] = None,
    elo_overrides: Optional[Dict[str, float]] = None,
    results: Optional[Dict[int, dict]] = None,
    live: Optional[Dict[int, LiveMatch]] = None,
    keep_hits: bool = False,
) -> SimResult:
    """Build and run the simulation.

    `results` pins played fixtures (fixture_id → {home_score, away_score,
    winner}); `live` conditions in-progress ones (fixture_id → LiveMatch).
    `keep_hits` returns per-sim champion indicators for paired CRN SEs.
    """
    return TournamentSimulator(
        odds_table=odds_table,
        n_sims=n_sims,
        seed=seed,
        elo_overrides=elo_overrides,
        results=results,
        live=live,
        keep_hits=keep_hits,
    ).run()
