"""
In-play match outcome model and Elo adjustment mapping.

Conditions the tournament simulation on live match state for the
counterfactual agent.

Pipeline:
    1. Fit full-match Poisson scoring rates to the pre-match W/D/L prior.
       Remaining goals are two Poisson streams at those rates scaled by the
       time left, giving an in-play W/D/L that depends on the current score,
       minutes remaining and red cards (and equals the prior at 0-0 kickoff).
    2. Map each team's shift in expected match points to a bounded Elo
       adjustment.
    3. The simulator runs before/after brackets with those ``elo_overrides``
       and common random numbers, so the difference isolates the event.

Pure math, no project imports.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

WDL = Tuple[float, float, float]  # (p_home_win, p_draw, p_away_win)

FULL_TIME = 90
# Mean added time at WC 2026 (104 matches, ESPN clocks): 5.7' first half,
# 8.0' second half. Fitted rates cover a whole match, stoppage included.
STOPPAGE_1H = 5.7
STOPPAGE_2H = 8.0
MATCH_MINUTES = FULL_TIME + STOPPAGE_1H + STOPPAGE_2H
BASE_TOTAL_GOALS = 2.6  # avg combined goals per 90' in men's international football
TOTAL_GOALS_BOUNDS = (0.2 * BASE_TOTAL_GOALS, 3.0 * BASE_TOTAL_GOALS)  # rate-fit search range
HOME_SHARE_BOUNDS = (0.02, 0.98)  # rate-fit search range for the home share of goals
RED_SELF = 0.72  # a sending-off cuts that team's remaining scoring ~28%
RED_OPP = 1.12  # ...and lifts the opponent's ~12%
MAX_GOALS = 12  # Poisson enumeration cutoff (truncated tail renormalised)


def _poisson_pmf(lmbda: float, k: int) -> float:
    return math.exp(-lmbda) * lmbda**k / math.factorial(k)


def _wdl_from_rates(lam_h, lam_a, lead: int = 0) -> np.ndarray:
    """
    Vectorised W/D/L for final margin ``lead + Poisson(lam_h) - Poisson(lam_a)``.

    ``lam_h`` / ``lam_a`` may be scalars or equal-shape arrays; returns an array
    of shape ``(..., 3)`` with rows summing to 1 (truncated tail renormalised).
    """
    lam_h = np.asarray(lam_h, dtype=np.float64)[..., None]
    lam_a = np.asarray(lam_a, dtype=np.float64)[..., None]
    k = np.arange(MAX_GOALS + 1)
    log_fact = np.array([math.lgamma(i + 1) for i in k])
    ph = np.exp(-lam_h + k * np.log(lam_h) - log_fact)
    pa = np.exp(-lam_a + k * np.log(lam_a) - log_fact)
    joint = ph[..., :, None] * pa[..., None, :]  # [..., gh, ga]
    margin = lead + k[:, None] - k[None, :]
    p_hw = (joint * (margin > 0)).sum(axis=(-2, -1))
    p_d = (joint * (margin == 0)).sum(axis=(-2, -1))
    p_aw = (joint * (margin < 0)).sum(axis=(-2, -1))
    out = np.stack([p_hw, p_d, p_aw], axis=-1)
    return out / out.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=4096)
def _fit_rates_cached(pre_wdl: WDL) -> Tuple[float, float]:
    target = np.asarray(pre_wdl, dtype=np.float64)
    target = target / target.sum()

    def sse(log_total, share):
        total = np.exp(log_total)
        wdl = _wdl_from_rates(total * share, total * (1.0 - share))
        return ((wdl - target) ** 2).sum(axis=-1)

    lo_t, hi_t = (math.log(b) for b in TOTAL_GOALS_BOUNDS)
    lo_s, hi_s = HOME_SHARE_BOUNDS

    # 1) coarse vectorised grid over (log total goals, home share)
    grid_t, grid_s = np.meshgrid(
        np.linspace(lo_t, hi_t, 40), np.linspace(lo_s, hi_s, 49), indexing="ij"
    )
    err = sse(grid_t, grid_s)
    i, j = np.unravel_index(np.argmin(err), err.shape)
    best_t, best_s, best_err = grid_t[i, j], grid_s[i, j], err[i, j]

    # 2) local refinement: shrinking 5x5 pattern search around the best point
    step_t = (hi_t - lo_t) / 39
    step_s = (hi_s - lo_s) / 48
    offsets = np.linspace(-1.0, 1.0, 5)
    for _ in range(25):
        cand_t = np.clip(best_t + offsets[:, None] * step_t, lo_t, hi_t)
        cand_s = np.clip(best_s + offsets[None, :] * step_s, lo_s, hi_s)
        cand_t, cand_s = np.broadcast_arrays(cand_t, cand_s)
        err = sse(cand_t, cand_s)
        i, j = np.unravel_index(np.argmin(err), err.shape)
        if err[i, j] <= best_err:
            best_t, best_s, best_err = cand_t[i, j], cand_s[i, j], err[i, j]
        step_t *= 0.5
        step_s *= 0.5

    total = math.exp(best_t)
    return float(total * best_s), float(total * (1.0 - best_s))


def fit_match_rates(pre_wdl: WDL) -> Tuple[float, float]:
    """Full-match Poisson rates ``(lam_home, lam_away)`` fitting ``pre_wdl``.

    Least squares; exact for any realistic prior. Cached on the prior
    rounded to 3 decimals.
    """
    key = tuple(round(float(p), 3) for p in pre_wdl)
    return _fit_rates_cached(key)  # type: ignore[arg-type]


_GRID: tuple[np.ndarray, np.ndarray] | None = None


def _rate_grid() -> tuple[np.ndarray, np.ndarray]:
    """(rates (G,2), wdl (G,3)) over a dense (total goals × home share) grid."""
    global _GRID
    if _GRID is None:
        t, s = np.meshgrid(
            np.exp(np.linspace(*(math.log(b) for b in TOTAL_GOALS_BOUNDS), 90)),
            np.linspace(*HOME_SHARE_BOUNDS, 241),
            indexing="ij",
        )
        rates = np.stack([(t * s).ravel(), (t * (1 - s)).ravel()], axis=1)
        _GRID = (rates, _wdl_from_rates(rates[:, 0], rates[:, 1]))
    return _GRID


def rates_from_wdl_batch(wdl: np.ndarray) -> np.ndarray:
    """Vectorised fit_match_rates: (P,3) W/D/L → (P,2) rates via a dense grid."""
    rates, grid_wdl = _rate_grid()
    q = np.asarray(wdl, dtype=np.float64)[:, [0, 1]]
    g = grid_wdl[:, [0, 1]]
    out = np.empty((len(q), 2))
    for i in range(0, len(q), 256):
        d = ((q[i : i + 256, None, :] - g[None, :, :]) ** 2).sum(-1)
        out[i : i + 256] = rates[d.argmin(1)]
    return out


def ko_advance_batch(wdl: np.ndarray) -> np.ndarray:
    """P(home advances): 90' W/D/L, 30' ET at a third of the rates, 50/50 pens."""
    wdl = np.asarray(wdl, dtype=np.float64)
    r = rates_from_wdl_batch(wdl)
    et = _wdl_from_rates(r[:, 0] / 3.0, r[:, 1] / 3.0)
    return wdl[:, 0] + wdl[:, 1] * (et[:, 0] + 0.5 * et[:, 1])


def remaining_fraction(minute: int, extra: int | None = None) -> float:
    """Share of real playing time still to come, including expected added time."""
    extra = extra or 0
    if minute <= 45:
        left = (45 - minute) + max(0.0, STOPPAGE_1H - extra) + 45 + STOPPAGE_2H
    elif minute < FULL_TIME:
        left = (FULL_TIME - minute) + STOPPAGE_2H
    else:
        left = max(0.0, STOPPAGE_2H - extra) if minute == FULL_TIME else 0.0
    return max(0.0, min(1.0, left / MATCH_MINUTES))


def inplay_wdl(
    pre_wdl: WDL,
    minute: int,
    home_score: int,
    away_score: int,
    red_home: int = 0,
    red_away: int = 0,
    extra: int | None = None,
) -> WDL:
    """In-play W/D/L given the current match state.

    ``pre_wdl`` is the pre-match prior (Elo or market odds). Remaining goals
    are Poisson at the fitted rates, scaled by time left and adjusted for
    red cards.
    """
    frac = remaining_fraction(max(0, int(minute)), extra)
    lead = int(home_score) - int(away_score)

    # Match effectively over → decide on current score.
    if frac <= 1e-6:
        if lead > 0:
            return (1.0, 0.0, 0.0)
        if lead < 0:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)

    lam_h90, lam_a90 = fit_match_rates(pre_wdl)
    lam_h = max(1e-6, lam_h90 * frac)
    lam_a = max(1e-6, lam_a90 * frac)

    for _ in range(max(0, int(red_home))):
        lam_h *= RED_SELF
        lam_a *= RED_OPP
    for _ in range(max(0, int(red_away))):
        lam_a *= RED_SELF
        lam_h *= RED_OPP

    p_hw, p_d, p_aw = _wdl_from_rates(lam_h, lam_a, lead)
    return (float(p_hw), float(p_d), float(p_aw))


def _expected_points(p_win: float, p_draw: float) -> float:
    """Group-stage expected points for a team given its win/draw probability."""
    return 3.0 * p_win + 1.0 * p_draw


def elo_deltas(
    pre_wdl: WDL,
    state_wdl: WDL,
    k_elo: float = 40.0,
    cap: float = 80.0,
) -> Tuple[float, float]:
    """Bounded (home_delta, away_delta) Elo adjustments.

    Mapped from each side's change in expected match points vs pre-match.
    An event that doesn't move the match yields (0, 0).
    """
    p_hw0, p_d0, p_aw0 = pre_wdl
    p_hw, p_d, p_aw = state_wdl
    dep_home = _expected_points(p_hw, p_d) - _expected_points(p_hw0, p_d0)
    dep_away = _expected_points(p_aw, p_d) - _expected_points(p_aw0, p_d0)

    def clamp(x: float) -> float:
        return max(-cap, min(cap, k_elo * x))

    return clamp(dep_home), clamp(dep_away)
