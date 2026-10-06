"""
In-play match outcome model.

Pipeline:
    1. Fit full-match Dixon-Coles scoring rates to the pre-match prior
       (W/D/L, plus the over/under line when the market quotes one).
    2. Remaining goals are Dixon-Coles at those rates scaled by the time
       left and by red cards, giving an in-play W/D/L that depends on the
       current score, minutes remaining and red cards (and equals the prior
       at 0-0 kickoff).

Used by GET /live-prob, the intel agent and the counterfactual agent's live
match conditioning (ml.tournament_sim.LiveMatch). Pure math, no project
imports.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

WDL = Tuple[float, float, float]  # (p_home_win, p_draw, p_away_win)
Totals = Tuple[float, float]  # (P(total goals > line), line)

FULL_TIME = 90
# Mean added time at WC 2026 (104 matches, ESPN clocks): 5.7' first half,
# 8.0' second half. Fitted rates cover a whole match, stoppage included.
STOPPAGE_1H = 5.7
STOPPAGE_2H = 8.0
MATCH_MINUTES = FULL_TIME + STOPPAGE_1H + STOPPAGE_2H
BASE_TOTAL_GOALS = 2.6  # avg combined goals per 90' in men's international football
TOTAL_GOALS_BOUNDS = (0.2 * BASE_TOTAL_GOALS, 3.0 * BASE_TOTAL_GOALS)  # rate-fit start grid
HOME_SHARE_BOUNDS = (0.02, 0.98)  # rate-fit start grid, home share of goals
LOG_RATE_BOUNDS = (math.log(0.005), math.log(12.0))  # per-side rate limits in the fit
RHO_BOUNDS = (-0.3, 0.3)  # Dixon-Coles rho range when fitted to totals
LM_ITERS = 60
LM_TRIES = 12  # damping increases per iteration before a row is declared stationary
LM_TOL = 1e-12  # residual at which a row counts as solved
LM_GTOL = 1e-12  # gradient at which an unsolvable row counts as done
LM_MU0 = 1e-9  # minimum damping (~ pure Newton)
MAX_STEP = 0.5  # step cap (log-rate units)
FD_STEP = 1e-7
FIT_TOL = 1e-8  # max residual for an accepted totals fit
RHO_PATH = Path(__file__).with_name("dixon_coles_coef.json")
# Red-card multipliers on the remaining scoring rate: Cerveny, van Ours & van Tuijl,
# "Effects of a red card on goal-scoring in World Cup football matches",
# Empirical Economics 55:883 (2018), IZA DP 10174 Table 6 col 1, WC 1998-2014:
# own red exp(-0.64), opponent's red exp(0.79). (Ridder, Cramer & Hopstaken,
# JASA 89:1124 (1994), Dutch league: +88% opponent, 10-man side unchanged.)
RED_SELF = math.exp(-0.64)  # ~0.53
RED_OPP = math.exp(0.79)  # ~2.20
MAX_GOALS = 12  # Poisson enumeration cutoff (truncated tail renormalised)
_RHO: float | None = None
_K = np.arange(MAX_GOALS + 1)
_LOG_FACT = np.array([math.lgamma(i + 1) for i in _K])
_MARGIN = _K[:, None] - _K[None, :]
_TOTAL = _K[:, None] + _K[None, :]


def default_rho() -> float:
    """Dixon-Coles rho for priors without a totals market, fitted by
    ml/fit_dixon_coles.py (0 = independent Poisson if the file is missing)."""
    global _RHO
    if _RHO is None:
        try:
            _RHO = float(json.loads(RHO_PATH.read_text(encoding="utf-8"))["rho"])
        except (OSError, KeyError, ValueError):
            _RHO = 0.0
    return _RHO


def _score_matrix(lam_h, lam_a, rho=0.0) -> np.ndarray:
    """
    Dixon-Coles (1997) joint pmf of (home goals, away goals), shape
    ``(..., K+1, K+1)``: independent Poisson with the 0-0, 0-1, 1-0, 1-1 cells
    scaled by ``1 - lh*la*rho``, ``1 + lh*rho``, ``1 + la*rho``, ``1 - rho``.
    rho < 0 moves mass onto 0-0 and 1-1; both marginals and P(total >= 3) are
    unchanged. Inputs broadcast. Truncated tail renormalised.
    """
    lam_h, lam_a, rho = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (lam_h, lam_a, rho))
    )
    ph = np.exp(-lam_h[..., None] + _K * np.log(lam_h[..., None]) - _LOG_FACT)
    pa = np.exp(-lam_a[..., None] + _K * np.log(lam_a[..., None]) - _LOG_FACT)
    joint = ph[..., :, None] * pa[..., None, :]  # [..., gh, ga]
    if np.any(rho != 0.0):
        joint[..., 0, 0] *= np.maximum(0.0, 1.0 - lam_h * lam_a * rho)
        joint[..., 0, 1] *= np.maximum(0.0, 1.0 + lam_h * rho)
        joint[..., 1, 0] *= np.maximum(0.0, 1.0 + lam_a * rho)
        joint[..., 1, 1] *= np.maximum(0.0, 1.0 - rho)
    return joint / joint.sum(axis=(-2, -1), keepdims=True)


def _wdl_from_rates(lam_h, lam_a, lead: int = 0, rho=0.0) -> np.ndarray:
    """W/D/L of final margin ``lead + home goals - away goals`` under
    ``_score_matrix``; shape ``(..., 3)``."""
    joint = _score_matrix(lam_h, lam_a, rho)
    margin = lead + _MARGIN
    return np.stack(
        [(joint * (margin > 0)).sum(axis=(-2, -1)),
         (joint * (margin == 0)).sum(axis=(-2, -1)),
         (joint * (margin < 0)).sum(axis=(-2, -1))],
        axis=-1,
    )


def _p_over(lam_h, lam_a, line: float, rho=0.0) -> np.ndarray:
    """P(total goals > line)."""
    return (_score_matrix(lam_h, lam_a, rho) * (_TOTAL > line)).sum(axis=(-2, -1))


def _solve(resid, x: np.ndarray, lo, hi) -> np.ndarray:
    """
    Batched Levenberg-Marquardt on ``resid(x, rows)`` (rows independent;
    ``rows`` indexes the rows passed in). Forward-difference Jacobian, steps
    capped at ``MAX_STEP``, iterates kept in ``[lo, hi]``. The damping shrinks
    after every accepted step, so a solvable system converges like Newton; a
    prior with no exact solution settles at its least-squares point. x: (P, n).
    """
    x = x.copy()
    n = x.shape[1]
    eye = np.eye(n)
    act = np.arange(len(x))
    mu = np.full(len(x), LM_MU0)
    r = resid(x, act)
    for _ in range(LM_ITERS):
        keep = np.abs(r).max(axis=1) >= LM_TOL
        act, r, mu = act[keep], r[keep], mu[keep]
        if not len(act):
            break
        xa, sse = x[act], (r**2).sum(axis=1)
        J = np.stack([(resid(xa + FD_STEP * eye[j], act) - r) / FD_STEP for j in range(n)], -1)
        JtJ = np.einsum("pij,pik->pjk", J, J)
        g = np.einsum("pij,pi->pj", J, r)
        keep = np.abs(g).max(axis=1) >= LM_GTOL  # least-squares stationary point
        act, r, mu, xa, sse, J, JtJ, g = (
            v[keep] for v in (act, r, mu, xa, sse, J, JtJ, g)
        )
        if not len(act):
            break
        for _ in range(LM_TRIES):
            dx = np.linalg.solve(JtJ + mu[:, None, None] * eye, g[..., None])[..., 0]
            xn = np.clip(xa - np.clip(dx, -MAX_STEP, MAX_STEP), lo, hi)
            rn = resid(xn, act)
            worse = (rn**2).sum(axis=1) >= sse
            if not worse.any():
                break
            mu = np.where(worse, mu * 10.0, mu)
        ok = ~worse  # no descent at any damping: stationary, done
        x[act[ok]] = xn[ok]
        act, r, mu = act[ok], rn[ok], np.maximum(mu[ok] * 0.1, LM_MU0)
    return x


_GRIDS: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def _rate_grid(rho: float) -> tuple[np.ndarray, np.ndarray]:
    """(log rates (G,2), W/D (G,2)) over a (total goals x home share) grid."""
    if rho not in _GRIDS:
        t, s = np.meshgrid(
            np.exp(np.linspace(*(math.log(b) for b in TOTAL_GOALS_BOUNDS), 40)),
            np.linspace(*HOME_SHARE_BOUNDS, 49),
            indexing="ij",
        )
        rates = np.stack([(t * s).ravel(), (t * (1 - s)).ravel()], axis=1)
        wd = _wdl_from_rates(rates[:, 0], rates[:, 1], rho=rho)[:, :2]
        _GRIDS[rho] = (np.log(rates), wd)
    return _GRIDS[rho]


def rates_from_wdl_batch(wdl: np.ndarray, rho: float | None = None) -> np.ndarray:
    """
    (P,3) W/D/L → (P,2) 90' rates ``(lam_home, lam_away)`` that reproduce W and
    D exactly under Dixon-Coles ``rho`` (default ``default_rho()``): nearest
    point on a grid, then Levenberg-Marquardt in log-rate space.
    """
    rho = default_rho() if rho is None else float(rho)
    q = np.asarray(wdl, dtype=np.float64)
    q = (q / q.sum(axis=1, keepdims=True))[:, :2]
    log_rates, g = _rate_grid(rho)
    x0 = log_rates[((g**2).sum(1) - 2 * q @ g.T).argmin(1)]  # nearest grid point

    def resid(x, rows):
        return _wdl_from_rates(np.exp(x[:, 0]), np.exp(x[:, 1]), rho=rho)[:, :2] - q[rows]

    return np.exp(_solve(resid, x0, *LOG_RATE_BOUNDS))


@lru_cache(maxsize=4096)
def fit_match_rates(pre_wdl: WDL, totals: Totals | None = None) -> Tuple[float, float, float]:
    """
    Full-match (90') Dixon-Coles parameters ``(lam_home, lam_away, rho)``.

    1X2 only: W and D are matched exactly at ``rho = default_rho()``. With a
    totals market rho is freed, and W, D and P(over) are all matched exactly,
    so the model carries the market's goal expectancy as well as its draw
    price. If no valid rho solves that, the totals are dropped. Either way the
    in-play model reproduces the prior at 0-0 kickoff.
    """
    rho0 = default_rho()
    lh, la = (float(v) for v in rates_from_wdl_batch(np.array([pre_wdl]), rho0)[0])
    if totals is None:
        return lh, la, rho0

    p_over, line = totals
    q = np.asarray(pre_wdl, dtype=np.float64)
    target = np.array([*(q[:2] / q.sum()), p_over])

    def resid(x, rows=None):
        h, a, rho = np.exp(x[:, 0]), np.exp(x[:, 1]), x[:, 2]
        wdl = _wdl_from_rates(h, a, rho=rho)
        return np.stack([wdl[:, 0], wdl[:, 1], _p_over(h, a, line, rho)], -1) - target

    lo = np.array([LOG_RATE_BOUNDS[0], LOG_RATE_BOUNDS[0], RHO_BOUNDS[0]])
    hi = np.array([LOG_RATE_BOUNDS[1], LOG_RATE_BOUNDS[1], RHO_BOUNDS[1]])
    x = _solve(resid, np.array([[math.log(lh), math.log(la), rho0]]), lo, hi)
    h, a, rho = math.exp(x[0, 0]), math.exp(x[0, 1]), float(x[0, 2])
    if 1 + max(h, a) * rho >= 0 and h * a * rho <= 1 and np.abs(resid(x)).max() < FIT_TOL:
        return h, a, rho
    return lh, la, rho0


def ko_advance_batch(wdl: np.ndarray) -> np.ndarray:
    """P(home side advances) in a knockout tie: 90' W/D/L, then 30' of extra
    time at a third of the fitted rates, then a 50/50 shootout."""
    wdl = np.asarray(wdl, dtype=np.float64)
    r = rates_from_wdl_batch(wdl)
    et = _wdl_from_rates(r[:, 0] / 3.0, r[:, 1] / 3.0, rho=default_rho())
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


def apply_reds(lam_h, lam_a, red_home: int, red_away: int):
    """Remaining-goal rates after each side's red cards (RED_SELF on the
    carded side, RED_OPP on its opponent, once per card)."""
    for _ in range(red_home):
        lam_h *= RED_SELF
        lam_a *= RED_OPP
    for _ in range(red_away):
        lam_a *= RED_SELF
        lam_h *= RED_OPP
    return lam_h, lam_a


def inplay_wdl(
    pre_wdl: WDL,
    minute: int,
    home_score: int,
    away_score: int,
    red_home: int = 0,
    red_away: int = 0,
    extra: int | None = None,
    totals: Totals | None = None,
) -> WDL:
    """
    In-play W/D/L given the current match state.

    ``pre_wdl`` is the pre-match prior (from Elo / market odds), ``totals`` the
    market's over/under when quoted. Full-match Dixon-Coles parameters are
    fitted so that they reproduce it (``fit_match_rates``); remaining goals use
    those rates scaled by the fraction of the match left and adjusted for red
    cards, with the same rho (an approximation: Dixon-Coles defines rho on
    final scores). The final-margin distribution is enumerated to give W/D/L.
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

    lam_h90, lam_a90, rho = fit_match_rates(tuple(pre_wdl), totals)
    lam_h = max(1e-6, lam_h90 * frac)
    lam_a = max(1e-6, lam_a90 * frac)

    lam_h, lam_a = apply_reds(lam_h, lam_a, int(red_home), int(red_away))
    p_hw, p_d, p_aw = _wdl_from_rates(lam_h, lam_a, lead, rho)
    return (float(p_hw), float(p_d), float(p_aw))
