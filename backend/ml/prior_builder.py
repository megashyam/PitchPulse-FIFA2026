import json
import math
from pathlib import Path

import numpy as np
from typing import Dict, Optional, Tuple

from ml.in_play import ko_advance_batch
from ml.wc_2026_config import TEAM_BY_NAME, WC2026_TEAMS

# match probability: (p_win, p_draw, p_loss) from team_a's view
MatchProb = Tuple[float, float, float]

# probability floor; avoids log(0) and extreme bracket distortion in the sim
MIN_PROB = 0.01

# ordered logit coefficients and home advantage, fitted by ml/fit_wdl_prior.py
WDL_COEF_PATH = Path(__file__).with_name("wdl_prior_coef.json")
_WDL_COEF: Optional[dict] = None
DEFAULT_WDL: "MatchProb" = (0.40, 0.25, 0.35)


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for A against B (1=win, 0.5=draw, 0=loss)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _wdl_coef() -> dict:
    global _WDL_COEF
    if _WDL_COEF is None:
        _WDL_COEF = json.loads(WDL_COEF_PATH.read_text(encoding="utf-8"))
    return _WDL_COEF


def elo_to_wdl(rating_a: float, rating_b: float) -> MatchProb:
    """
    Elo ratings to W/D/L via a symmetric ordered logit on x = rating_a - rating_b:

        p_win  = sigmoid(beta*x - theta)
        p_loss = sigmoid(-beta*x - theta)
        p_draw = 1 - p_win - p_loss

    beta, theta from ml/wdl_prior_coef.json. Swapping teams swaps win and loss.
    """
    c = _wdl_coef()
    bx = c["beta"] * (rating_a - rating_b)
    p_win = 1.0 / (1.0 + math.exp(c["theta"] - bx))
    p_loss = 1.0 / (1.0 + math.exp(c["theta"] + bx))
    return _normalise(p_win, 1.0 - p_win - p_loss, p_loss)


def _shin_probs(pi: np.ndarray, z: float) -> np.ndarray:
    """Shin (1993) true probabilities for insider fraction ``z``."""
    big_pi = pi.sum()
    return (np.sqrt(z**2 + 4.0 * (1.0 - z) * pi**2 / big_pi) - z) / (2.0 * (1.0 - z))


def shin_devig(odds: Tuple[float, ...], tol: float = 1e-12) -> Tuple[np.ndarray, float]:
    """
    Shin (1993) de-vigging of decimal odds.

    With raw implied probabilities pi_i = 1/odds_i and booksum Pi = sum(pi_i),
    the fair probabilities are

        p_i(z) = (sqrt(z^2 + 4(1-z) * pi_i^2 / Pi) - z) / (2(1-z))

    where z (the share of insider money) is solved by bisection on [0, 0.5]
    so that sum_i p_i(z) = 1. sum_i p_i(z) is strictly decreasing in z, so the
    root is unique. Unlike proportional normalisation this strips more margin
    from longshots than from favourites (the favourite-longshot bias).

    Returns ``(probs, z)``. A book with no overround (Pi <= 1) has no margin to
    attribute to insiders: z = 0 and the result is plain normalisation.
    """
    pi = 1.0 / np.asarray(odds, dtype=np.float64)
    big_pi = pi.sum()
    if big_pi <= 1.0:
        return pi / big_pi, 0.0

    lo, hi = 0.0, 0.5
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _shin_probs(pi, mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    return _shin_probs(pi, z), z


def oddsapi_to_wdl(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> MatchProb:
    """Decimal odds → fair W/D/L via Shin de-vigging, then the MIN_PROB floor."""
    fair, _z = shin_devig((odds_home, odds_draw, odds_away))
    return _normalise(fair[0], fair[1], fair[2])


def _normalise(p_win: float, p_draw: float, p_loss: float) -> MatchProb:
    """Clip to floor then re-normalise to sum exactly to 1.0."""
    arr = np.array([p_win, p_draw, p_loss], dtype=np.float64)
    arr = np.clip(arr, MIN_PROB, None)
    arr /= arr.sum()
    return float(arr[0]), float(arr[1]), float(arr[2])


def build_prior_table(
    betfair_odds: Optional[Dict[Tuple[str, str], Tuple[float, float, float]]] = None,
    elo_overrides: Optional[Dict[str, float]] = None,
) -> Dict[Tuple[str, str], MatchProb]:
    """W/D/L prior for every possible pairing of the 48 WC 2026 teams.

    Args:
        betfair_odds: optional {(home, away): (decimal_home, decimal_draw,
                      decimal_away)} market odds; override Elo where present.

    Returns:
        {(team_a, team_b): (p_win, p_draw, p_loss)} from team_a's perspective.
    """
    teams = WC2026_TEAMS
    n = len(teams)
    ovr = elo_overrides or {}
    priors: Dict[Tuple[str, str], MatchProb] = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ta = teams[i]
            tb = teams[j]
            key = (ta.name, tb.name)
            # conditioned team: bypass the static Betfair line when either side is
            # nudged, so its adjusted strength is used
            conditioned = ta.name in ovr or tb.name in ovr

            if not conditioned and betfair_odds and key in betfair_odds:
                o_h, o_d, o_a = betfair_odds[key]
                priors[key] = oddsapi_to_wdl(o_h, o_d, o_a)
            elif (
                not conditioned and betfair_odds and (tb.name, ta.name) in betfair_odds
            ):
                o_h, o_d, o_a = betfair_odds[(tb.name, ta.name)]
                rev = oddsapi_to_wdl(o_h, o_d, o_a)
                priors[key] = (rev[2], rev[1], rev[0])  # flip perspective
            else:
                priors[key] = elo_to_wdl(
                    ta.elo + ovr.get(ta.name, 0.0),
                    tb.elo + ovr.get(tb.name, 0.0),
                )

    return priors


def match_wdl(
    home: str,
    away: str,
    *,
    host_side: Optional[str] = None,
    odds_table: Optional[Dict[Tuple[str, str], Tuple[float, float, float]]] = None,
    elo_overrides: Optional[Dict[str, float]] = None,
) -> MatchProb:
    """Pre-match W/D/L prior for a real fixture.

    Market odds when quoted and neither side is conditioned, else Elo with
    home advantage for a host nation playing in its own country.
    """
    ovr = elo_overrides or {}
    if odds_table and home not in ovr and away not in ovr:
        if (home, away) in odds_table:
            return oddsapi_to_wdl(*odds_table[(home, away)])
        if (away, home) in odds_table:
            w, d, lo = oddsapi_to_wdl(*odds_table[(away, home)])
            return (lo, d, w)
    h, a = TEAM_BY_NAME.get(home), TEAM_BY_NAME.get(away)
    if h is None or a is None:
        return DEFAULT_WDL
    home_adv = _wdl_coef()["home_adv"]
    rh = h.elo + ovr.get(home, 0.0) + (home_adv if host_side == "home" else 0.0)
    ra = a.elo + ovr.get(away, 0.0) + (home_adv if host_side == "away" else 0.0)
    return elo_to_wdl(rh, ra)


def ko_prob(p_win: float, p_draw: float, p_loss: float) -> Tuple[float, float]:
    """(p_team_a_advances, p_team_b_advances) from 90' W/D/L.

    Extra time at a third of the fitted Poisson rates, then a 50/50 shootout.
    """
    p_a = float(ko_advance_batch(np.array([[p_win, p_draw, p_loss]]))[0])
    return p_a, 1.0 - p_a


def build_ko_matrix(
    priors: Dict[Tuple[str, str], MatchProb],
    elo_overrides: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """(48 × 48) float32 matrix: P(team i advances vs team j), neutral venue.

    Indexed as ko_matrix[home_idxs, away_idxs] in the simulator.
    """
    teams = WC2026_TEAMS
    n = len(teams)
    ovr = elo_overrides or {}
    iu, ju = np.triu_indices(n, k=1)
    wdl = np.array(
        [
            priors.get(
                (teams[i].name, teams[j].name),
                elo_to_wdl(
                    teams[i].elo + ovr.get(teams[i].name, 0.0),
                    teams[j].elo + ovr.get(teams[j].name, 0.0),
                ),
            )
            for i, j in zip(iu, ju)
        ]
    )
    adv = ko_advance_batch(wdl)
    matrix = np.zeros((n, n), dtype=np.float32)
    matrix[iu, ju] = adv
    matrix[ju, iu] = 1.0 - adv
    return matrix
