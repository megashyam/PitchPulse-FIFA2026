"""
World Football Elo (eloratings.net method).

Computed from the martj42 international results dataset (every men's
international since 1872, https://github.com/martj42/international_results).

    K   60 World Cup finals · 50 continental finals / Confederations Cup ·
        40 World Cup & continental qualifiers, Nations Leagues · 30 other
        tournaments · 20 friendlies
    G   goal-difference multiplier: 1 (≤1), 1.5 (2), (11+N)/8 (N≥3)
    H   +100 to the home side unless neutral
    Shootouts count as draws.

ratings_before(date) is point-in-time: only matches strictly before `date`,
so a pre-tournament snapshot never sees tournament results.
"""

from __future__ import annotations

import csv
import io
from typing import Iterable, Optional

import httpx

from ml.team_names import canonical

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
INITIAL = 1500.0
HOME_ADV = 100.0

_K60 = {"FIFA World Cup"}
_K50 = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
    "Confederations Cup",
}


def k_factor(tournament: str) -> float:
    if tournament in _K60:
        return 60.0
    if tournament in _K50:
        return 50.0
    if "qualification" in tournament or "Nations League" in tournament:
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def goal_mult(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def expected(dr: float) -> float:
    return 1.0 / (10 ** (-dr / 400.0) + 1.0)


def load_results(text: Optional[str] = None) -> list[dict]:
    if text is None:
        r = httpx.get(RESULTS_URL, timeout=60)
        r.raise_for_status()
        text = r.text
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["home_score"] in ("", "NA") or row["away_score"] in ("", "NA"):
            continue
        rows.append(
            {
                "date": row["date"],
                "home": canonical(row["home_team"]),
                "away": canonical(row["away_team"]),
                "hs": int(row["home_score"]),
                "as": int(row["away_score"]),
                "tournament": row["tournament"],
                "neutral": row["neutral"].upper() == "TRUE",
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def run_elo(results: Iterable[dict], before: Optional[str] = None) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for m in results:
        if before is not None and m["date"] >= before:
            break
        rh = ratings.get(m["home"], INITIAL)
        ra = ratings.get(m["away"], INITIAL)
        dr = rh - ra + (0.0 if m["neutral"] else HOME_ADV)
        we = expected(dr)
        w = 1.0 if m["hs"] > m["as"] else 0.5 if m["hs"] == m["as"] else 0.0
        delta = k_factor(m["tournament"]) * goal_mult(m["hs"] - m["as"]) * (w - we)
        ratings[m["home"]] = rh + delta
        ratings[m["away"]] = ra - delta
    return ratings


def ratings_before(date: str, results: Optional[list[dict]] = None) -> dict[str, float]:
    return run_elo(results if results is not None else load_results(), before=date)
