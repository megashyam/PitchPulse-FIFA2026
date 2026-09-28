"""
Pre-match briefing facts.

Every number a briefing may cite, computed as of kickoff so a briefing
requested later never leaks the result:
    - stage, venue, host advantage
    - pre-match W/D/L (market odds or Elo) and point-in-time Elo
    - each team's earlier WC 2026 matches: results, possession, shots, xG
    - last meetings between the two (martj42 international results)
"""

from __future__ import annotations

from typing import Optional

from feeds import snapshot
from ml.prior_builder import match_wdl
from ml.wc_2026_config import FIXTURE_BY_ID, FIXTURES, TEAM_BY_NAME


def _form(team: str, before: str) -> list[dict]:
    out = []
    for f in FIXTURES:
        if f["date"] >= before or f["status"] not in ("FT", "AET", "PEN"):
            continue
        if team not in (f["home_name"], f["away_name"]):
            continue
        home = f["home_name"] == team
        d = snapshot.match_detail(f["fixture_id"])
        mine = (d["home_stats"] if home else d["away_stats"]) if d else None
        theirs = (d["away_stats"] if home else d["home_stats"]) if d else None
        out.append(
            {
                "opponent": f["away_name"] if home else f["home_name"],
                "round": f["round"],
                "gf": f["home_score"] if home else f["away_score"],
                "ga": f["away_score"] if home else f["home_score"],
                "status": f["status"],
                "won": f["winner"] == team,
                "possession": mine.possession if mine else None,
                "shots": mine.shots_total if mine else None,
                "xg_for": mine.expected_goals if mine else None,
                "xg_against": theirs.expected_goals if theirs else None,
            }
        )
    return out


def briefing_facts(
    fixture_id: Optional[int], home: str, away: str, odds_table=None
) -> dict:
    fx = FIXTURE_BY_ID.get(fixture_id or -1)
    wdl = match_wdl(
        home, away, host_side=fx.get("host_side") if fx else None, odds_table=odds_table
    )
    quoted = bool(odds_table) and ((home, away) in odds_table or (away, home) in odds_table)
    h, a = TEAM_BY_NAME.get(home), TEAM_BY_NAME.get(away)
    before = fx["date"] if fx else "9999"
    return {
        "round": fx["round"] if fx else None,
        "venue": fx.get("venue") if fx else None,
        "host_side": fx.get("host_side") if fx else None,
        "wdl": wdl,
        "wdl_source": "market odds" if quoted else "Elo",
        "elo": (
            fx["elo_home_pre"] if fx else (h.elo if h else None),
            fx["elo_away_pre"] if fx else (a.elo if a else None),
        ),
        "form": {home: _form(home, before), away: _form(away, before)},
        "h2h": fx.get("h2h", []) if fx else [],
    }


def _form_line(team: str, games: list[dict]) -> str:
    if not games:
        return f"{team}: first match of the tournament."
    parts = []
    for g in games:
        res = "W" if g["won"] else ("D" if g["gf"] == g["ga"] and g["status"] == "FT" else "L")
        extra = " aet" if g["status"] == "AET" else " pens" if g["status"] == "PEN" else ""
        stats = (
            f", {g['possession']:.0f}% possession, {g['shots']} shots, "
            f"xG {g['xg_for']:.2f}-{g['xg_against']:.2f}"
            if g["possession"] is not None
            else ""
        )
        parts.append(f"{res} {g['gf']}-{g['ga']}{extra} vs {g['opponent']} ({g['round']}{stats})")
    xg = [g for g in games if g["xg_for"] is not None]
    avg = (
        f" Averages: xG {sum(g['xg_for'] for g in xg) / len(xg):.2f} for, "
        f"{sum(g['xg_against'] for g in xg) / len(xg):.2f} against; possession "
        f"{sum(g['possession'] for g in xg) / len(xg):.0f}%."
        if xg
        else ""
    )
    return f"{team} so far: " + "; ".join(parts) + "." + avg


def render(facts: dict, home: str, away: str) -> str:
    w, d, lo = facts["wdl"]
    eh, ea = facts["elo"]
    lines = []
    if facts["round"]:
        lines.append(f"Stage: {facts['round']}" + (f" at {facts['venue']}" if facts["venue"] else "") + ".")
    if facts["host_side"]:
        host = home if facts["host_side"] == "home" else away
        lines.append(f"{host} are hosts playing at home.")
    lines.append(
        f"Pre-match probabilities ({facts['wdl_source']}): {home} {w:.0%}, draw {d:.0%}, {away} {lo:.0%}."
    )
    if eh and ea:
        lines.append(f"Elo before kickoff: {home} {eh:.0f}, {away} {ea:.0f}.")
    lines.append(_form_line(home, facts["form"][home]))
    lines.append(_form_line(away, facts["form"][away]))
    if facts["h2h"]:
        lines.append(
            "Last meetings: "
            + "; ".join(
                f"{m['date'][:4]} {m['home']} {m['hs']}-{m['as']} {m['away']} ({m['tournament']})"
                for m in facts["h2h"]
            )
            + "."
        )
    else:
        lines.append("No previous meetings on record.")
    return "\n".join(lines)
