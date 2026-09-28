"""
WC 2026 snapshot builder.

Freezes the real tournament into committed files so the app never depends
on a live feed for finished matches:
    data/wc2026/tournament.json         teams, groups, fixtures, pre-match
                                        Elo, head-to-head (last 5 meetings)
    data/wc2026/third_place_table.json  FIFA Annex C (495 combinations)
    data/wc2026/matches/{fixture}.json  stats, events, lineups, shots (xG)

Sources (free, no keys): ESPN public API, martj42 international results
(Elo), Wikipedia's transcription of Annex C. Sources are cross-validated;
the build fails on any mismatch.

Usage:
    python -m feeds.build_snapshot [--cache DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import httpx

from feeds import espn
from ml import elo_ratings
from ml.wc2026_format import (
    GROUP_LETTERS,
    KO_TREE,
    R32_SLOTS,
    THIRD_SLOT_WINNERS,
    third_slot_of,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "wc2026"
ANNEX_C_URL = (
    "https://en.wikipedia.org/w/index.php?title="
    "Template:2026_FIFA_World_Cup_third-place_table&action=raw"
)
# Wikimedia rejects default UAs; ESPN rejects custom ones — set per request.
WIKI_HEADERS = {"User-Agent": "PitchPulse/1.0 (WC2026 snapshot builder)"}


def parse_annex_c(wikitext: str) -> dict[str, dict[str, str]]:
    """Wikitext table → {"BDEFIJKL": {"1A": "E", ...}} and validate it."""
    table: dict[str, dict[str, str]] = {}
    for block in re.split(r'\n!\s*scope="row"\s*\|', wikitext)[1:]:
        block = block.split("\n|-")[0]
        groups = re.findall(r"'''([A-L])'''", block)
        thirds = re.findall(r"\b3([A-L])\b", block)
        if len(groups) != 8 or len(thirds) != 8:
            continue
        table["".join(sorted(groups))] = dict(zip(THIRD_SLOT_WINNERS, thirds))
    if len(table) != 495:
        raise ValueError(f"Annex C: expected 495 combinations, parsed {len(table)}")
    for combo in ("".join(c) for c in combinations(GROUP_LETTERS, 8)):
        row = table.get(combo)
        if row is None:
            raise ValueError(f"Annex C: missing combination {combo}")
        if sorted(row.values()) != list(combo):
            raise ValueError(f"Annex C {combo}: thirds {row} are not a permutation")
        for w, g in row.items():
            if g not in third_slot_of(w)[1]:
                raise ValueError(f"Annex C {combo}: 3{g} not eligible to face {w}")
    return table


def assign_match_numbers(fixtures: list[dict], groups: dict[str, list[dict]], annex: dict) -> None:
    """Attach FIFA match numbers (73-104) and verify the bracket."""
    pos: dict[str, str] = {}
    for letter, rows in groups.items():
        for r in rows:
            pos[r["team"]] = f"{r['rank']}{letter}"
    thirds = sorted(
        (rows[2] for rows in groups.values()),
        key=lambda r: (-r["points"], -r["gd"], -r["gf"]),
    )[:8]
    combo = "".join(sorted(pos[t["team"]][1] for t in thirds))
    third_for = {w: f"3{g}" for w, g in annex[combo].items()}

    def slot_team(code: str, winner_code: str) -> str:
        return third_for[winner_code] if code.startswith("3:") else code

    expected = {m: {slot_team(b, a), a} for m, (a, b) in R32_SLOTS.items()}
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    ko = [f for f in fixtures if f["stage"] != "group"]
    for f in sorted(ko, key=lambda f: f["date"]):
        teams = {f["home_name"], f["away_name"]}
        if f["stage"] == "r32":
            codes = {pos[t] for t in teams}
            m = next((m for m, c in expected.items() if c == codes), None)
        elif f["stage"] == "3rd":
            m = 103
        else:
            m = next(
                (m for m, (a, b) in KO_TREE.items() if {winners.get(a), winners.get(b)} == teams),
                None,
            )
        if m is None:
            raise ValueError(f"{f['home_name']} v {f['away_name']} ({f['stage']}) fits no bracket slot")
        f["match_no"] = m
        if f["winner"]:
            winners[m] = f["winner"]
            losers[m] = (teams - {f["winner"]}).pop()
    if 103 in {f.get("match_no") for f in ko} and {losers.get(101), losers.get(102)} != {
        next(f for f in ko if f.get("match_no") == 103)[k] for k in ("home_name", "away_name")
    }:
        raise ValueError("third-place match teams are not the semi-final losers")


async def _build(cache: Path | None) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "matches").mkdir(exist_ok=True)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        events = await espn.fetch_scoreboard(client)
        groups = espn.parse_standings(await espn.fetch_standings(client))
        annex_raw = (await client.get(ANNEX_C_URL, headers=WIKI_HEADERS)).text
        results = elo_ratings.load_results((await client.get(elo_ratings.RESULTS_URL)).text)

        annex = parse_annex_c(annex_raw)
        (DATA / "third_place_table.json").write_text(
            json.dumps({"source": ANNEX_C_URL, "table": annex}, indent=1), encoding="utf-8"
        )

        fixtures = [f for e in events if (f := espn.parse_event(e))]
        fixtures.sort(key=lambda f: (f["date"], f["fixture_id"]))
        if len(fixtures) != 104:
            raise ValueError(f"expected 104 fixtures, got {len(fixtures)}")
        team_names = {r["team"] for rows in groups.values() for r in rows}
        if len(team_names) != 48:
            raise ValueError(f"expected 48 teams, got {len(team_names)}")
        assign_match_numbers(fixtures, groups, annex)

        start = min(f["date"][:10] for f in fixtures)
        elo_start = elo_ratings.run_elo(results, before=start)
        missing = sorted(t for t in team_names if t not in elo_start)
        if missing:
            raise ValueError(f"no Elo history for {missing} — add aliases to ml/team_names.py")

        # Point-in-time pre-match Elo and head-to-head per fixture.
        for f in fixtures:
            day = f["date"][:10]
            e = elo_ratings.run_elo(results, before=day)
            f["elo_home_pre"] = round(e[f["home_name"]], 1)
            f["elo_away_pre"] = round(e[f["away_name"]], 1)
            pair = {f["home_name"], f["away_name"]}
            f["h2h"] = [
                {k: m[k] for k in ("date", "home", "away", "hs", "as", "tournament")}
                for m in reversed(results)
                if m["date"] < day and {m["home"], m["away"]} == pair
            ][:5]

        for f in fixtures:
            fid = f["fixture_id"]
            raw = None
            if cache and (cache / f"{fid}.json").exists():
                raw = json.loads((cache / f"{fid}.json").read_text(encoding="utf-8"))
            if raw is None:
                raw = await espn.fetch_summary(client, fid)
                if cache:
                    (cache / f"{fid}.json").write_text(json.dumps(raw), encoding="utf-8")
            detail = espn.parse_summary(raw, f)
            out = {
                "home_stats": detail["home_stats"].model_dump(),
                "away_stats": detail["away_stats"].model_dump(),
                "events": [e.model_dump() for e in detail["events"]],
                "lineups": detail["lineups"],
                "shots": detail["shots"],
            }
            (DATA / "matches" / f"{fid}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    by_team = {r["team"]: (letter, r) for letter, rows in groups.items() for r in rows}
    teams = [
        {
            "name": name,
            "group": by_team[name][0],
            "espn_id": by_team[name][1]["espn_id"],
            "elo": round(elo_start[name], 1),
        }
        for name in sorted(team_names, key=lambda n: (by_team[n][0], -elo_start[n]))
    ]
    for f in fixtures:
        f.pop("kickoff", None)
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "fixtures": f"{espn.SITE}/{espn.LEAGUE}/scoreboard?dates={espn.SEASON}",
            "groups": f"{espn.API}/{espn.LEAGUE}/standings",
            "elo": elo_ratings.RESULTS_URL,
            "annex_c": ANNEX_C_URL,
        },
        "elo_asof": start,
        "teams": teams,
        "groups": groups,
        "fixtures": fixtures,
    }
    (DATA / "tournament.json").write_text(json.dumps(snapshot, indent=1, default=str), encoding="utf-8")
    print(f"snapshot: {len(teams)} teams, {len(fixtures)} fixtures → {DATA}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, help="dir of cached raw ESPN summaries")
    a = ap.parse_args()
    asyncio.run(_build(a.cache))


if __name__ == "__main__":
    main()
