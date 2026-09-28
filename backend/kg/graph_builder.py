"""
Neo4j knowledge graph builder.

Builds the graph from the static tournament config (ml/wc_2026_config.py)
and StatsBomb historical results (ml/statsbomb.py).

Usage:
    python -m kg.graph_builder          # build (idempotent)
    python -m kg.graph_builder --check  # report counts only
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import httpx

from kg.neo4j_client import Neo4jClient, get_neo4j_client
from ml.statsbomb import COMPETITION_ID, SB_BASE, SEASON_IDS
from ml.team_names import SIM_NAMES, to_sim
from ml.wc2026_format import R32_SLOTS
from ml.wc_2026_config import GROUPS, WC2026_TEAMS

log = logging.getLogger(__name__)


def build_static_graph(client: Neo4jClient) -> None:
    """Team/Group PLAYS_IN edges and BracketRound nodes from the static config."""
    if not client.ready:
        log.warning("Neo4j not ready — skipping static graph build")
        return

    for group_code, teams in GROUPS.items():
        for t in teams:
            client.run(
                "MERGE (t:Team {name: $name}) "
                "SET t.elo = $elo, t.fifa_rank = $fifa_rank "
                "MERGE (g:Group {code: $code}) "
                "MERGE (t)-[:PLAYS_IN]->(g)",
                name=t.name,
                elo=t.elo,
                fifa_rank=t.fifa_rank,
                code=group_code,
            )

    # FIFA R32 slots ("1E" v "3:ABCDF"), keyed by match number.
    for match_no, (slot_a, slot_b) in R32_SLOTS.items():
        client.run(
            "MERGE (r:BracketRound {round: 'R32', match_no: $m}) "
            "SET r.slot_a = $slot_a, r.slot_b = $slot_b",
            m=match_no,
            slot_a=slot_a,
            slot_b=slot_b,
        )
    log.info(
        f"Static graph built: {len(WC2026_TEAMS)} teams, {len(GROUPS)} groups, "
        f"{len(R32_SLOTS)} R32 slots"
    )


async def build_head_to_head_edges(client: Neo4jClient) -> int:
    """HEAD_TO_HEAD edges from StatsBomb results between two WC 2026 teams."""
    if not await asyncio.to_thread(lambda: client.ready):
        log.warning("Neo4j not ready — skipping head-to-head build")
        return 0

    written = 0
    async with httpx.AsyncClient() as http:
        for sid in SEASON_IDS:
            url = f"{SB_BASE}/matches/{COMPETITION_ID}/{sid}.json"
            try:
                r = await http.get(url, timeout=30)
                r.raise_for_status()
                matches = r.json()
            except Exception as exc:
                log.warning(f"StatsBomb season {sid} failed: {exc}")
                continue

            for m in matches:
                home = to_sim(m["home_team"]["home_team_name"])
                away = to_sim(m["away_team"]["away_team_name"])
                if home == away or home not in SIM_NAMES or away not in SIM_NAMES:
                    continue
                await asyncio.to_thread(
                    client.run,
                    "MATCH (h:Team {name: $home}) "
                    "MATCH (a:Team {name: $away}) "
                    "MERGE (h)-[r:HEAD_TO_HEAD {match_id: $mid}]->(a) "
                    "SET r.home_score = $home_score, r.away_score = $away_score, "
                    "r.competition = $comp, r.season = $season, "
                    "r.match_date = $date",
                    home=home,
                    away=away,
                    mid=m["match_id"],
                    home_score=m.get("home_score", 0),
                    away_score=m.get("away_score", 0),
                    comp=m.get("competition", {}).get("competition_name", "World Cup"),
                    season=sid,
                    date=m.get("match_date", ""),
                )
                written += 1
    log.info(f"Head-to-head graph build: {written} historical fixtures written")
    return written


async def ensure_graph_built() -> None:
    """Build the graph at startup if it looks empty. Never raises."""
    try:
        # The driver is synchronous; keep it off the event loop.
        client = await asyncio.to_thread(get_neo4j_client)
        if not await asyncio.to_thread(lambda: client.ready):
            return
        existing = await asyncio.to_thread(client.run, "MATCH (t:Team) RETURN count(t) AS n")
        if existing and existing[0].get("n", 0) > 0:
            return
        await asyncio.to_thread(build_static_graph, client)
        await build_head_to_head_edges(client)
    except Exception as exc:
        log.warning(f"Neo4j auto-build failed: {exc}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build the WC2026 Neo4j knowledge graph")
    parser.add_argument("--check", action="store_true", help="Report counts only, no writes")
    args = parser.parse_args()

    client = get_neo4j_client()
    if not client.ready:
        print("Neo4j not reachable — is `docker compose up neo4j` running?")
        return

    if args.check:
        teams = client.run("MATCH (t:Team) RETURN count(t) AS n")
        h2h = client.run("MATCH ()-[r:HEAD_TO_HEAD]->() RETURN count(r) AS n")
        print(f"Teams: {teams[0]['n'] if teams else 0}")
        print(f"Head-to-head edges: {h2h[0]['n'] if h2h else 0}")
        return

    build_static_graph(client)
    written = asyncio.run(build_head_to_head_edges(client))
    print(f"Graph build complete: {len(WC2026_TEAMS)} teams, {written} head-to-head edges")


if __name__ == "__main__":
    main()
