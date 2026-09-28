"""
Neo4j driver wrapper that degrades on failure.

connect() never raises, `ready` reflects live reachability, and every public
method returns an empty result when Neo4j is down.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from neo4j import GraphDatabase

from kg.schema import CONSTRAINT_STATEMENTS

log = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "wc2026pass")
READY_TTL_S = 5.0


class Neo4jClient:

    def __init__(self) -> None:
        self._driver = None
        self._ready = False
        self._ready_at = float("-inf")

    def connect(self) -> None:
        try:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._driver.verify_connectivity()
            with self._driver.session() as session:
                for stmt in CONSTRAINT_STATEMENTS:
                    session.run(stmt)
            log.info(f"Neo4j connected ({NEO4J_URI})")
        except Exception as exc:
            log.error(
                f"Neo4j connection failed ({NEO4J_URI}): {exc}. "
                "Graph context will be skipped — agents still work without it."
            )
            self._driver = None

    def close(self) -> None:
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass

    @property
    def ready(self) -> bool:
        """verify_connectivity(), cached for READY_TTL_S."""
        if self._driver is None:
            return False
        now = time.monotonic()
        if now - self._ready_at < READY_TTL_S:
            return self._ready
        try:
            self._driver.verify_connectivity()
            self._ready = True
        except Exception:
            self._ready = False
        self._ready_at = now
        return self._ready

    def run(self, cypher: str, **params) -> list[dict]:
        if not self.ready:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(cypher, **params)
                return [dict(record) for record in result]
        except Exception as exc:
            log.warning(f"Neo4j query failed: {exc}")
            return []

    # ── Query helpers ────────────────────────────────────────────────────

    def get_head_to_head(self, team_a: str, team_b: str, limit: int = 5) -> list[dict]:
        """Prior WC meetings between two teams, most recent first."""
        cypher = (
            "MATCH (a:Team {name: $team_a})-[r:HEAD_TO_HEAD]-(b:Team {name: $team_b}) "
            "RETURN a.name AS team_a, b.name AS team_b, r.home_score AS home_score, "
            "r.away_score AS away_score, r.competition AS competition, "
            "r.season AS season, r.match_date AS match_date "
            "ORDER BY r.match_date DESC LIMIT $limit"
        )
        return self.run(cypher, team_a=team_a, team_b=team_b, limit=limit)


_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
        _client.connect()
    return _client
