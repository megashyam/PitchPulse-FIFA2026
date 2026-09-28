"""
Knowledge-graph schema and persistence tests.

Covers pure schema-shape checks that require no live service, plus a
round-trip through the real Neo4j driver that is skipped when the graph
database is unavailable.

Run from backend/:
    python -m pytest tests/test_kg_schema.py -v
"""

from __future__ import annotations

import pytest

from kg.neo4j_client import get_neo4j_client
from kg.schema import (
    CONSTRAINT_STATEMENTS,
    GROUP,
    HEAD_TO_HEAD,
    HISTORICAL_MATCH,
    PLAYS_IN,
    TEAM,
)

# --------------------------------------------------------------- pure schema shape


def test_constraint_statements_reference_declared_labels():
    """Every constraint statement targets one of the schema's node labels."""
    labels = {TEAM, GROUP, HISTORICAL_MATCH}
    for stmt in CONSTRAINT_STATEMENTS:
        assert any(f":{label})" in stmt for label in labels), stmt


def test_constraint_statements_are_idempotent():
    """Constraint DDL is idempotent (IF NOT EXISTS), safe on every startup."""
    for stmt in CONSTRAINT_STATEMENTS:
        assert "IF NOT EXISTS" in stmt


def test_relationship_types_are_distinct():
    rels = {PLAYS_IN, HEAD_TO_HEAD}
    assert len(rels) == 2


# --------------------------------------------------------------- real Neo4j round-trip


@pytest.fixture(scope="module")
def neo4j_client():
    client = get_neo4j_client()
    if not client.ready:
        pytest.skip("Neo4j not ready — start it with `docker compose up neo4j`")
    return client


@pytest.mark.integration
def test_head_to_head_round_trip(neo4j_client):
    neo4j_client.run(
        "MERGE (a:Team {name: 'Testland'}) MERGE (b:Team {name: 'Testopia'}) "
        "MERGE (a)-[r:HEAD_TO_HEAD {match_id: 999999001}]->(b) "
        "SET r.home_score = 2, r.away_score = 1, r.match_date = '2000-01-01'"
    )
    try:
        rows = neo4j_client.get_head_to_head("Testland", "Testopia")
        assert rows and rows[0]["home_score"] == 2
    finally:
        neo4j_client.run(
            "MATCH (t:Team) WHERE t.name IN ['Testland', 'Testopia'] DETACH DELETE t"
        )
