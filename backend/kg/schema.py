"""
Neo4j schema for the WC 2026 knowledge graph.

Teams, groups, bracket rounds and historical StatsBomb matches.
"""

from __future__ import annotations

# Node labels
TEAM = "Team"
GROUP = "Group"
BRACKET_ROUND = "BracketRound"
HISTORICAL_MATCH = "HistoricalMatch"

# Relationship types
PLAYS_IN = "PLAYS_IN"
ADVANCES_TO = "ADVANCES_TO"
HEAD_TO_HEAD = "HEAD_TO_HEAD"

CONSTRAINT_STATEMENTS: list[str] = [
    f"CREATE CONSTRAINT team_name IF NOT EXISTS FOR (t:{TEAM}) REQUIRE t.name IS UNIQUE",
    f"CREATE CONSTRAINT group_code IF NOT EXISTS FOR (g:{GROUP}) REQUIRE g.code IS UNIQUE",
    f"CREATE CONSTRAINT historical_match_id IF NOT EXISTS FOR (m:{HISTORICAL_MATCH}) "
    "REQUIRE m.match_id IS UNIQUE",
]
