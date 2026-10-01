"""
Pre-match briefing agent.

Generates a Groq briefing grounded only in facts computed as of kickoff
(agents/briefing_facts.py): pre-match probabilities, point-in-time Elo,
each team's earlier results at this World Cup and head-to-head history.
Briefings requested mid-match or after full time never see the result.

Fallbacks:
    - No GROQ_API_KEY or an API error: template built from the same facts.
    - Neo4j head-to-head: used only for fixtures outside the snapshot.
"""

import asyncio
import logging
import os
from typing import List, Optional

import httpx

from agents.briefing_facts import briefing_facts, render
from agents.langsmith_tracing import traceable
from agents.ollama_client import groq_chat
from ml.odds_api_client import get_oddsapi_client

log = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def _model_label() -> str:
    return f"{GROQ_MODEL} via Groq"


def _graph_context(home_name: str, away_name: str) -> List[str]:
    """Prior WC meetings from the Neo4j graph; [] if Neo4j is down or none exist."""
    try:
        from kg.neo4j_client import get_neo4j_client

        kg = get_neo4j_client()
        if not kg.ready:
            return []
        meetings = kg.get_head_to_head(home_name, away_name, limit=3)
        lines = []
        for m in meetings:
            if m.get("team_a") == home_name:
                hs, as_ = m.get("home_score"), m.get("away_score")
            else:
                hs, as_ = m.get("away_score"), m.get("home_score")
            lines.append(
                f"{home_name} {hs}-{as_} {away_name} ({m.get('competition', 'WC')} "
                f"{m.get('season', '')})"
            )
        return lines
    except Exception as exc:
        log.warning(f"Briefing graph context failed: {exc}")
        return []


@traceable(name="briefing_agent.generate", run_type="chain")
async def generate(
    home_name: str,
    away_name: str,
    competition: str = "WC 2026",
    fixture_id: Optional[int] = None,
) -> tuple[str, str]:
    """Return (briefing_text, model_label); label is "template" without an LLM."""
    try:
        odds_table = await get_oddsapi_client().get_all_odds()
    except Exception:
        odds_table = None
    facts = briefing_facts(fixture_id, home_name, away_name, odds_table)
    if not facts["h2h"]:
        graph = await asyncio.to_thread(_graph_context, home_name, away_name)
        facts_text = render(facts, home_name, away_name)
        if graph:
            facts_text += "\nPrevious World Cup meetings: " + "; ".join(graph) + "."
    else:
        facts_text = render(facts, home_name, away_name)

    if not _groq_key():
        log.warning("No GROQ_API_KEY — using template briefing")
        return _template(home_name, away_name, facts_text), "template"

    prompt = (
        f"Write a pre-match briefing for {home_name} vs {away_name} at the "
        f"{competition}, in 3 sentences:\n"
        f"1. Who is favoured and by how much.\n"
        f"2. The one statistical contrast from their tournament so far that "
        f"matters most (chance creation, chances conceded, or control of the ball).\n"
        f"3. What would have to change for the underdog, citing a number.\n\n"
        f"FACTS (as of kickoff):\n{facts_text}\n\n"
        f"Every number you state must appear in FACTS. Do not mention pressing, "
        f"formations, injuries or players — none are in FACTS. Do not invent "
        f"previous meetings. No cliches."
    )

    try:
        text = await groq_chat(
            [
                {
                    "role": "system",
                    "content": "You are a precise pre-match football analyst. "
                    "You only use the numbers you are given.",
                },
                {"role": "user", "content": prompt},
            ],
            GROQ_MODEL,
            max_tokens=200,
            temperature=0.4,
            timeout=12.0,
        )
        if not text:
            return _template(home_name, away_name, facts_text), "template"
        log.info(f"Briefing for {home_name} vs {away_name} ({len(text)} chars)")
        return text, _model_label()

    except httpx.HTTPStatusError as exc:
        log.warning(
            f"Groq briefing HTTP {exc.response.status_code}: "
            f"{exc.response.text[:200]}"
        )
    except Exception as exc:
        log.warning(f"Groq briefing error: {exc}")
    return _template(home_name, away_name, facts_text), "template"


def _template(home_name: str, away_name: str, facts_text: str) -> str:
    """Fallback: the facts themselves, no invented claims."""
    return f"{home_name} vs {away_name} — pre-match facts. " + facts_text.replace("\n", " ")
