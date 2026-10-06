"""
Narrative arc synthesis for narrative spikes.

Generates a short arc for each spike from the per-source signal, similar
past arcs in Weaviate and the live match context, with a template fallback.

Design:
    - The prompt discloses when any source value is synthetic (mock data).
    - The template fallback describes only the observed signal.
"""

from __future__ import annotations

import logging
from typing import List

from agents.arc_prompts import BANNED_PHRASES, _build_prompt, template_arc
from agents.narrative_spike_detector import NarrativeSpike
from agents.ollama_client import generate_with_source
from agents.rag import retrieve
from agents.weaviate_client import NARRATIVE_ARCS

from config import ARC_TOP_K

log = logging.getLogger(__name__)


def _build_query(spike: NarrativeSpike) -> str:
    return (
        f"{spike.topic} World Cup narrative arc tournament storyline "
        f"{spike.summary} historical precedent momentum shift"
    )


def sanitize_arc(text: str, max_chars: int = 780) -> str:
    text = text.strip().strip('"').strip()
    for prefix in ("Here's", "Here is", "Sure,", "Sure!", "Certainly,"):
        if text.startswith(prefix):
            colon_idx = text.find(":")
            if 0 < colon_idx < 60:
                text = text[colon_idx + 1 :].strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_boundary = max(
        truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?")
    )
    if last_boundary > max_chars * 0.5:
        return truncated[: last_boundary + 1]
    return truncated.rstrip() + "…"


def _violates_rules(text: str) -> bool:
    """True if a generated arc uses banned phrasing (falls back to template)."""
    low = text.lower()
    return any(bp in low for bp in BANNED_PHRASES)


async def synthesise(spike: NarrativeSpike) -> str:
    rag_docs: List[str] = []
    try:
        rag_docs = await retrieve(_build_query(spike), NARRATIVE_ARCS, ARC_TOP_K)
    except Exception as e:
        log.warning(f"Arc RAG retrieval failed for {spike.topic}: {e}")

    prompt = _build_prompt(spike, rag_docs)

    try:
        arc_text, served_by = await generate_with_source(prompt, timeout=20.0)
    except Exception as e:
        log.warning(f"Arc generation failed: {e}")
        arc_text, served_by = "", ""

    if not arc_text:
        arc_text = template_arc(spike)
        via = "template"
    else:
        arc_text = sanitize_arc(arc_text)
        if _violates_rules(arc_text):
            log.info(
                f"Arc for {spike.topic} rejected (banned phrasing / parroted "
                f"precedent) — using template"
            )
            arc_text = template_arc(spike)
            via = "template(guard)"
        else:
            via = served_by or "llm"

    log.info(
        f"Arc synthesised for {spike.topic} spike {spike.spike_id} "
        f"via={via} rag={len(rag_docs)} chars={len(arc_text)}"
    )

    return arc_text
