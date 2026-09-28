"""
Shared `all-MiniLM-L6-v2` instance for every agent that embeds text.

Used by match_intel_agent, tactical_agent, briefing_agent and
narrative_arc_agent.
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info("Loading all-MiniLM-L6-v2 (first use, shared across agents)…")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embedding model ready")
    return _embed_model
