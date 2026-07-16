"""
ml/embedding_model.py
======================
Single shared `all-MiniLM-L6-v2` instance for every agent that embeds text
(match_intel_agent, tactical_agent, briefing_agent, narrative_arc_agent).

Previously each agent module lazily instantiated its own copy of the same
model — four separate ~68MB weight sets (~270MB RSS) doing identical work,
since sentence-transformers' SentenceTransformer holds no per-caller state.
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
