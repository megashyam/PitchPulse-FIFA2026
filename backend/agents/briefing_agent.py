"""Pre-match tactical briefing engine powered by grounded LLM context generation.

Orchestrates multi-modal retrieval-augmented generation (RAG) by fetching historical
context from a Weaviate `NarrativeArcs` collection using hybrid search (combining dense
vectors with sparse BM25 keywords) and synthesizing insights via Groq inference.

Key Architectural Implementations:
    - Resilient Context Initialization: Validates `GROQ_API_KEY` and resolves target
      LLM configurations dynamically at call time to guarantee smooth environment isolation.
    - Asynchronous Resource Allocation: Offloads computational embedding routines to a
      dedicated `EMBED_EXECUTOR` thread pool, preventing blocking operations on the
      primary asyncio event loop.
    - Graceful Degradation: Implements strict fallback protocols. In the absence of upstream
      database connectivity or LLM provider credentials, the system degrades to deterministic,
      factually conservative templates to eliminate model hallucination.
"""

import asyncio
import logging
import os
from typing import List, Optional

import httpx

from agents.ollama_client import _clean
from ml.embedding_model import get_embed_model as _get_embed_model
from ml.executors import EMBED_EXECUTOR

log = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE = "https://api.groq.com/openai/v1"


def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def _model_label(with_rag: bool) -> str:
    base = f"{GROQ_MODEL} via Groq"
    return f"{base} + RAG" if with_rag else base


async def _retrieve_context(
    home_name: str,
    away_name: str,
    competition: str,
    loop: asyncio.AbstractEventLoop,
) -> List[str]:
    """Hybrid-search NarrativeArcs for historical WC situations relevant to
    this fixture. Returns up to 3 doc strings, or [] if Weaviate is down."""
    try:
        from agents.weaviate_client import NARRATIVE_ARCS, get_weaviate_client

        wv = get_weaviate_client()
        if not wv.ready:
            return []

        query = (
            f"{home_name} versus {away_name} {competition} tactical matchup "
            f"pressing high press knockout bracket historical precedent"
        )
        model = _get_embed_model()
        vec = await loop.run_in_executor(
            EMBED_EXECUTOR,
            lambda: model.encode(query, normalize_embeddings=True).tolist(),
        )
        return wv.hybrid_search(
            query_vector=vec,
            query_text=query,
            top_k=3,
            collection=NARRATIVE_ARCS,
        )
    except Exception as exc:
        log.warning(f"Briefing RAG retrieval failed: {exc}")
        return []


async def generate(
    home_name: str,
    away_name: str,
    competition: str = "WC 2026",
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> tuple[str, str]:
    """Returns (briefing_text, model_label). model_label distinguishes whether
    RAG context was actually used, and 'template' when no Groq key is set or
    the API call failed."""
    loop = loop or asyncio.get_running_loop()
    rag_docs = await _retrieve_context(home_name, away_name, competition, loop)

    if not _groq_key():
        log.warning("No GROQ_API_KEY — using template briefing")
        return _template(home_name, away_name, rag_docs), "template"

    rag_section = ""
    if rag_docs:
        rag_section = (
            "\n\nHISTORICAL PRECEDENT — similar WC situations from the archive:\n"
            + "\n".join(f"• {d}" for d in rag_docs)
            + "\n\nGround your historical reference in the precedent above. "
            "Do not invent precedents that are not in the archive.\n"
        )

    prompt = (
        f"Write a gripping, high-stakes detailed pre-match tactical briefing for {home_name} vs {away_name} "
        f"at the {competition}. "
        f"1. The Fault Line: Identify the core pressing matchup and aggressively state which team's high press will crack first.\n"
        f"2. The Decider: Name one highly specific tactical variable (with numbers if possible) that will decide the match.\n"
        f"3. The Ghost of the Past: ONLY if precedent context is provided below, use it as a stark warning of what happens when this tactical variable goes wrong. If no precedent is provided, finish with a bold prediction.\n"
        f"Do not use generic words like 'crucial', 'significant', or 'battle'. Sound like a ruthless, elite scout. "
        f"{rag_section}"
    )

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                f"{GROQ_BASE}/chat/completions",
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a sharp pre-match football analyst. "
                            "Be specific, use numbers, avoid clichés. Never invent "
                            "historical facts.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 180,
                    "temperature": 0.70,
                },
                headers={
                    "Authorization": f"Bearer {_groq_key()}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            text = _clean(resp.json()["choices"][0]["message"]["content"])
            if not text:
                return _template(home_name, away_name, rag_docs), "template"
            label = _model_label(with_rag=bool(rag_docs))
            log.info(
                f"Briefing for {home_name} vs {away_name} "
                f"({len(text)} chars, rag={len(rag_docs)})"
            )
            return text, label

    except httpx.HTTPStatusError as exc:
        log.warning(
            f"Groq briefing HTTP {exc.response.status_code}: "
            f"{exc.response.text[:200]}"
        )
        return _template(home_name, away_name, rag_docs), "template"
    except Exception as exc:
        log.warning(f"Groq briefing error: {exc}")
        return _template(home_name, away_name, rag_docs), "template"


def _template(
    home_name: str, away_name: str, rag_docs: Optional[List[str]] = None
) -> str:
    """Honest fallback — states what to watch without asserting facts we do
    not have. If real archive precedent was retrieved, quote its source line;
    never invent one."""
    base = (
        f"{home_name} vs {away_name}: watch the pressing matchup — which side "
        f"sustains its press deeper into the half usually decides territorial "
        f"control, and press-bypass pass rate is the variable to track."
    )
    if rag_docs:
        first_line = rag_docs[0].splitlines()[0][:120]
        base += f" Archive precedent on file: {first_line}."
    else:
        base += (
            " No language model or archive precedent was available for this briefing."
        )
    return base
