"""
LangGraph orchestration for match-intel narration.

    retrieve ─▶ generate ─▶ check ──▶ END             grounded
       │           ▲          │
       │           └──────────┤                        retry once, violations fed back
       │                      ▼
       └────────────────▶ template ─▶ END             LLM off/down, or still ungrounded

One compiled graph serves event reactions, periodic colour and the FT wrap-up.
Each caller passes a NarrationSpec (query, prompt builder, template) as the
graph's runtime context. The LLM priority gate sits inside
generate_with_source; the caller's llm_priority contextvar carries into the
node tasks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.grounding import violations
from agents.ollama_client import generate_with_source
from agents.weaviate_client import get_weaviate_client
from api.schemas.schema import MatchState
from ml.embedding_model import get_embed_model
from ml.executors import EMBED_EXECUTOR
from monitoring.metrics import INTEL_NARRATIONS

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # first draft + one grounded retry


@dataclass(frozen=True)
class NarrationSpec:
    kind: str  # "event" | "colour" | "ft_summary" — metrics label
    state: MatchState
    query: str
    collection: str
    build_prompt: Callable[[List[str]], str]
    template: Callable[[], str]
    event_filter: Optional[str] = None
    ref_minutes: Sequence[Optional[int]] = ()
    use_llm: bool = True


class NarrationState(TypedDict, total=False):
    use_llm: bool
    rag_docs: List[str]
    prompt: str
    narrative: str
    via: str
    attempts: int
    violations: List[str]


def facts(prompt: str) -> str:
    """Data section of an intel prompt (everything before the instructions)."""
    return prompt.rsplit("\n\n", 1)[0].replace("[INST]", "").strip()


def with_feedback(prompt: str, found: List[str]) -> str:
    note = (
        " Your previous draft was rejected for unsupported claims: "
        + "; ".join(found[:5])
        + ". Rewrite it using only the facts above. "
    )
    return prompt.replace("[/INST]", note + "[/INST]")


async def _retrieve(state: NarrationState, runtime: Runtime[NarrationSpec]) -> dict:
    spec = runtime.context
    try:
        model = get_embed_model()
        qv = await asyncio.get_running_loop().run_in_executor(
            EMBED_EXECUTOR, lambda: model.encode(spec.query, normalize_embeddings=True).tolist()
        )
        docs = await asyncio.to_thread(
            get_weaviate_client().hybrid_search,
            query_vector=qv,
            query_text=spec.query,
            top_k=5,
            event_filter=spec.event_filter,
            collection=spec.collection,
        )
    except Exception as exc:
        log.debug(f"[{spec.state.fixture_id}] {spec.kind} RAG failed: {exc}")
        docs = []
    return {"use_llm": spec.use_llm, "rag_docs": docs or [], "attempts": 0, "violations": []}


async def _generate(state: NarrationState, runtime: Runtime[NarrationSpec]) -> dict:
    spec = runtime.context
    prompt = spec.build_prompt(state["rag_docs"])
    ask = with_feedback(prompt, state["violations"]) if state["violations"] else prompt
    try:
        text, via = await generate_with_source(ask)
    except Exception as exc:
        log.debug(f"[{spec.state.fixture_id}] {spec.kind} LLM failed: {exc}")
        text, via = "", None
    return {
        "prompt": prompt,
        "narrative": text or "",
        "via": via or "",
        "attempts": state["attempts"] + 1,
        "violations": [],
    }


def _check(state: NarrationState, runtime: Runtime[NarrationSpec]) -> dict:
    if not state["narrative"]:
        return {}
    spec = runtime.context
    found = violations(
        state["narrative"], facts(state["prompt"]), spec.state, state["rag_docs"], spec.ref_minutes
    )
    if found:
        log.warning(f"[{spec.state.fixture_id}] {spec.kind} draft {state['attempts']} ungrounded: {found}")
    return {"violations": found}


def _template(state: NarrationState, runtime: Runtime[NarrationSpec]) -> dict:
    return {"narrative": runtime.context.template(), "via": "template"}


def _after_retrieve(state: NarrationState) -> str:
    return "generate" if state["use_llm"] else "template"


def _after_check(state: NarrationState) -> str:
    if not state["narrative"]:
        return "template"
    if not state["violations"]:
        return END
    return "generate" if state["attempts"] < MAX_ATTEMPTS else "template"


def build_graph():
    g = StateGraph(NarrationState, context_schema=NarrationSpec)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("check", _check)
    g.add_node("template", _template)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", _after_retrieve, ["generate", "template"])
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", _after_check, ["generate", "template", END])
    g.add_edge("template", END)
    return g.compile(name="match_intel_narration")


GRAPH = build_graph()


def _outcome(out: NarrationState) -> str:
    if out["via"] != "template":
        return "grounded_first" if out["attempts"] == 1 else "grounded_retry"
    if not out["use_llm"]:
        return "template_only"
    return "template_ungrounded" if out.get("violations") else "template_llm_down"


async def narrate(spec: NarrationSpec) -> NarrationState:
    out = await GRAPH.ainvoke({}, context=spec)
    INTEL_NARRATIONS.labels(spec.kind, _outcome(out)).inc()
    return out
