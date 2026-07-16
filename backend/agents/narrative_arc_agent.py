"""
Engaging-voice arc synthesis for narrative spikes.

"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List

from agents.narrative_spike_detector import NarrativeSpike
from agents.ollama_client import generate
from agents.weaviate_client import get_weaviate_client, NARRATIVE_ARCS
from ml.embedding_model import get_embed_model as _get_embed_model
from ml.executors import EMBED_EXECUTOR

log = logging.getLogger(__name__)


def _build_query(spike: NarrativeSpike) -> str:
    return (
        f"{spike.topic} World Cup narrative arc tournament storyline "
        f"{spike.summary} historical precedent momentum shift"
    )


BANNED_PHRASES = [
    "indicates",
    "indicate",
    "indicating",
    "significant",
    "notable",
    "noteworthy",
    "comparable to",
    "heightened interest",
    "it is worth noting",
    "in conclusion",
    "overall",
    "furthermore",
    "this event",
    "this spike",
    "this suggests",
    "this pattern",
    "crucial",
    "exciting",
]

FEW_SHOT_EXAMPLES = """
GOOD (leads with a number, gives a probable cause reasoned from which
sources moved and which didn't — no match data needed, just signal shape):
"86 posts a minute on Mastodon right now — triple the pre-tournament
baseline, with almost no matching move on Bluesky or Wikipedia. That
lopsided pattern is the signature of a single viral post or clip getting
shared rapidly on one platform, not a broad multi-source reaction like a
goal or a major news story would produce."

GOOD (a comparison that draws a real inference from the signal pattern):
"Wikipedia edits on the squad page jumped to 4.2/min with Mastodon and
Bluesky both still near baseline — that mismatch usually means a lineup
or injury story breaking on a slower news cycle, since fan chatter would
normally lead a Wikipedia spike, not lag behind it like this one does."

BAD — do not write like this:
"The recent social signal spike for WC2026 on Reddit indicates a
heightened interest in the upcoming tournament among this community.
This is comparable to the previous historical precedent, which also
experienced a significant spike in social signals."
""".strip()


def _build_prompt(spike: NarrativeSpike, rag_docs: List[str]) -> str:
    """
    Construct the LLM instruction prompt for generating a narrative arc.

    Formats live signal metrics, source movement patterns, historical context,
    and generation constraints into a structured prompt that guides the model
    toward evidence-based narrative analysis.

    Args:
        spike (NarrativeSpike): Narrative spike containing detected signals,
            severity, and source metadata.
        rag_docs (List[str]): Retrieved historical narrative examples used as
            contextual references.

    Returns:
        str: Fully formatted prompt for narrative generation.
    """
    signal_lines = []
    s = spike.sources
    if s.get("mastodon", 0) > 5:
        signal_lines.append(f"Mastodon: {s['mastodon']:.0f} posts/min")
    if s.get("bluesky", 0) > 3:
        signal_lines.append(f"Bluesky: {s['bluesky']:.0f} mentions/min")
    if s.get("trends", 0) > 30:
        signal_lines.append(f"Google Trends: {s['trends']:.0f}/100")
    if s.get("wikipedia", 0) > 0.5:
        signal_lines.append(f"Wikipedia: {s['wikipedia']:.1f} edits/min")
    signal_str = "; ".join(signal_lines) if signal_lines else spike.summary

    precedent = ""
    if rag_docs:
        precedent = (
            "\n\nHistorical precedent from WC archive (for color/comparison only — don't just restate it):\n"
            + "\n".join(f"• {d[:200]}" for d in rag_docs[:2])
        )

    driving = set(spike.source_names or [])
    all_four = {"mastodon", "bluesky", "trends", "wikipedia"}
    quiet = all_four - driving
    shape_note = (
        f"Moved: {', '.join(sorted(driving)) or 'none clearly'}. "
        f"Stayed near baseline: {', '.join(sorted(quiet)) or 'none — all four moved together'}."
    )

    mock_note = ""
    if spike.data_sources:
        mock_sources = [s for s, kind in spike.data_sources.items() if kind == "mock"]
        if mock_sources:
            mock_note = (
                f"\nNOTE: {', '.join(mock_sources)} signal is SIMULATED placeholder "
                f"data (the live API wasn't available), not a real measurement — "
                f"describe the pattern without implying it's a confirmed real-world number.\n"
            )

    return (
        f"[INST] You are a sharp, cynical digital culture trend-spotter for a live sports intelligence desk. "
        f"Your job is to cut through the noise and tell the audience if a social spike is a real-world event or just a viral meme.\n\n"
        f"SPIKE: topic '{spike.topic}', severity {spike.severity:.0%} above baseline.\n"
        f"Signal: {signal_str}.\n"
        f"Source pattern: {shape_note}"
        f"{mock_note}"
        f"{precedent}\n\n"
        f"HARD RULES:\n"
        f"- Write 4-5 sentences — enough room to actually develop the "
        f"reasoning, not a compressed summary.\n"
        f"- Sentence 1: open with the single most interesting number or claim — "
        f"never the topic name or 'this spike/event'.\n"
        f"- Sentences 2-3: explain the PROBABLE CAUSE using the source pattern "
        f"above. A spike concentrated in ONE source (others near baseline) "
        f"usually means a single viral post/clip, not a broad event. A spike "
        f"where MULTIPLE sources moved together usually means something real "
        f"happened (goal, injury, controversy) that's generating reaction "
        f"across platforms simultaneously. Reason explicitly from which "
        f"sources moved and which didn't — that pattern IS the evidence.\n"
        f"- Sentence 4-5: add a second layer — what would confirm or "
        f"contradict this read (e.g. 'if Wikipedia edits follow in the next "
        f"tick, that points to X; if it stays Mastodon-only, it's more likely "
        f"Y'), or a specific comparison to how a different kind of spike "
        f"usually looks. Give the reader something to watch for, not just a "
        f"restated fact.\n"
        f"- Never invent statistics not given above (win rates, historical "
        f"percentages, etc.) — only reason from the signal numbers provided.\n"
        f"- NEVER use these words: {', '.join(BANNED_PHRASES)}.\n"
        f"- Active voice, present tense. Sound like someone who actually finds "
        f"this pattern interesting, not like a template filling in blanks.\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Now write ONE 4-5 sentence take about the spike above, in that voice. "
        f"Output ONLY the take — no preamble. [/INST]"
    )


def sanitize_arc(text: str, max_chars: int = 780) -> str:
    """
    Clean and normalize generated narrative text before returning it.

    Removes common model preambles, trims unnecessary formatting, and truncates
    long outputs while preserving sentence boundaries when possible.

    Args:
        text (str): Raw generated narrative text.
        max_chars (int, optional): Maximum allowed output length in characters.

    Returns:
        str: Sanitized narrative arc text.
    """
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
    """
    Check whether generated text contains prohibited phrases.

    Used as a safety guard to prevent the final narrative from including banned
    template-like language or unsupported analytical claims.

    Args:
        text (str): Generated narrative text to validate.

    Returns:
        bool: True if banned phrasing is detected, otherwise False.
    """
    """True if generated text uses banned phrasing."""
    low = text.lower()
    return any(bp in low for bp in BANNED_PHRASES)


def _template_arc(spike: NarrativeSpike) -> str:
    """
    Generate a deterministic fallback narrative from observed signal patterns.

    Creates a rule-based narrative when LLM generation fails or produces output
    that violates quality constraints. The fallback reasons from source movement
    patterns without introducing unsupported statistics.

    Args:
        spike (NarrativeSpike): Narrative spike containing signal metrics and
            source activity information.

    Returns:
        str: Generated fallback narrative arc.
    """
    """Template fallback based on observed per-source signal shape."""
    sev_pct = round(spike.severity * 100)
    s = spike.sources or {}
    driving = spike.source_names or []
    all_four = ["mastodon", "bluesky", "trends", "wikipedia"]
    quiet = [src for src in all_four if src not in driving]

    mock_note = ""
    if spike.data_sources and any(v == "mock" for v in spike.data_sources.values()):
        mock_note = " (signal is partly simulated — live source unavailable)"

    lead_parts = []
    if s.get("mastodon", 0) > 10:
        lead_parts.append(f"{s['mastodon']:.0f} Mastodon posts/min")
    if s.get("bluesky", 0) > 8:
        lead_parts.append(f"{s['bluesky']:.0f} Bluesky mentions/min")
    if s.get("trends", 0) > 50:
        lead_parts.append(f"a Trends index of {s['trends']:.0f}")
    if s.get("wikipedia", 0) > 1:
        lead_parts.append(f"{s['wikipedia']:.1f} Wikipedia edits/min")
    lead = lead_parts[0] if lead_parts else f"{sev_pct}% above its rolling baseline"

    if len(driving) == 1:
        shape_sentence = (
            f"The move is concentrated entirely on {driving[0]}, with "
            f"{', '.join(quiet) or 'every other source'} still flat — that "
            f"one-source pattern usually points to a single post or clip "
            f"spreading fast rather than a broad tournament event."
        )
    elif len(driving) >= 3:
        shape_sentence = (
            f"{', '.join(driving)} are all moving together, which is the "
            f"signature of something real happening — a goal, injury, or "
            f"news story generating reaction across platforms at once "
            f"rather than one viral post."
        )
    elif len(driving) == 2:
        shape_sentence = (
            f"{driving[0]} and {driving[1]} are moving together while "
            f"{', '.join(quiet) or 'the rest'} lag behind — a partial "
            f"multi-source pattern worth watching to see if it broadens."
        )
    else:
        shape_sentence = (
            "No single source clearly leads this move — it reads as a "
            "diffuse lift across the board rather than one platform driving it."
        )

    watch_for = (
        f"If {quiet[0] if quiet else 'the remaining sources'} follows in "
        f"the next tick, that's the stronger signal of a real event; if "
        f"{driving[0] if driving else 'the lead source'} alone fades back "
        f"to baseline, this was likely a single-post spike."
    )

    return (
        f"{spike.topic} is running at {lead} right now{mock_note}, "
        f"{sev_pct}% above its 72-hour baseline. {shape_sentence} {watch_for}"
    )


async def synthesise(
    spike: NarrativeSpike,
    loop: asyncio.AbstractEventLoop,
) -> str:
    """
    Generate a narrative arc for a detected narrative spike.

    Retrieves related historical arcs using vector search, builds an LLM prompt,
    generates and validates the narrative response, and falls back to a
    rule-based template when generation fails. Optionally stores generated arcs
    back into the vector database.

    Args:
        spike (NarrativeSpike): Narrative spike requiring narrative synthesis.
        loop (asyncio.AbstractEventLoop): Event loop used for asynchronous
            embedding execution.

    Returns:
        str: Final synthesized narrative arc.
    """
    wv = get_weaviate_client()
    rag_docs: List[str] = []

    if wv.ready:
        try:
            query = _build_query(spike)
            model = _get_embed_model()
            vec = await loop.run_in_executor(
                EMBED_EXECUTOR,
                lambda: model.encode(query, normalize_embeddings=True).tolist(),
            )
            rag_docs = wv.hybrid_search(
                query_vector=vec,
                query_text=query,
                top_k=3,
                collection=NARRATIVE_ARCS,
            )
        except Exception as e:
            log.warning(f"Arc RAG retrieval failed for {spike.topic}: {e}")

    prompt = _build_prompt(spike, rag_docs)

    try:
        arc_text = await generate(prompt, timeout=20.0)
    except Exception as e:
        log.warning(f"Arc generation failed: {e}")
        arc_text = ""

    if not arc_text:
        arc_text = _template_arc(spike)
        via = "template"
    else:
        arc_text = sanitize_arc(arc_text)
        if _violates_rules(arc_text):
            log.info(
                f"Arc for {spike.topic} rejected (banned phrasing / parroted "
                f"precedent) — using template"
            )
            arc_text = _template_arc(spike)
            via = "template(guard)"
        else:
            via = "mistral"

    log.info(
        f"Arc synthesised for {spike.topic} spike {spike.spike_id} "
        f"via={via} rag={len(rag_docs)} chars={len(arc_text)}"
    )

    if wv.ready and os.getenv("ARC_STOREBACK", "0") == "1":
        try:
            content = (
                f"WC 2026 · {spike.topic} · Narrative Arc · Tick {spike.tick}\n"
                f"Signal: {spike.summary}\n"
                f"Arc: {arc_text}\n"
                f"Severity: {spike.severity:.2f} · Sources: {', '.join(spike.source_names or [])}"
            )
            model = _get_embed_model()
            vec = await loop.run_in_executor(
                EMBED_EXECUTOR,
                lambda: model.encode(content, normalize_embeddings=True).tolist(),
            )
            wv.insert_document(
                collection=NARRATIVE_ARCS,
                properties={
                    "content": content,
                    "match_id": f"narrative:{spike.spike_id}",
                    "competition": "WC 2026",
                    "season": "2026",
                    "minute": 0,
                    "event_type": "narrative_arc",
                },
                vector=vec,
            )
        except Exception as e:
            log.warning(f"Arc store-back failed: {e}")

    return arc_text
