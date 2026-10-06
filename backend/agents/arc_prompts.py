"""Prompt and template text for agents.narrative_arc_agent. Pure functions
of a NarrativeSpike; no I/O."""

from __future__ import annotations

from typing import List

from agents.narrative_spike_detector import NarrativeSpike
from agents.rag import precedent_block

SOURCES = ("mastodon", "bluesky", "trends", "wikipedia")

ARC_PRECEDENT_HEADER = (
    "\n\nHistorical precedent from WC archive (for color/comparison only \u2014 "
    "don't just restate it):\n"
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
"86 posts an hour on Mastodon right now — triple the pre-tournament
baseline, with almost no matching move on Bluesky or Wikipedia. That
lopsided pattern is the signature of a single viral post or clip getting
shared rapidly on one platform, not a broad multi-source reaction like a
goal or a major news story would produce."

GOOD (a comparison that draws a real inference from the signal pattern):
"Wikipedia edits on the squad page jumped to 12 an hour with Mastodon and
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
    signal_lines = []
    s = spike.sources
    z = spike.z_scores or {}

    def _z(src: str) -> str:
        return f" ({z[src]:.1f}σ vs 3h baseline)" if z.get(src) is not None else ""

    if s.get("mastodon", 0) > 3:
        signal_lines.append(f"Mastodon: {s['mastodon']:.0f} posts/hr{_z('mastodon')}")
    if s.get("bluesky", 0) > 20:
        signal_lines.append(f"Bluesky: {s['bluesky']:.0f} posts/hr{_z('bluesky')}")
    if s.get("trends", 0) > 1.2:
        signal_lines.append(
            f"Google Trends: {s['trends']:.1f}× the last hour{_z('trends')}"
        )
    if s.get("wikipedia", 0) > 1:
        signal_lines.append(
            f"Wikipedia: {s['wikipedia']:.0f} edits/hr{_z('wikipedia')}"
        )
    signal_str = "; ".join(signal_lines) if signal_lines else spike.summary

    precedent = precedent_block(rag_docs, 2, 200, ARC_PRECEDENT_HEADER)

    driving = set(spike.source_names or [])
    quiet = set(SOURCES) - driving
    shape_note = (
        f"Moved: {', '.join(sorted(driving)) or 'none clearly'}. "
        f"Stayed near baseline: {', '.join(sorted(quiet)) or 'none — all four moved together'}."
    )

    # states when the data is mock
    mock_sources = [src for src, kind in (spike.data_sources or {}).items() if kind == "mock"]
    mock_note = (
        f"\nNOTE: {', '.join(mock_sources)} signal is SIMULATED placeholder "
        f"data (the live API wasn't available), not a real measurement — "
        f"describe the pattern without implying it's a confirmed real-world number.\n"
        if mock_sources
        else ""
    )

    return (
        f"[INST] You are a sharp, cynical digital culture trend-spotter for a live sports intelligence desk. "
        f"Your job is to cut through the noise and tell the audience if a social spike is a real-world event or just a viral meme.\n\n"
        f"SPIKE: topic '{spike.topic}', severity index {spike.severity:.2f} "
        f"(0 = at the alert threshold, 1 = far beyond it; robust z-scores vs a "
        f"rolling 3-hour baseline).\n"
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


def template_arc(spike: NarrativeSpike) -> str:
    """Template arc built from the per-source numbers (which moved, which didn't)."""
    s = spike.sources or {}
    driving = spike.source_names or []
    quiet = [src for src in SOURCES if src not in driving]

    mock_note = ""
    if spike.data_sources and any(v == "mock" for v in spike.data_sources.values()):
        mock_note = " (signal is partly simulated — live source unavailable)"

    # loudest concrete number first
    lead_parts = []
    if s.get("mastodon", 0) > 6:
        lead_parts.append(f"{s['mastodon']:.0f} Mastodon posts/hr")
    if s.get("bluesky", 0) > 50:
        lead_parts.append(f"{s['bluesky']:.0f} Bluesky posts/hr")
    if s.get("trends", 0) > 1.5:
        lead_parts.append(f"Google Trends at {s['trends']:.1f}× the last hour")
    if s.get("wikipedia", 0) > 3:
        lead_parts.append(f"{s['wikipedia']:.0f} Wikipedia edits/hr")
    lead = lead_parts[0] if lead_parts else "an elevated multi-source signal"

    # shape-based rules, same as the LLM prompt
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
        f"severity index {spike.severity:.2f} against its rolling 3-hour baseline. "
        f"{shape_sentence} {watch_for}"
    )
