"""
Match-intel LangGraph pipeline tests.

Covers the grounding check and every graph path: grounded, retried,
ungrounded, LLM down and template-only. LLM, embedder and Weaviate are
faked; no services needed.
"""

import asyncio

import pytest

from agents import intel_graph, match_intel_agent as mi
from agents.grounding import _roster, violations
from agents.intel_graph import NarrationSpec, facts, narrate
from agents.llm_queue import Priority, current_priority, llm_priority
from api.schemas.schema import MatchEvent, MatchState, TeamStats
from api.workers.match_producer import build_state
from feeds import snapshot


def _run(coro):
    return asyncio.run(coro)


def _state(**kw) -> MatchState:
    base = dict(
        fixture_id=1, status_short="FT", elapsed=90,
        home_id=1, home_name="Brazil", home_score=2,
        away_id=2, away_name="Japan", away_score=1,
        home_stats=TeamStats(possession=61.0, expected_goals=1.93),
        away_stats=TeamStats(possession=39.0, expected_goals=0.44),
        events=[
            MatchEvent(elapsed=20, team_id=1, team_name="Brazil", player_name="Casemiro", type="goal"),
            MatchEvent(elapsed=56, team_id=2, team_name="Japan", player_name="Kaoru Mitoma", type="goal"),
            MatchEvent(elapsed=90, team_id=1, team_name="Brazil", player_name="Casemiro", type="goal"),
        ],
    )
    return MatchState(**{**base, **kw})


FACTS = (
    "At minute 56', a moment shifted the game: goal by Kaoru Mitoma (Japan). "
    "The score after it: Brazil 1–1 Japan. Current match totals: xG 1.93 vs 0.44, "
    "possession 61% vs 39%. Win probability shift: Japan 5% → 24%. This match has finished.\n"
    "• FIFA World Cup 2022 — Croatia vs Japan — Minute 43 — Goal\n"
    "Situation: Score was 0-0, Japan scored to make it 0-1."
)


def _v(text, state=None, docs=(FACTS,)):
    return violations(text, FACTS, state or _state(), docs, (56, 90))


# ── grounding check ────────────────────────────────────────────────────────


def test_grounded_text_passes():
    text = (
        "Kaoru Mitoma levelled it 1-1 at 56', lifting Japan's win probability from 5% "
        "to 24%, a 19-point swing. Brazil still led the xG 1.9 to 0.44, and 34 minutes "
        "remained; Croatia vs Japan saw a similar 0-1 goal in 2022."
    )
    assert _v(text) == []


@pytest.mark.parametrize(
    "text, kind",
    [
        ("Japan's win probability jumped from 5% to 67%.", "67%"),
        ("Brazil's xG rose from 1.58 to 1.93.", "1.58"),
        ("Japan now trail 3-1.", "scoreline 3-1"),
        ("Just like Argentina in 2014.", "team Argentina"),
        ("Echoes of Lionel Messi's late winner.", "player Lionel Messi"),
        ("Japan scored again at 71'.", "minute 71"),
        ("Level with 45 minutes remaining.", "45 minutes remaining"),
    ],
)
def test_fabrications_are_flagged(text, kind):
    found = _v(text)
    assert any(kind in f for f in found), found


def test_formation_and_true_progression_not_flagged():
    # 2-1 is the final score, 1-0 an intermediate one; 4-2-3-1 isn't a score.
    assert _v("Brazil's 4-2-3-1 went 1-0 up and won 2-1.") == []


def test_roster_has_snapshot_players():
    assert "Lionel Messi" in _roster().values()


# ── graph paths ────────────────────────────────────────────────────────────


class _Model:
    def encode(self, text, normalize_embeddings=True):
        import numpy as np

        return np.zeros(4)


class _Weaviate:
    def __init__(self, docs=None, fail=False):
        self.docs, self.fail = docs or [], fail

    def hybrid_search(self, **kw):
        if self.fail:
            raise RuntimeError("weaviate down")
        return self.docs


@pytest.fixture
def fake_llm(monkeypatch):
    """Returns queued drafts in order and records prompts and priorities."""
    calls = {"prompts": [], "priorities": [], "drafts": []}

    async def gen(prompt):
        calls["prompts"].append(prompt)
        calls["priorities"].append(current_priority())
        text = calls["drafts"].pop(0) if calls["drafts"] else ""
        return text, ("ollama" if text else None)

    monkeypatch.setattr(intel_graph, "generate_with_source", gen)
    monkeypatch.setattr(intel_graph, "get_embed_model", lambda: _Model())
    monkeypatch.setattr(intel_graph, "get_weaviate_client", lambda: _Weaviate([FACTS.split("\n", 1)[1]]))
    return calls


def _spec(**kw):
    base = dict(
        kind="event",
        state=_state(),
        query="goal minute 56",
        collection="NarrativeArcs",
        build_prompt=lambda docs: f"[INST] {FACTS.split(chr(10))[0]}\n" + "\n".join(docs)
        + "\n\nWrite 2 sentences. [/INST]",
        template=lambda: "TEMPLATE",
        ref_minutes=(56, 90),
    )
    return NarrationSpec(**{**base, **kw})


GOOD = "Kaoru Mitoma made it 1-1, Japan's win probability rising from 5% to 24%."
BAD = "Kaoru Mitoma made it 1-1, Japan's win probability rising from 5% to 67%."


def test_grounded_first_draft(fake_llm):
    fake_llm["drafts"] = [GOOD]
    out = _run(narrate(_spec()))
    assert (out["narrative"], out["via"], out["attempts"]) == (GOOD, "ollama", 1)
    assert len(out["rag_docs"]) == 1


def test_retry_feeds_violations_back(fake_llm):
    fake_llm["drafts"] = [BAD, GOOD]
    out = _run(narrate(_spec()))
    assert (out["narrative"], out["via"], out["attempts"]) == (GOOD, "ollama", 2)
    first, second = fake_llm["prompts"]
    assert "rejected" not in first
    assert "rejected for unsupported claims: 67% not in facts" in second
    assert second.endswith("[/INST]")


def test_still_ungrounded_falls_back_to_template(fake_llm):
    fake_llm["drafts"] = [BAD, BAD, GOOD]
    out = _run(narrate(_spec()))
    assert (out["narrative"], out["via"], out["attempts"]) == ("TEMPLATE", "template", 2)
    assert out["violations"]
    assert len(fake_llm["prompts"]) == intel_graph.MAX_ATTEMPTS


def test_llm_down_goes_straight_to_template(fake_llm):
    out = _run(narrate(_spec()))
    assert (out["via"], out["attempts"]) == ("template", 1)


def test_template_only_skips_llm(fake_llm):
    out = _run(narrate(_spec(use_llm=False)))
    assert out["via"] == "template" and fake_llm["prompts"] == []


def test_rag_failure_still_generates(fake_llm, monkeypatch):
    monkeypatch.setattr(intel_graph, "get_weaviate_client", lambda: _Weaviate(fail=True))
    fake_llm["drafts"] = [GOOD]
    out = _run(narrate(_spec()))
    assert out["rag_docs"] == [] and out["via"] == "ollama"


def test_caller_priority_reaches_llm(fake_llm):
    fake_llm["drafts"] = [GOOD]

    async def main():
        with llm_priority(Priority.LIVE):
            await narrate(_spec())

    _run(main())
    assert fake_llm["priorities"] == [Priority.LIVE]


def test_facts_excludes_instructions():
    p = "[INST] Score 1-1. 5% → 24%.\n\nWrite 2-3 sentences in the next 5 minutes. [/INST]"
    assert facts(p) == "Score 1-1. 5% → 24%."


def test_analyze_event_end_to_end_on_snapshot(fake_llm):
    """Real snapshot match through the production event path: a fabricated
    first draft is retried, and the entry carries the grounded retry."""
    f = next(f for f in snapshot.fixtures() if f["status"] == "FT" and snapshot.match_detail(f["fixture_id"]))
    state = build_state(f, snapshot.match_detail(f["fixture_id"]))
    ev = next(e for e in state.events if e.type == "goal")
    fake_llm["drafts"] = [f"{state.home_name} lead 7-5 now.", f"{state.home_name} and {state.away_name} play on."]

    entry = _run(mi.analyze_event(state, ev, None))
    assert entry["via"] == "ollama"
    assert entry["narrative"] == f"{state.home_name} and {state.away_name} play on."
    assert "scoreline 7-5" in fake_llm["prompts"][1]


def test_event_template_keeps_name_case():
    state = _state()
    text = mi._event_template(state, state.events[1], completed=True)
    assert text.startswith("Goal by Kaoru Mitoma (Japan) at 56' makes it 1–1.")
