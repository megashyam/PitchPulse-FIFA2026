"""
Faithfulness eval for match-intel event narration (RAGAS).

Two stages, because ragas pins langchain versions the app can't share:
    generate  (app env) every goal / red card in the WC 2026 snapshot runs
              the production path: event_query → Weaviate hybrid_search →
              _win_prob_swing → _event_prompt → generate_with_source. Keeps
              the raw LLM text and the facts block the model saw. Odds are
              disabled so the prior is Elo (reproducible).
    judge     (backend/.venv-eval) RAGAS faithfulness on a seeded sample,
              judged by an independent Groq model. Verdicts are cached, so
              a rate-limited run resumes.

`generate --graph` runs the same events through agents.intel_graph
(grounding check, one retry, template fallback) for a before/after:
    PYTHONPATH=. python eval/eval_generation_ragas.py generate --graph --out ragas_samples_graph.json
    PYTHONPATH=. .venv-eval/Scripts/python eval/eval_generation_ragas.py judge \
        --samples ragas_samples_graph.json --report ragas_report_graph.json

Faithfulness context is the prompt's data section (event, score, xG,
possession, win-prob shift, retrieved precedent). CIs are a cluster
bootstrap over fixtures.

Run from backend/:
    PYTHONPATH=. python eval/eval_generation_ragas.py generate
    PYTHONPATH=. .venv-eval/Scripts/python eval/eval_generation_ragas.py judge --n 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

SAMPLES = Path("ragas_samples.json")
CACHE = Path(".cache/ragas_judge.jsonl")
REPORT = Path("ragas_report.json")
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
GROQ_BASE = "https://api.groq.com/openai/v1"
N_BOOT = 2000
INST_HEAD = "[INST] You are a football intelligence analyst, not a commentator. "
INST_RULES = "\n\nWrite 2 sentences"


def facts_block(prompt: str) -> str:
    return prompt.split(INST_RULES)[0].replace(INST_HEAD, "").strip()


# ── stage 1 ─────────────────────────────────────────────────────────────────


async def _generate(out: Path, graph: bool = False) -> None:
    from agents import match_intel_agent as mi
    from agents.intel_graph import narrate
    from agents.ollama_client import OLLAMA_MODEL, generate_with_source
    from agents.weaviate_client import NARRATIVE_ARCS, get_weaviate_client
    from api.schemas.event_types import GOAL_TYPES, RED_TYPES
    from api.workers.match_producer import build_state
    from feeds import snapshot
    from ml.embedding_model import get_embed_model

    class _NoOdds:
        async def get_all_odds(self):
            raise RuntimeError("odds disabled for eval")

    mi.get_oddsapi_client = lambda: _NoOdds()

    wv = get_weaviate_client()
    if not wv.ready:
        raise SystemExit("Weaviate not ready — start it and run agents/rag_indexer.py first")
    model = get_embed_model()

    done = {}
    if out.exists():
        done = {r["id"]: r for r in json.loads(out.read_text())["rows"] if r["response"]}

    meta = {
        "generator": (
            f"{OLLAMA_MODEL} via agents.intel_graph (grounding check, 1 retry, template fallback)"
            if graph else f"{OLLAMA_MODEL} via agents.ollama_client.generate_with_source"
        ),
        "prompt": "agents.match_intel_agent._event_prompt, live Weaviate retrieval, Elo prior",
    }
    fixtures = sorted((f for f in snapshot.fixtures() if f.get("kickoff")), key=lambda f: f["kickoff"])
    rows = []
    for f in fixtures:
        detail = snapshot.match_detail(f["fixture_id"])
        if not detail:
            continue
        state = build_state(f, detail)
        completed = state.status_short in ("FT", "AET", "PEN")
        for ev in state.events:
            if ev.type not in GOAL_TYPES | RED_TYPES:
                continue
            sid = f"{state.fixture_id}:{ev.elapsed}:{ev.type}:{ev.team_id}"
            if sid in done:
                rows.append(done[sid])
                continue
            wp = await mi._win_prob_swing(state, ev)
            t0 = time.perf_counter()
            if graph:
                res = await narrate(mi.event_spec(state, ev, wp))
                rag_docs, text, via = res["rag_docs"], res["narrative"], res["via"]
                prompt = mi._event_prompt(state, ev, rag_docs, completed, wp)
                extra = {"attempts": res["attempts"], "violations": res.get("violations", [])}
            else:
                query, ev_filter = mi.event_query(state, ev)
                qv = model.encode(query, normalize_embeddings=True).tolist()
                rag_docs = wv.hybrid_search(
                    query_vector=qv, query_text=query, top_k=5,
                    event_filter=ev_filter, collection=NARRATIVE_ARCS,
                )
                prompt = mi._event_prompt(state, ev, rag_docs, completed, wp)
                text, via = await generate_with_source(prompt)
                extra = {}
            rows.append({
                "id": sid,
                "fixture_id": state.fixture_id,
                "match": f"{state.home_name} vs {state.away_name}",
                "event_type": "goal" if ev.type in GOAL_TYPES else "red_card",
                "minute": ev.elapsed,
                "context": facts_block(prompt),
                "n_precedent_docs": min(2, len(rag_docs)),
                "response": text,
                "via": via,
                "latency_s": round(time.perf_counter() - t0, 2),
                "grounding_violation": bool(text) and mi._grounding_violation(text, state, rag_docs),
                **extra,
            })
            print(f"{len(rows):>4} {sid:<24} {via or 'FAILED':<7} {text[:70]!r}")
            if len(rows) % 10 == 0:  # checkpoint so an interrupted run resumes
                out.write_text(json.dumps({**meta, "rows": rows}, indent=2))
    out.write_text(json.dumps({**meta, "rows": rows}, indent=2))
    ok = [r for r in rows if r["response"]]
    print(f"\n{len(ok)}/{len(rows)} generated, "
          f"{sum(r['grounding_violation'] for r in ok)} grounding violations -> {out}")


# ── stage 2 ─────────────────────────────────────────────────────────────────


def _judge_metric():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set")
    llm = ChatOpenAI(
        model=JUDGE_MODEL, base_url=GROQ_BASE, api_key=key, temperature=0.0,
        max_retries=0, timeout=120, reasoning_effort="low",
    )
    return Faithfulness(llm=LangchainLLMWrapper(llm))


async def _judge_one(metric, row: dict) -> float:
    from ragas import SingleTurnSample

    sample = SingleTurnSample(
        user_input="Write 2 sentences on the impact of this moment.",
        response=row["response"],
        retrieved_contexts=[row["context"]],
    )
    for attempt in range(8):
        try:
            return float(await metric.single_turn_ascore(sample))
        except Exception as exc:
            wait = min(90, 15 * (attempt + 1))
            print(f"  {row['id']}: {type(exc).__name__}: {str(exc)[:120]} — retry in {wait}s")
            await asyncio.sleep(wait)
    raise SystemExit("judge kept failing — rerun later, cached verdicts are kept")


def _boot(rows: list[dict], key: str, rng: np.random.Generator) -> list[float]:
    by_fx = defaultdict(list)
    for r in rows:
        by_fx[r["fixture_id"]].append(r[key])
    sums = np.array([sum(g) for g in by_fx.values()])
    counts = np.array([len(g) for g in by_fx.values()])
    idx = rng.integers(0, len(sums), size=(N_BOOT, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _summary(rows: list[dict], key: str, seed: int) -> dict:
    vals = [r[key] for r in rows]
    return {
        "n": len(rows),
        "mean": round(float(np.mean(vals)), 4),
        "ci95": _boot(rows, key, np.random.default_rng(seed)),
        "share_below_1": round(float(np.mean([v < 1 for v in vals])), 4),
        "share_at_or_below_0.5": round(float(np.mean([v <= 0.5 for v in vals])), 4),
    }


async def _judge(
    n: int, seed: int, pace_s: float, limit: int | None = None,
    samples: Path = SAMPLES, report_path: Path = REPORT,
) -> None:
    data = json.loads(samples.read_text())
    gen = [r for r in data["rows"] if r["response"]]
    sample = random.Random(seed).sample(gen, min(n, len(gen)))
    # Judge only the first `limit` of the fixed n-sample (Groq daily token cap).
    sample = sample[:limit] if limit else sample

    CACHE.parent.mkdir(exist_ok=True)
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            c = json.loads(line)
            if c["judge"] == JUDGE_MODEL:
                cache[(c["id"], c["response"])] = c["faithfulness"]

    metric = _judge_metric()
    for i, r in enumerate(sample, 1):
        k = (r["id"], r["response"])
        if k not in cache:
            cache[k] = await _judge_one(metric, r)
            with CACHE.open("a") as fh:
                fh.write(json.dumps({"id": r["id"], "response": r["response"],
                                     "judge": JUDGE_MODEL, "faithfulness": cache[k]}) + "\n")
            print(f"{i:>4}/{len(sample)} {r['id']:<24} {cache[k]:.3f}")
            await asyncio.sleep(pace_s)
        r["faithfulness"] = cache[k]

    scored = [r for r in sample if not np.isnan(r["faithfulness"])]
    passed = [r for r in scored if not r["grounding_violation"]]
    report = {
        "generator": data["generator"],
        "prompt": data["prompt"],
        "judge": f"RAGAS {_ragas_version()} Faithfulness, {JUDGE_MODEL} on Groq (independent of the generator)",
        "context": "the prompt's data section: event, score, model xG, possession, "
        "win-prob shift, first two retrieved precedent docs",
        "population": {
            "events": len(data["rows"]),
            "generated": len(gen),
            "via": dict(sorted(_count(r["via"] or "failed" for r in data["rows"]).items())),
            "grounding_violation_rate": round(sum(r["grounding_violation"] for r in gen) / len(gen), 4),
        },
        "sample": {"n": len(sample), "planned_n": n, "seed": seed,
                   "rule": "simple random sample of generated events (first n of the planned sample)",
                   "unscorable_nan": len(sample) - len(scored)},
        "ci": f"cluster bootstrap over fixtures, {N_BOOT} resamples, 95% percentile",
        "faithfulness_raw_llm": _summary(scored, "faithfulness", seed),
        "faithfulness_after_grounding_guard": _summary(passed, "faithfulness", seed),
        "by_event_type": {
            t: _summary([r for r in scored if r["event_type"] == t], "faithfulness", seed)
            for t in sorted({r["event_type"] for r in scored})
        },
        "by_precedent_docs": {
            str(k): _summary([r for r in scored if r["n_precedent_docs"] == k], "faithfulness", seed)
            for k in sorted({r["n_precedent_docs"] for r in scored})
        },
        "rows": [
            {k: r[k] for k in ("id", "match", "event_type", "minute", "response",
                               "grounding_violation", "faithfulness")}
            for r in sample
        ],
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("population", "faithfulness_raw_llm",
                                               "faithfulness_after_grounding_guard",
                                               "by_event_type")}, indent=2))


def _count(it) -> dict:
    out: dict = defaultdict(int)
    for x in it:
        out[x] += 1
    return out


def _ragas_version() -> str:
    import ragas

    return ragas.__version__


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", default=str(SAMPLES))
    g.add_argument("--graph", action="store_true", help="narrate via agents.intel_graph")
    j = sub.add_parser("judge")
    j.add_argument("--n", type=int, default=120)
    j.add_argument("--seed", type=int, default=42)
    j.add_argument("--pace", type=float, default=8.0, help="seconds between judge calls (Groq TPM)")
    j.add_argument("--limit", type=int, help="judge only the first K of the --n sample")
    j.add_argument("--samples", default=str(SAMPLES))
    j.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    if args.cmd == "generate":
        asyncio.run(_generate(Path(args.out), args.graph))
    else:
        asyncio.run(_judge(args.n, args.seed, args.pace, args.limit, Path(args.samples), Path(args.report)))


if __name__ == "__main__":
    main()
