"""
Retrieval eval against the live Weaviate NarrativeArcs index.

For each WC 2026 goal or red card, match_intel_agent retrieves WC 2018/2022
StatsBomb docs as precedent. The eval asks whether they are situational
precedent.

    Queries   every goal / red card in the WC 2026 snapshot, built with
              match_intel_agent.event_query and searched with hybrid_search.
    Label     relevant = same event type, same game state for the acting
              team before the event (trailing / level / leading) and same
              minute band (0-30 / 31-60 / 61-90 / 90+). "Loose" drops the
              minute band. Labels come from the corpus situation metadata.
    Split     fixtures by kickoff; alpha tuned on the first half (dev),
              numbers reported on the second half (test).
    CIs       cluster bootstrap over fixtures, 2000 resamples, 95%.
    Baseline  expected precision of a random doc from the same pool.

Requires Weaviate indexed with agents/rag_indexer.py. Run from backend/:
    PYTHONPATH=. python eval/eval_retrieval.py --json retrieval_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

from agents.match_intel_agent import event_query
from agents.rag_indexer import build_corpus, game_state, minute_band
from agents.weaviate_client import HYBRID_ALPHA, NARRATIVE_ARCS, get_weaviate_client
from api.match_timeline import around
from api.schemas.event_types import GOAL_TYPES, RED_TYPES
from api.workers.match_producer import build_state
from feeds import snapshot
from ml.embedding_model import get_embed_model

K = 5  # production top_k
PROMPT_DOCS = 2  # the event prompt shows the first two docs
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_BOOT = 2000


def build_queries() -> list[dict]:
    queries = []
    fixtures = sorted(
        (f for f in snapshot.fixtures() if f.get("kickoff")), key=lambda f: f["kickoff"]
    )
    for order, f in enumerate(fixtures):
        detail = snapshot.match_detail(f["fixture_id"])
        if not detail:
            continue
        state = build_state(f, detail)
        for ev in state.events:
            if ev.type not in GOAL_TYPES | RED_TYPES:
                continue
            before, _ = around(state, ev)
            home_acts = ev.team_id == 1
            if ev.type == "own_goal":  # team_id is the conceding side
                home_acts = not home_acts
            own, other = (
                (before.home_score, before.away_score)
                if home_acts
                else (before.away_score, before.home_score)
            )
            text, ev_filter = event_query(state, ev)
            queries.append(
                {
                    "fixture_id": f["fixture_id"],
                    "order": order,
                    "query": text,
                    "event_type": ev_filter,
                    "state_before": game_state(own, other),
                    "minute_band": minute_band(ev.elapsed + (ev.extra or 0)),
                }
            )
    n_fx = len({q["fixture_id"] for q in queries})
    cut = sorted({q["order"] for q in queries})[n_fx // 2]
    for q in queries:
        q["split"] = "dev" if q["order"] < cut else "test"
    return queries


def relevant(doc: dict, q: dict, loose: bool = False) -> bool:
    s = doc["situation"]
    return (
        doc["event_type"] == q["event_type"]
        and s["state_before"] == q["state_before"]
        and (loose or s["minute_band"] == q["minute_band"])
    )


def score(ranked: list[dict], q: dict, pool: list[dict]) -> dict:
    rel = [relevant(d, q) for d in ranked[:K]]
    loose = [relevant(d, q, loose=True) for d in ranked[:K]]
    n_rel_pool = sum(relevant(d, q) for d in pool)
    first = next((i for i, r in enumerate(rel) if r), None)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(K, n_rel_pool)))
    return {
        "p@2": sum(rel[:PROMPT_DOCS]) / PROMPT_DOCS,
        "p@5": sum(rel) / K,
        "hit@1": float(bool(rel) and rel[0]),
        "mrr@5": 0.0 if first is None else 1 / (first + 1),
        "ndcg@5": dcg / idcg if idcg else 0.0,
        "loose_p@5": sum(loose) / K,
        "random_p@5": n_rel_pool / len(pool) if pool else 0.0,
        "n_relevant_in_pool": n_rel_pool,
    }


METRICS = ["p@2", "p@5", "hit@1", "mrr@5", "ndcg@5", "loose_p@5", "random_p@5"]


def cluster_boot(rows: list[dict], metric: str, rng: np.random.Generator) -> list[float]:
    by_fx = defaultdict(list)
    for r in rows:
        by_fx[r["fixture_id"]].append(r[metric])
    groups = list(by_fx.values())
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def summarise(rows: list[dict], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"n_queries": len(rows), "n_fixtures": len({r["fixture_id"] for r in rows})}
    for m in METRICS:
        out[m] = round(float(np.mean([r[m] for r in rows])), 4)
        out[f"{m}_ci95"] = cluster_boot(rows, m, rng)
    return out


def paired(rows_a: list[dict], rows_b: list[dict], metric: str, seed: int = 0) -> dict:
    """a - b on the same queries, cluster-bootstrapped."""
    diff = [
        {"fixture_id": a["fixture_id"], "d": a[metric] - b[metric]}
        for a, b in zip(rows_a, rows_b)
    ]
    rng = np.random.default_rng(seed)
    return {
        "mean": round(float(np.mean([d["d"] for d in diff])), 4),
        "ci95": cluster_boot(diff, "d", rng),
    }


def run(json_path: str | None) -> dict:
    wv = get_weaviate_client()
    if not wv.ready:
        raise SystemExit("Weaviate not ready — start it and run agents/rag_indexer.py first")

    corpus = build_corpus()
    by_content = {d["content"]: d for d in corpus}
    pools = defaultdict(list)
    for d in corpus:
        pools[d["event_type"]].append(d)
    indexed = wv.get_count(NARRATIVE_ARCS)
    if indexed != len(corpus):
        raise SystemExit(
            f"Index has {indexed} docs, corpus builder {len(corpus)} — re-run rag_indexer.py"
        )

    queries = build_queries()
    model = get_embed_model()
    vecs = model.encode([q["query"] for q in queries], normalize_embeddings=True)

    def evaluate(alpha: float, use_filter: bool = True) -> list[dict]:
        rows = []
        for q, v in zip(queries, vecs):
            objs = wv.hybrid_search(
                query_vector=v.tolist(),
                query_text=q["query"],
                top_k=K,
                event_filter=q["event_type"] if use_filter else None,
                collection=NARRATIVE_ARCS,
                return_objects=True,
                alpha=alpha,
            )
            ranked = [by_content[o["content"]] for o in objs if o["content"] in by_content]
            pool = pools[q["event_type"]] if use_filter else corpus
            rows.append({**q, **score(ranked, q, pool)})
        return rows

    runs = {a: evaluate(a) for a in ALPHAS}
    no_filter = evaluate(HYBRID_ALPHA, use_filter=False)

    def split(rows, name):
        return [r for r in rows if r["split"] == name]

    dev_scores = {a: float(np.mean([r["ndcg@5"] for r in split(runs[a], "dev")])) for a in ALPHAS}
    tuned = max(ALPHAS, key=lambda a: (round(dev_scores[a], 6), -abs(a - HYBRID_ALPHA)))

    test = {a: split(runs[a], "test") for a in ALPHAS}
    prod = test[HYBRID_ALPHA]
    by_type = {
        t: summarise([r for r in prod if r["event_type"] == t])
        for t in sorted({r["event_type"] for r in prod})
    }
    report = {
        "index": f"live Weaviate {NARRATIVE_ARCS}: {indexed} docs from StatsBomb WC 2018/2022 "
        f"({dict((k, len(v)) for k, v in pools.items())})",
        "queries": "every goal / red card in the WC 2026 snapshot, production query builder "
        "and filter (match_intel_agent.event_query)",
        "relevance": "same event type + acting team's game state before the event + minute band; "
        "loose = without minute band",
        "split": {
            "dev": {"queries": len(split(queries, "dev")), "fixtures": len({q["fixture_id"] for q in split(queries, "dev")})},
            "test": {"queries": len(split(queries, "test")), "fixtures": len({q["fixture_id"] for q in split(queries, "test")})},
            "rule": "fixtures ordered by kickoff; first half dev, second half test",
        },
        "ci": f"cluster bootstrap over fixtures, {N_BOOT} resamples, 95% percentile",
        "k": K,
        "alpha_production": HYBRID_ALPHA,
        "alpha_dev_ndcg@5": {str(a): round(s, 4) for a, s in dev_scores.items()},
        "alpha_tuned_on_dev": tuned,
        "test": {
            f"alpha={a}" + (" (production)" if a == HYBRID_ALPHA else "") + (" (dev-tuned)" if a == tuned else ""): summarise(test[a])
            for a in ALPHAS
        },
        "test_production_by_event_type": by_type,
        "test_production_without_event_filter": summarise(split(no_filter, "test")),
        "test_paired_differences": {
            "hybrid_0.75_minus_dense_1.0 (p@5)": paired(prod, test[1.0], "p@5"),
            "hybrid_0.75_minus_bm25_0.0 (p@5)": paired(prod, test[0.0], "p@5"),
            "hybrid_0.75_minus_random (p@5)": paired(
                prod, [{**r, "p@5": r["random_p@5"]} for r in prod], "p@5"
            ),
        },
    }
    if json_path:
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


def _print(report: dict) -> None:
    print(f"\n{report['index']}")
    print(f"split: {report['split']['dev']} dev / {report['split']['test']} test")
    print(f"dev nDCG@5 by alpha: {report['alpha_dev_ndcg@5']} -> tuned {report['alpha_tuned_on_dev']}\n")
    print(f"{'test':<34} {'p@2':>6} {'p@5':>16} {'hit@1':>6} {'mrr@5':>6} {'loose p@5':>9} {'random':>7}")
    rows = list(report["test"].items()) + [
        ("alpha=0.75, no event filter", report["test_production_without_event_filter"])
    ]
    for name, m in rows:
        ci = m["p@5_ci95"]
        print(
            f"{name:<34} {m['p@2']:>6.3f} {m['p@5']:>6.3f} [{ci[0]:.2f},{ci[1]:.2f}] "
            f"{m['hit@1']:>6.3f} {m['mrr@5']:>6.3f} {m['loose_p@5']:>9.3f} {m['random_p@5']:>7.3f}"
        )
    print("\npaired (test):")
    for k, v in report["test_paired_differences"].items():
        print(f"  {k:<36} {v['mean']:+.4f}  CI {v['ci95']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the report to this path")
    args = ap.parse_args()
    _print(run(args.json))


if __name__ == "__main__":
    main()
