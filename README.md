# PitchPulse: FIFA 2026 Live Football Analytics Engine


![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge)
![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=for-the-badge)
![Weaviate](https://img.shields.io/badge/Weaviate-2A2A5C?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge)
![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?style=for-the-badge&logo=neo4j&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)

PitchPulse is a live match tracking, statistical inference, and AI narration engine for the 2026 FIFA World Cup.

* Ingests live match data from ESPN's public API.
* Enriches it with statistical models, vectorized Monte Carlo tournament simulation, and retrieval-grounded LLM narration.
* Pushes updates to a Next.js UI over SSE: win probability, momentum, match intelligence, counterfactual bracket shifts, tactical comparisons, social narrative surges, and pre-match briefings.

![Demo GIF](data/demo.gif)

## Why Build This Project?

### The Need

- Following a World Cup properly means tracking live events, fan sentiment, tactical patterns, and tournament projections, yet each of these lives on a different platform.
- Fans like me end up stitching these signals together by hand to answer the questions that actually matter: Why did the momentum swing? How big was that goal really? What did one red card do to the title race?
- The deeper analytics, including xG, pressing metrics, and live win probability, usually sit behind paid data providers, so free platforms rarely go beyond the scoreline.
- The moments that define a tournament are measured by their ripple effect: one goal, penalty, or sending-off can redraw qualification paths and championship odds for every team in the bracket.

### The Problem

- Real-time football intelligence is normally built on commercial feeds carrying possession, shots, xG, player tracking, and tactical events.
- Free live feeds tend to stop at the basics: score, clock, and match status.
- With only the basics, there is nothing to drive tactical analysis, probabilistic forecasting, or event-impact modelling.

### The Solution

- PitchPulse builds on free data: ESPN's public match feed for the live tournament, a committed snapshot of every fixture as a fallback, and StatsBomb's World Cup archive for historical context.
- Its own shot model turns raw shot locations into model xG, filling the gap left by the free feed.
- Elo and market priors, an in-play Poisson model, and vectorized Monte Carlo simulation turn each match state into live win probabilities and tournament-wide impact.
- Retrieval-grounded LLM agents narrate those numbers and stream them to the UI in real time, all without a commercial sports data license.

## Overview

* **Live data:** ESPN's public API (`fifa.world`, no key) supplies score, real clock, team stats, key events with players, confirmed lineups, and Opta play-by-play with shot coordinates.
* **Fallback data:** a committed snapshot of the full tournament (`backend/data/wc2026/`) serves every fixture when the feed is unreachable.
* **Model xG:** `ml/shot_xg.py` scores ESPN shots from their coordinates.
* **Historical data:** StatsBomb Open Data (World Cups 2018 and 2022) supplies retrieval precedent, tactical profiles, and evaluation data.
* **Retrieval:** prompt construction, hybrid retrieval, and fallbacks call the Weaviate client and LLM providers directly, without LangChain or LlamaIndex.
* **Orchestration:** match-intel narration is a LangGraph state graph (retrieve → generate → grounding check → one retry → template fallback). The other agents are plain asyncio.
* **Runtime:** any number of backend instances can share one Redis. The instance holding the Redis leader lock runs the producer and workers; the others serve HTTP and SSE from Redis and take over if the leader stops.

## Index

* [Why Build This Project?](#why-build-this-project)
* [Overview](#overview)
* [Core Features](#core-features)
* [The Stack](#the-stack)
* [Architecture at a Glance](#architecture-at-a-glance)
* [Evaluation Results](#evaluation-results)
    - [Common Random Numbers: Counterfactual Variance Reduction](#common-random-numbers-counterfactual-variance-reduction)
    - [Narrative Surge Detection (synthetic)](#narrative-surge-detection-synthetic)
    - [Hybrid Retrieval: WC 2026 Goals and Red Cards](#hybrid-retrieval-wc-2026-goals-and-red-cards)
    - [Generation Faithfulness: RAGAS on WC 2026 Event Narration](#generation-faithfulness-ragas-on-wc-2026-event-narration)
    - [Pre-Match Elo Prior (n=128)](#pre-match-elo-prior-n128)
    - [In-Play Model Calibration (n=64)](#in-play-model-calibration-n64)
* [Performance](#performance)
* [Data Pipeline](#data-pipeline)
    - [Data Ingestion Architecture](#data-ingestion-architecture)
    - [External Data Sources](#external-data-sources)
    - [Live Match Data](#live-match-data)
    - [External Signals](#external-signals)
    - [Schemas](#schemas)
* [The ML Core](#the-ml-core)
* [The Agent Layer](#the-agent-layer)
    - [Match Intelligence Agent](#match-intelligence-agent)
    - [Counterfactual Agent](#counterfactual-agent)
    - [Tactical Agent](#tactical-agent)
    - [Narrative Intelligence Agent](#narrative-intelligence-agent)
    - [Briefing Agent](#briefing-agent)
* [RAG + Knowledge Infrastructure](#rag--knowledge-infrastructure)
    - [Knowledge Construction](#knowledge-construction)
    - [Embedding Pipeline](#embedding-pipeline)
    - [Hybrid Retrieval](#hybrid-retrieval)
    - [Vector Database](#vector-database)
    - [Knowledge Graph](#knowledge-graph)
    - [Grounded Generation](#grounded-generation)
* [Real-Time Intelligence Runtime](#real-time-intelligence-runtime)
    - [Worker Architecture](#worker-architecture)
    - [Redis State Layer](#redis-state-layer)
    - [State Recovery](#state-recovery)
    - [Async Execution](#async-execution)
    - [Streaming Layer](#streaming-layer)
* [Performance Optimizations](#performance-optimizations)
* [Repository Structure](#repository-structure)
* [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Install and Run](#install-and-run)
    - [Populate the Knowledge Base (offline, one time)](#populate-the-knowledge-base-offline-one-time)
    - [Run the Tests and Elo Calibration Backtest](#run-the-tests-and-elo-calibration-backtest)
* [Tech Stack](#tech-stack)
* [Limitations](#limitations)



## Core Features

1. **Live Match Intelligence**
    - Scores goals, cards, momentum shifts, and xG-versus-scoreline divergence every 30 seconds per fixture.
    - Narrates high-value moments with retrieval-grounded LLM generation and a deterministic grounding check; other updates use numeric templates.
    - Streams narratives to the UI over SSE.

    ![Live Match Intelligence](data/05-match-live.png)
---

2. **The Counterfactual What-If Engine**
    - Measures how a goal, card, penalty, or substitution changes every team's championship probability.
    - Compares paired tournament simulations of the pre-event and post-event states with shared random seeds.
    - Outputs probability shifts, team-level deltas, and a simulation-grounded explanation.

    ![Counterfactual What-If Engine](data/03-match-counterfactual.png)
---

3. **The Tournament Simulation Engine**
    - Simulates the 48-team World Cup format from Elo ratings, with market odds as the preferred prior when available.
    - Runs group and knockout stages as vectorized NumPy operations across all simulations.
    - Produces advancement probabilities, championship odds, and confidence intervals for the prediction API and the counterfactual engine.

    ![Tournament Simulation Engine](data/06-match-predictor.png)
---

4. **Tactical Intelligence**
    - Converts live possession, shot volume, and passing accuracy into a tactical style descriptor.
    - Retrieves the closest historical pressing fingerprints from the StatsBomb-based `TacticalProfiles` index, with a live-statistics fallback.

    ![Tactical Intelligence](data/02-match-tactical.png)
---

5. **The Narrative Intelligence Hub**
    - Polls Mastodon, Bluesky, Google Trends, and Wikipedia every 60 seconds for tracked topics.
    - Flags a topic when at least two sources surge together above their own 3-hour baselines (one-sided robust z-score).
    - Produces anomaly scores, source attribution, and retrieval-grounded narrative summaries.

    ![Narrative Intelligence Hub](data/07-match-narrative.png)
---

6. **Pre-Match Briefing**
    - Generates a pre-match preview inside a scheduled kickoff window from team context, head-to-head history, and retrieved precedent.
    - Uses no live match-state signals.

    ![Pre-Match Briefing](data/04-match-briefing.png)

## The Stack

* **Backend**
    - FastAPI with one Uvicorn worker per instance; a Redis leader lock selects the instance that runs the pipeline
    - `redis.asyncio` as the only persistence layer
    - LangGraph for the match-intel narration graph, plain asyncio elsewhere
* **ML**
    - Elo rating with exact Shin (1993) de-vigging of market odds
    - A prior-calibrated Poisson in-play scoring model
    - A vectorized NumPy Monte Carlo tournament simulator
    - A stateless logistic momentum model over recent model xG, sharing one feature function with its offline trainer
    - A shot-level xG model fitted on 32.7k ESPN club shots
    - PPDA-based tactical feature engineering
    - A per-topic robust z-score surge detector with cross-source corroboration
* **AI**
    - Local-first inference via **Ollama** running `mistral:7b-instruct-q4_K_M` (on an RTX 3060)
    - **Groq** as cloud fallback: `llama-3.1-8b-instant` primary, `llama-3.3-70b-versatile` on a 429 rate limit
    - `sentence-transformers/all-MiniLM-L6-v2` for retrieval embeddings
* **Data**
    - ESPN public API (live WC 2026 scores, stats, events, lineups, play-by-play) with a committed tournament snapshot as fallback
    - StatsBomb Open Data (WC 2018/2022 event streams for retrieval, tactical profiles, and head-to-head history)
    - martj42 international results (point-in-time Elo)
    - The Odds API
    - Mastodon, Bluesky, Google Trends, Wikipedia
    - Lineups from ESPN, then API-Sports, then Zafronix squads
* **Frontend**
    - Next.js App Router, entirely client-rendered
    - Native `EventSource` API for streaming
    - No WebSocket library, no server-side data fetching
* **Knowledge Graph**
    - Neo4j holds teams, groups, bracket rounds, and StatsBomb head-to-head history used by pre-match briefings.
* **Observability**
    - Prometheus and Grafana for API, worker, and SSE metrics; LangSmith tracing for agent and LLM calls.
* **CI/CD**
    - GitHub Actions runs `ruff`, the `pytest` suite, and a Docker image build on every push and pull request to `main`.



## Architecture at a Glance

```
ESPN API + wc2026 snapshot ──► match_producer (30s poll) ──► MatchState
                                                                   │
                                                                   ⭣
                                                             Redis (only store)
        ┌───────────┬────────────┬───────────────┬────────────┬───────────┬────────────┐
        ⭣           ⭣            ⭣               ⭣            ⭣           ⭣
   momentum     tactical      intel        counterfactual  narrative   briefing
   worker       worker        worker       worker          worker      worker
   (logistic)   (Weaviate     (LLM+RAG)    (CRN Monte      (LLM+RAG)   (LLM+RAG)
                cosine)                    Carlo, LLM+RAG)
        │           │             │              │              │           │
        │           │             └──────┬───────┴──────────────┴───────────┘
        │           │                    ⭣
        │           │            Agent Layer (Ollama → Groq → template)
        │           │                    │
        ⭣           ⭣                    ⭣
                  FastAPI SSE + REST ──► Next.js UI
```

* One Uvicorn process runs the producer and seven asyncio workers: the six above and a prediction worker that refreshes the tournament simulation every 30 minutes. Workers share state through Redis keys and pub/sub.
* A Redis leader lock (15 s TTL) selects the one instance that runs the producer and workers. Other instances serve REST and SSE from Redis and take over within one lock TTL if the leader stops.
* Each worker runs under a supervisor that restarts it with backoff. `/health` reports per-worker liveness.
* LLM calls pass through per-backend priority gates (`agents/llm_queue.py`). Live event narration is served ahead of queued background trending and backfill jobs.
* Prometheus scrapes API and worker metrics.



## Evaluation Results

Offline evaluations of the core ML components.

* Each evaluation lists its setup, metric, and result, and names the JSON report its table is generated from.
* Tuned parameters are selected on a dev split and reported on a held-out split.
* Intervals are 95% confidence intervals.

> **Data note:** StatsBomb Open Data contains World Cup matches from 2018 and 2022 (128 matches).

### Common Random Numbers: Counterfactual Variance Reduction

* **Setup:**
    - Paired runs of the production 48-team tournament simulator (WC 2026 field and bracket, Poisson scorelines).
    - Argentina at +30 Elo; 10,000 simulations per leg, 20 repeats.
* **Metric:** standard deviation of the change in championship probability, shared seed (CRN) vs independent seeds.
* **Result:** CRN reduces estimator variance 9.9x.
* **Source:** `backend/crn_report.json`

| Metric | CRN (shared seed) | Independent seeds |
|:---|---:|---:|
| Δ champ prob, mean | +0.03905 | +0.03989 |
| Δ champ prob, std | 0.00183 | 0.00576 |
| **Variance reduction** | **9.9x** | — |

### Narrative Surge Detection (synthetic)

* **Setup:**
    - Simulated topic-days of per-minute posts, edits, and Trends intensity (daily cycle, per-topic base rates) with injected, labelled events.
    - Positives: multi-source surges (2–4 sources, ×2–8, 3-min ramp, 5–30 min plateau).
    - Hard negatives: single-source bursts, outages, a source switching to mock data, and a slow 4-hour rise to ×3.
    - Simulated activity runs through the live window-counting helpers and the production `SpikeScorer`.
    - Thresholds are selected by best F1 on 20 dev topic-days (seeds 0–19) and scored on 100 held-out topic-days (seeds 1000–1099, 300 events).
* **Metric:**
    - Event recall (an alert during the event or within 10 minutes after)
    - Alert precision
    - False alerts per topic-day
    - Median latency
* **Result:** the shipped detector (z ≥ 3.0; a lone source fires at z ≥ 6.0) has 0.924 precision, 0.483 recall, and 0.12 false alerts per topic-day.
* **Source:** `backend/anomaly_report.json`

| Detector | z | Event recall | Alert precision | False alerts / topic-day | Median latency |
|:---|---:|---:|---:|---:|---:|
| **Production (corroborated), shipped** | **3.0** | **0.483** [0.427, 0.540] | **0.924** [0.871, 0.956] | **0.12** [0.06, 0.21] | **10 min** |
| Production (corroborated), dev-tuned | 2.0 | 0.660 [0.605, 0.711] | 0.798 [0.744, 0.844] | 0.50 [0.37, 0.66] | 8 min |
| Trends only | 2.0 | 0.687 [0.632, 0.737] | 0.912 [0.867, 0.942] | 0.20 [0.12, 0.31] | 9 min |
| Bluesky only | 3.0 | 0.583 [0.527, 0.638] | 0.888 [0.837, 0.925] | 0.22 [0.14, 0.33] | 4 min |
| Any single source | 3.5 | 0.673 [0.618, 0.724] | 0.762 [0.708, 0.810] | 0.63 [0.48, 0.81] | 5 min |
| Mastodon only | 2.5 | 0.267 [0.220, 0.319] | 0.494 [0.418, 0.570] | 0.82 [0.65, 1.02] | 11.5 min |
| Wikipedia only | 2.0 | 0.240 [0.195, 0.291] | 0.400 [0.331, 0.473] | 1.08 [0.89, 1.30] | 15.5 min |

Shipped-detector recall is 89.8% for four-source ×4–8 surges and 10.3% for two-source ×2–4 surges.

### Hybrid Retrieval: WC 2026 Goals and Red Cards

* **Setup:**
    - Index: the live Weaviate `NarrativeArcs` index (348 docs from StatsBomb WC 2018/2022: 341 goals, 7 red cards).
    - Queries: every goal and red card in the WC 2026 snapshot, through the production query builder and event filter (`match_intel_agent.event_query`).
    - Split: fixtures ordered by kickoff; the first 48 (163 queries) select alpha, the last 49 (161 queries) are the test set.
    - CIs are a cluster bootstrap over fixtures.
* **Metric:**
    - P@5, Hit@1, MRR@5, NDCG@5.
    - A doc is relevant if it matches the event type, the acting team's game state before the event (leading, level, trailing), and the minute band.
    - *Loose* drops the minute band.
* **Result:** production hybrid (alpha 0.75) reaches P@5 0.137 against 0.104 for a random draw from the same filtered pool (paired difference +3.3 points, CI [+1.1, +5.4]).
* **Source:** `backend/retrieval_report.json`

| Alpha | P@5 | Hit@1 | MRR@5 | NDCG@5 | Loose P@5 |
|:---|---:|---:|---:|---:|---:|
| 0.00 (BM25-only) | 0.158 [0.133, 0.183] | 0.155 | 0.298 | 0.163 | 0.424 |
| 0.25 | 0.148 | 0.174 | 0.298 | 0.156 | 0.396 |
| 0.50 | 0.145 | 0.168 | 0.294 | 0.154 | 0.379 |
| **0.75 (production)** | **0.137** [0.114, 0.158] | **0.162** | **0.279** | **0.146** | **0.373** |
| 1.00 (dense-only, dev-tuned) | 0.133 [0.112, 0.152] | 0.143 | 0.264 | 0.139 | 0.368 |
| Random 5 docs | 0.104 [0.098, 0.110] | — | — | — | — |

Without the `event_type` filter, production P@5 is 0.127.

### Generation Faithfulness: RAGAS on WC 2026 Event Narration

* **Setup:**
    - Every goal and red card in the WC 2026 snapshot (324 events) runs through the production narration inputs: event query, live Weaviate hybrid retrieval, win-probability swing, `match_intel_agent._event_prompt`, and Ollama `mistral:7b-instruct-q4_K_M`.
    - Prior: Elo (odds disabled).
    - The raw LLM text is scored against the facts block the model saw: event, score, model xG, possession, win-probability shift, up to two retrieved precedent docs.
    - A seeded random sample of 117 events is judged by `openai/gpt-oss-120b` on Groq.
    - CIs are a cluster bootstrap over fixtures (2,000 resamples).
* **Metric:** RAGAS Faithfulness.
* **Result:** mean faithfulness 0.684, 95% CI [0.639, 0.728].
* **Source:** `backend/ragas_report.json`, script `backend/eval/eval_generation_ragas.py`

| Metric | n | Mean | 95% CI |
|:---|---:|---:|:---|
| Faithfulness, raw LLM output | 117 | **0.684** | [0.639, 0.728] |
| Faithfulness, after team-name guard | 116 | 0.684 | [0.640, 0.728] |
| Share of answers with any unsupported claim | 117 | 0.82 | |
| Share with half or more claims unsupported | 117 | 0.26 | |
| Team-name guard violation rate, all 324 events | 324 | 0.012 | |

### Pre-Match Elo Prior (n=128)

* **Setup:**
    - Every WC 2018 and 2022 match in StatsBomb, scored prequentially: each match is predicted from results-only Elo (started at 1500) built on the matches before it, then Elo is updated.
    - Labels: the 90-minute result (knockout scores rebuilt from first- and second-half goals).
    - Home advantage goes only to the host nation.
    - The live prior uses point-in-time Elo from 49.5k internationals (`ml/elo_ratings.py`).
* **Metric:** log-loss and Brier score against the in-sample outcome base rate.
* **Result:**
    - Δ log-loss +0.000 [−0.034, +0.035] over all 128 matches.
    - Predicted draw rate is 27.9% against 22.7% realised.
* **Source:** `backend/elo_backtest_report.json`

| Matches | Elo log-loss | Base-rate log-loss | Δ log-loss (95% CI) | Δ Brier (95% CI) |
|:---|---:|---:|---:|---:|
| All 128 | 1.068 | 1.068 | +0.000 [−0.034, +0.035] | −0.001 [−0.024, +0.022] |
| Second 64 (warmed up) | 1.077 | 1.066 | +0.012 [−0.042, +0.069] | +0.007 [−0.029, +0.046] |

### In-Play Model Calibration (n=64)

* **Setup:**
    - Elo is warmed up on the 64 WC 2018 matches.
    - All 64 WC 2022 matches are scored with the same prior, host rule, and 90-minute labels as the Elo backtest.
    - At each checkpoint the true score and red cards are fed to `inplay_wdl`.
    - CIs are a paired bootstrap over matches against the pre-match prior.
* **Metric:** log-loss and Brier score at minutes 0, 15, 45, and 75.
* **Result:** log-loss falls from 1.077 (prior) to 0.852 at minute 45 and 0.611 at minute 75.
* **Source:** `backend/inplay_report.json`

| Checkpoint | Log-loss | Brier | Δ log-loss vs prior (95% CI) |
|:---|---:|---:|---:|
| Pre-match prior (Elo) | 1.077 | 0.652 | — |
| Base rate (in-sample) | 1.062 | 0.642 | — |
| Minute 0 | 1.077 | 0.652 | −0.000 [−0.000, +0.000] |
| Minute 15 | 1.095 | 0.658 | +0.018 [−0.063, +0.105] |
| Minute 45 | 0.852 | 0.496 | −0.225 [−0.363, −0.077] |
| Minute 75 | 0.611 | 0.348 | −0.467 [−0.649, −0.262] |

## Performance

| Component | Runs | Time |
|:---|---:|---:|
| Tournament sim (vectorized) | 50,000 | 0.70 s |
| Tournament sim (vectorized) | 10,000 | 0.13 s |
| Counterfactual pair | 20,000×2 | 0.25 s |
| Momentum inference | per call | ~11.3 µs |



## Data Pipeline

### Data Ingestion Architecture

```
 ESPN public API          StatsBomb Open Data           The Odds API
 (scores, stats, events,  (WC 2018/2022 events)         (bookmaker odds)
  lineups, shots)                  │                          │
 + data/wc2026 snapshot            │                          │
        │                          ⭣                          ⭣
        ⭣                rag_indexer.py / tactical_indexer.py   odds_api_client.py
 match_producer.py       (offline)                     (on demand, 5400s TTL cache)
 (30s poll)                        │                          │
        │                          ⭣                          ⭣
 status + real clock      narrative + PPDA document build  per-book Shin de-vig,
 stats/events/lineups                                      then averaged
 shot xG (shot_xg.py)              │                          │
        ⭣                          ⭣                          ⭣
   MatchState / TeamStats    Weaviate (NarrativeArcs,     prior_builder.py
   MatchEvent, shots         TacticalProfiles)            (W/D/L probabilities)
        │                                                       │
        └───────────────────────┬───────────────────────────────┘
                                 ⭣
                              Redis
                    match:{id}:state, match:{id}:shots,
                    match:{id}:lineups, match:{id}:momentum,
                    match:{id}:intel:*, predict:*




 Mastodon      Bluesky      Google Trends      Wikipedia
 (search)      (searchPosts)  (pytrends)       (article revisions)
        │           │              │                │
        └───────────┴──────┬───────┴────────────────┘
                            │
              narrative_spike_detector.py (60s tick)
              per-source rates, mock fallback (never scored)
                            │
              spike_scorer.py (robust z-score, corroboration)
                            ⭣
                          Redis
              narrative:spike:*, narrative:spikes:feed,
              narrative:trending:latest
```

### External Data Sources

* **Live match state:** ESPN public API (`fifa.world`, no key), polled every 30 seconds.
* **Tournament snapshot:** `backend/data/wc2026/`, built by `feeds/build_snapshot.py`.
* **Historical match data:** StatsBomb Open Data (World Cups 2018 and 2022): event streams for retrieval, tactical fingerprints, head-to-head history, and evaluation; match results for Elo calibration.
* **Market data:** The Odds API, fetched on demand and cached for 90 minutes. Each bookmaker's odds are de-vigged (Shin) and the fair probabilities averaged. Market odds are the preferred pre-match prior.
* **Lineups:** ESPN confirmed XI, then API-Sports if `API_SPORTS_KEY` is set, then a Zafronix squad with a projected XI (labelled as projected).
* **Social and search signals:** Mastodon, Bluesky, Google Trends, and Wikipedia, polled every 60 seconds.

### Live Match Data

`match_producer.py` builds every fixture's `MatchState` from ESPN, with the snapshot as fallback.

**Each 30-second poll**

1. One scoreboard request returns every fixture's score, status, and real clock (including stoppage time, extra time, and penalties). If ESPN is unreachable, fixtures come from the snapshot.
2. Match detail per fixture:
    - snapshotted and completed: read from the snapshot file, no request
    - live: ESPN summary every poll
    - completed but not in the snapshot: ESPN summary once, then cached
3. The detail supplies team stats, key events with players (goals, cards, substitutions), confirmed lineups, and shots with coordinates.
4. `MatchState` is written to Redis and `match_update` is published only when the state changes.

**Model xG** (`ml/shot_xg.py`)

* Logistic model over shot coordinates, body part, and assist type parsed from the ESPN commentary line.
* Fitted on 32.7k ESPN club shots (`ml/fit_shot_xg.py`): held-out log-loss 0.265 vs 0.325 for the base rate.
* Per-team-match correlation with StatsBomb xG on WC 2022: r = 0.915. WC 2026 non-penalty shots: 279.6 model xG vs 278 goals.
* Labelled "model xG" in prompts and the UI.

**Snapshot** (`feeds/build_snapshot.py`, `backend/data/wc2026/`)

* 104 fixtures with FIFA match numbers, 12 groups, and per-match stats, events, lineups, and shots (about 3 MB).
* Point-in-time Elo from the martj42 international results dataset (`ml/elo_ratings.py`), head-to-head per fixture, and FIFA Annex C for third-place slotting.
* Sources are cross-validated at build time; `tests/test_wc2026_data.py` checks the snapshot offline.

### External Signals

1. Mastodon, Bluesky, Google Trends, and Wikipedia are polled concurrently every 60 seconds for each tracked topic.
2. An unavailable or rate-limited source produces deterministic mock activity, flagged `mock` in the UI. Mock values are not scored.
3. Counts are converted to per-hour rates (Trends to a within-query lift) and scored against each source's own baseline.
4. Each topic and source keeps a rolling 3-hour window of live values. A topic is flagged when at least two sources surge together.
5. Results feed the narrative agent and the narrative API.

### Schemas

**MatchState** (Pydantic, `api/schemas/schema.py`). Rebuilt every 30 seconds by `match_producer.py`.

```python
class MatchState(BaseModel):
    fixture_id: int
    league_id: int = 1
    season: int = 2026
    round: str = ""
    venue: str = ""
    referee: str = ""
    status_short: str = "NS"
    status_long: str = "Not Started"
    elapsed: Optional[int] = None
    elapsed_extra: Optional[int] = None
    elapsed_estimated: bool = False
    kickoff_time: Optional[datetime] = None
    home_id: int = 0
    home_name: str = ""
    home_logo: str = ""
    home_score: int = 0
    home_stats: TeamStats = Field(default_factory=TeamStats)
    away_id: int = 0
    away_name: str = ""
    away_logo: str = ""
    away_score: int = 0
    away_stats: TeamStats = Field(default_factory=TeamStats)
    events: list[MatchEvent] = Field(default_factory=list)
    stats_source: str = "unavailable"   # "espn" | "unavailable"
    home_pens: Optional[int] = None
    away_pens: Optional[int] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

**MatchEvent** (Pydantic, nested in `MatchState.events`).

```python
class MatchEvent(BaseModel):
    elapsed: int
    extra: Optional[int] = None
    team_id: int
    team_name: str
    player_name: Optional[str] = None
    type: str
    detail: Optional[str] = None
    source: str = "espn"   # "espn" | "synthesised" (from a score delta)
```

**TeamStats** (Pydantic, nested in `MatchState.home_stats` / `away_stats`). Real ESPN match stats; `expected_goals` is the shot model's xG over the match's real shots.

```python
class TeamStats(BaseModel):
    possession: float = 0.0
    shots_total: int = 0
    shots_on_goal: int = 0
    shots_off_goal: int = 0
    passes_total: int = 0
    passes_accurate: int = 0
    pass_accuracy: float = 0.0
    corner_kicks: int = 0
    fouls: int = 0
    offsides: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    goalkeeper_saves: int = 0
    expected_goals: float = 0.0
```

**Momentum features** (`ml/momentum_features.py`). The model is stateless: every tick recomputes features from the match's shot list and events, using the same function the trainer replays over ESPN play-by-play.

```python
FEATURES = [
    "xg15_for", "xg15_against",      # model xG in the last 15 minutes
    "shots15_for", "shots15_against",
    "xg_for", "xg_against",          # cumulative model xG
    "score_diff", "red_diff", "minute_norm",
]
```


## The ML Core

**Point-in-time Elo** (`ml/elo_ratings.py`) is built from 49.5k martj42 international results, every team starting at 1500. After each match:

$$
R_h' = R_h + K\,G\,(W-W_e),\qquad R_a' = R_a - K\,G\,(W-W_e),\qquad W_e=\frac{1}{1+10^{-(R_h-R_a+H)/400}}
$$

- $W$ = result for the home side (1 win, 0.5 draw, 0 loss); $W_e$ = expected result
- $H$ = 100 home advantage, 0 at a neutral venue
- $K$ = 60 (World Cup), 50 (continental finals), 40 (qualifiers, Nations League), 30 (other tournaments), 20 (friendlies)
- $G$ = goal-difference multiplier: 1 for $|gd|\le1$, 1.5 for $|gd|=2$, $(11+|gd|)/8$ for $|gd|\ge3$

---

**Team strength to outcome probability.** Elo expectation:

$$
E_a=\frac{1}{1+10^{\frac{R_b-R_a}{400}}}
$$

Where:

- $E_a$ = expected score for Team A
- $R_a$ = Elo rating of Team A
- $R_b$ = Elo rating of Team B

Three-outcome distribution with a rating-gap-sensitive draw model:

$$
p_{\text{draw}}=\mathrm{clip}\left(0.25e^{-\Delta R/450}+0.05,\;0.03,\;0.30\right)
$$

$$
p_{\text{win}}=(1-p_{\text{draw}})E_a
$$

$$
p_{\text{loss}}=(1-p_{\text{draw}})(1-E_a)
$$

where $\Delta R = |R_a - R_b|$ is the absolute rating gap between the two teams.

---

**Market odds** replace Elo when available and are de-vigged with **Shin's (1993) method**, which assigns more of the bookmaker margin to longshots than to favourites:

$$
p_i(z)=\frac{\sqrt{z^2+4(1-z)\,\dfrac{\pi_i^2}{\Pi}}-z}{2(1-z)},\qquad \pi_i=\frac{1}{o_i},\qquad \Pi=\sum_j \pi_j
$$

$z$ is solved by bisection on $[0, 0.5]$ so that $\sum_i p_i(z)=1$; $\sum_i p_i$ is strictly decreasing in $z$, so the root is unique.

- $p_i$ = fair, de-vigged probability for outcome $i$ (home / draw / away)
- $o_i$ = quoted decimal odds for outcome $i$; $\pi_i$ = raw implied probability
- $\Pi$ = booksum; $\Pi-1$ is the overround
- $z$ = Shin's insider-trading share, $0 < z \le \Pi-1$ for a book with margin; a book with no overround reduces to plain normalisation

---

**In-play win probability** (`ml/in_play.py`)

* Updates from score, minute, and red cards using two independent Poisson goal processes.
* Full-match rates are **calibrated to the pre-match prior**: the 90-minute rates are fitted so the Poisson W/D/L reproduces the prior.
* Fit: least squares over a vectorised grid of total goals × home share, then local refinement; cached per prior.

$$
(\lambda_h^{90},\lambda_a^{90})=\arg\min_{\lambda_h,\lambda_a}\;\big\lVert \mathrm{WDL}_{\text{Pois}}(\lambda_h,\lambda_a)-(p_{\text{win}},p_{\text{draw}},p_{\text{loss}})\big\rVert_2^2
$$

Remaining goals are Poisson at those rates scaled by the time left and by red-card multipliers, and the final margin is enumerated from the current score:

$$
\lambda_{\text{home}}=f\,\lambda_h^{90},\qquad\lambda_{\text{away}}=f\,\lambda_a^{90},\qquad P(g_h,g_a)=\mathrm{Pois}(g_h;\lambda_{\text{home}})\cdot\mathrm{Pois}(g_a;\lambda_{\text{away}}),\qquad g_h,g_a\in[0,12]
$$

where

- $\lambda_{\text{home}}$, $\lambda_{\text{away}}$ = expected goals for each team over the remaining match
- $f$ is the fraction of the match remaining
- $g_h$, $g_a$ = candidate further home/away goals, enumerated over $[0,12]$ (truncated tail renormalised)
- a red card multiplies the offending side's rate by 0.72 and the opponent's by 1.12

With $\ell$ the current home lead, the enumerated scorelines aggregate into W/D/L:

$$
P(\text{home})=\sum_{g_h-g_a>-\ell}P(g_h,g_a),\qquad
P(\text{draw})=\sum_{g_h-g_a=-\ell}P(g_h,g_a),\qquad
P(\text{away})=\sum_{g_h-g_a<-\ell}P(g_h,g_a)
$$

At 0-0 kickoff ($f=1$) the model returns the prior. At full time the result is read from the final score.

`elo_deltas()` converts the change in each side's expected group points into a bounded Elo adjustment for the counterfactual engine and live-adjusted tournament simulations:

$$
\mathrm{EP}=3\,p_{\text{win}}+p_{\text{draw}},\qquad
\Delta R=\mathrm{clip}\big(40\,(\mathrm{EP}_{\text{now}}-\mathrm{EP}_{\text{pre}}),\;-80,\;80\big)
$$

---

**Shot xG** (`ml/shot_xg.py`) is a logistic model over shot geometry and play type:

$$
\mathrm{xG}=\sigma\Big(\beta_0+\beta_1\log d+\beta_2\,\theta+\beta_3 h+\beta_4\,h\log d+\sum_k\gamma_k c_k\Big),\qquad
\theta=\operatorname{atan2}\!\Big(w\,d_x,\;d_x^2+d_y^2-\big(\tfrac{w}{2}\big)^2\Big)
$$

- $d_x$, $d_y$ = metres from the goal line and from the goal's centre line; $d=\sqrt{d_x^2+d_y^2}$
- $\theta$ = angle subtended by the goal mouth, $w=7.32$ m
- $h$ = header flag; $c_k$ = cross, through ball, corner, set piece, fast break, direct free kick
- Penalties take a fixed xG of 0.7711; coefficients are in `ml/shot_xg_coef.json`

---

**Momentum** (`ml/momentum_model.py`) is a stateless logistic model:

$$
p_{\text{team}}=P(\text{team scores within 5 min})=\sigma\left(\beta_0+\sum_i\beta_i\,\mathrm{feature}_i\right),\qquad
\mathrm{momentum}_{\text{home}}=\frac{p_{\text{home}}}{p_{\text{home}}+p_{\text{away}}}
$$

- $\sigma(\cdot)$ = logistic sigmoid function
- $\beta_0$ = intercept; $\beta_i$ = trained coefficient for $\mathrm{feature}_i$: model xG and shots for and against in the last 15 minutes, cumulative xG for and against, score differential, red-card differential, and match minute

* Features are recomputed each tick from the match's shot list by `ml/momentum_features.py`, the same function the trainer uses.
* No per-process state and no hand-set event bumps.
* Coefficients (`ml/momentum_coef.json`) are trained on 1,388 ESPN club matches by `ml/momentum_trainer.py` (report: `ml/momentum_report.json`).
* WC 2026 log-loss: 0.2371, against 0.2403 for a score-and-clock-only model and 0.2420 for the base rate (ECE 0.011).

---

**Tournament simulation** (`ml/tournament_sim.py`)

* Resolves every group match and knockout round for all $N$ runs at once with vectorized NumPy operations.
* Third-place slotting follows FIFA Annex C; the knockout bracket follows matches 73–104 (`ml/wc2026_format.py`).
* Each stage probability carries a confidence margin:

$$
\text{margin}=1.96\sqrt{\frac{\hat{p}(1-\hat{p})}{N}}
$$

- $\hat{p}$ = simulated probability estimate for a given stage outcome (e.g. reaching the quarterfinal)
- $N$ = number of simulation runs
- $\text{margin}$ = half-width of the 95% confidence interval around $\hat{p}$

---

**Tactical identity** is computed from pressing features and matched by cosine similarity:

$$
\mathrm{PPDA}=\frac{\text{Opponent completed passes in press zone}}{\text{Defensive actions in press zone}}
$$

$$
\text{press}\_\text{intensity}=0.7\,\min\left(1,\frac{8}{\mathrm{PPDA}}\right)+0.3\,\min\left(1,\frac{\text{pressures}}{150}\right)
$$

- $\mathrm{PPDA}$ = passes per defensive action; lower values mean more aggressive pressing
- $\text{pressures}$ = count of pressure events in the press zone

---

**Narrative surge detection** (`agents/spike_scorer.py`) scores each live source per topic against its own 3-hour baseline (180 ticks, live values only).

* Per-hour rates:
    - Mastodon: posts in the last 30 min
    - Bluesky: posts in the last 15 min
    - Wikipedia: edits to the topic's article in the last hour
* Trends: lift of the last 5 minutes over the earlier median of the same hourly query.
* With $x=\log(1+\text{rate})$ (or $\log\text{lift}$):

$$
z=\frac{x-\mathrm{median}(h_{:-5})}{\max(1.4826\cdot\mathrm{MAD}(h_{:-5}),\,0.35)}
$$

* $h$ = the source's live history with the 5 most recent points excluded.
* A source is *surging* when $z\ge 3$ and its raw rate clears a per-source floor.
* A topic alerts when:

$$
|\text{surging}|\ge 2 \quad\text{or}\quad \max z \ge 6,
$$

* Alerts fire once per episode; an episode ends after 5 ticks with no source at $z\ge1.5$.
* Severity is $\mathrm{clip}((\bar z_{\text{top2}}-3)/5,0,1)$, using the max $z$ for a single-source alert.
* At most 3 topics alert per tick; the rest stay pending for the next tick.

<br />



## The Agent Layer

Each agent computes a deterministic result before any generation step; the LLM narrates the precomputed outputs. Every agent has a deterministic template fallback built from the same outputs.

| Agent | Inference |
|:---|:---|
| Match intelligence | Local-first (Ollama → Groq → template), LangGraph narration graph |
| Counterfactual | Local-first (Ollama → Groq → template) |
| Narrative | Local-first (Ollama → Groq → template) |
| Briefing | Groq, then template |
| Tactical | No generation |

### Match Intelligence Agent

Runs every 30 seconds per fixture against the live `MatchState` and momentum snapshot.

**Relevance scoring**

* Priority score inputs:
    - Uncovered goals and red cards
    - Momentum change since the last narrated snapshot
    - xG-versus-scoreline divergence
    - Stoppage-time and extra-time context
* A content hash over the scoreline, momentum state, and latest event skips unchanged states.
* Ticks below the relevance threshold are skipped unless a forced trigger fires:
    - a baseline narrative once the fixture reaches an initial live-state milestone
    - a tactical read every 5 match minutes (scheduled on the match clock, not wall time)

**Narrative categories and retrieval**

| Category | Trigger | Retrieval |
|:---|:---|:---|
| Event reaction | Goal or card | `NarrativeArcs` |
| xG divergence | xG well above actual goals | `NarrativeArcs` |
| Tactical analysis | Forced tactical read or momentum shift | `TacticalProfiles` |

**Generation**

* LLM generation runs for major events, forced tactical reads, and high-relevance states. Other updates use deterministic numeric templates.
* Narration is a LangGraph `StateGraph` (`agents/intel_graph.py`): retrieve → generate → grounding check → one retry with the violations in the prompt → template fallback.
* The grounding check (`agents/grounding.py`) validates teams, players, scorelines, percentages, decimals, and minutes against `MatchState` and the prompt facts.
* Outcomes are counted in `wc2026_intel_narrations_total{kind,outcome}`.

**Event timeline**

* A separate event timeline holds a retrieval-grounded reaction for every goal and red card.
* Every fixture gets a full-time summary; matches without major events use xG and possession.

### Counterfactual Agent

* Triggers on goals, cards, penalties, own goals, and substitutions, with a 45-second minimum gap per fixture.
* Reconstructs the pre-event match state by reversing the event's effect on score and cards.
* Computes in-play W/D/L before and after the event and converts the change into a bounded Elo adjustment.
* An event that changes neither the score nor the number of players on the pitch skips the simulations and the LLM call.
* Runs paired 20,000-run tournament simulations (`CF_SIMS`) with shared seeds for the pre-event and post-event states.
* Teams whose championship probability moves by at least 0.003 are listed; if none moves, the template is used without an LLM call.

Per-team change and aggregate tournament impact:

$$
\Delta p_i = p_i^{after} - p_i^{before}
$$

$$
\text{path shift} = \min\left(1,\frac{\sum_i |\Delta p_i|}{2}\right)
$$

Both legs use the same seed (common random numbers), so their estimates are positively correlated and the variance of $\Delta p_i$ drops:

$$
\mathrm{Var}(\hat p_i^{after}-\hat p_i^{before})=\mathrm{Var}(\hat p_i^{after})+\mathrm{Var}(\hat p_i^{before})-2\,\mathrm{Cov}(\hat p_i^{after},\hat p_i^{before})
$$

The largest movers and the in-play win-probability swing go to local-first generation.

### Tactical Agent

* Deterministic; no generation step.
* Converts live possession, shot volume, and passing accuracy into a tactical style descriptor.
* Embeds the descriptor and retrieves the closest pressing fingerprints from `TacticalProfiles` by cosine similarity, with additional comparable profiles.
* Falls back to live statistics when tactical retrieval is unavailable.

### Narrative Intelligence Agent

* Reads spikes and trending topics from the surge detector ([External Signals](#external-signals)).
* Collects from Mastodon (authenticated search), Bluesky (authenticated `searchPosts` with session recovery), Google Trends (`pytrends` on the I/O thread pool), and Wikipedia (article revision history).
* Keeps a separate trending ranking from relative activity changes across tracked topics.
* Builds topic-aware queries against `NarrativeArcs` and generates a narrative from the retrieved context and per-source activity.
* Validates output against banned-phrasing rules; failed or unavailable generation uses a rule-based fallback.
* Output per spike: anomaly score, source attribution, narrative summary.

### Briefing Agent

* Generates pre-match briefings inside a scheduled pre-kickoff window, once per match status, or on demand.
* Inputs: team context, as-of-kickoff facts (`agents/briefing_facts.py`), up to three head-to-head meetings from Neo4j, and retrieved precedent from `NarrativeArcs`.
* Uses no live match-state signals.
* Calls Groq directly; the deterministic fallback summarizes the retrieved context.



## RAG + Knowledge Infrastructure

### Knowledge Construction

1. StatsBomb WC 2018/2022 event streams feed both retrieval collections.
2. Goal and red-card situations become natural-language documents with `match_id`, competition, season, minute, and event-type metadata, plus `situation` metadata (game state, minute band) used as the retrieval eval label.
3. Tactical profiles hold team-level pressing features per match, including PPDA and pressing by pitch third.
4. Indexing runs offline through CLIs. The tactical index and the knowledge graph also populate on startup when empty.

### Embedding Pipeline

1. Each document is a self-contained passage; there is no chunking step.
2. Embeddings come from `sentence-transformers/all-MiniLM-L6-v2` with normalized vectors, loaded once in `ml/embedding_model.py`.
3. Vectors and metadata are stored together in Weaviate.
4. Embedding runs on a dedicated thread pool.

### Hybrid Retrieval

1. Dense retrieval: cosine similarity over embedding vectors.
2. Sparse retrieval: BM25 over the same corpus.
3. Fusion: Weaviate relative-score fusion with `alpha=0.75` (75% dense). Each result list is min-max normalized per query, then combined:

$$
s(d)=\alpha\,\hat s_{\text{dense}}(d)+(1-\alpha)\,\hat s_{\text{BM25}}(d),\qquad
\hat s_{\text{dense}}\propto\cos(q,d)=\frac{q\cdot d}{\lVert q\rVert\,\lVert d\rVert}
$$

4. Narrative retrieval filters by event type (`goal`, `red_card`) when the event is known.

### Vector Database

Weaviate 1.27 runs without a built-in vectorizer; embeddings are computed by the application and supplied with each insert.

| Collection | Contents |
|:---|:---|
| `NarrativeArcs` | Goal and red-card narratives from every StatsBomb WC 2018/2022 match (348 docs) |
| `TacticalProfiles` | Team-level pressing fingerprints (PPDA, pressing by third) |

### Knowledge Graph

Neo4j 5 stores tournament structure and head-to-head history.

* Nodes: `Team`, `Group`, `BracketRound`, `HistoricalMatch`. Relationships: `PLAYS_IN`, `ADVANCES_TO`, `HEAD_TO_HEAD`.
* Teams, groups, and bracket rounds come from the same configuration as the simulator.
* `HEAD_TO_HEAD` edges come from StatsBomb historical matches between teams in the simulator field.
* `kg/graph_builder.py` builds the graph offline and on startup when the store is empty.

### Grounded Generation

1. Top-ranked retrieved passages go into the generation prompt with computed metrics: scoreline, xG, probability changes, and activity signals.
2. Narrative and briefing prompts restrict the model to retrieved context and computed outputs.
3. Match-intel drafts pass the grounding check described in [Match Intelligence Agent](#match-intelligence-agent).
4. Unavailable generation falls back to deterministic templates over the same values.



## Real-Time Intelligence Runtime

### Worker Architecture

| Worker | Cadence | Responsibility |
|:---|:---|:---|
| Producer | 30s | Polls the live feed, builds `MatchState` |
| Momentum | 30s | Logistic inference from recent shot xG |
| Intel | 30s | Event, xG, and tactical scoring; live narration |
| Counterfactual | 30s | Trigger detection; paired Monte Carlo; bracket-impact narration |
| Tactical | 120s | Style descriptor; cosine match against history |
| Briefing | 300s | Pre-match window gating; tactical preview |
| Narrative | 60s | Social signal aggregation; anomaly scoring; arc synthesis |
| Prediction | 1800s | Background tournament-simulation refresh |

Two one-time startup tasks populate the tactical index and the knowledge graph when empty. Simulation requests create short-lived tasks on demand through `predict.py`.

### Redis State Layer

* Redis holds live match state, momentum snapshots, intelligence feeds, counterfactual results, narrative signals, and simulation outputs.
* Keys carry TTLs per match lifecycle stage, from pre-kickoff through completed-match history.
* Pub/sub channels carry match state, momentum, intelligence, counterfactual, and narrative updates. Tactical and briefing outputs are read from their Redis keys.

### State Recovery

* `MatchState` is updated in place; there is no durable event log.
* On restart, workers reload tracking state from Redis and skip already-processed events.
* Workers reset per-fixture caches when a fixture's match state moves backwards.

### Async Execution

1. The producer, the seven workers, and the startup tasks are `asyncio` tasks sharing one event loop and Redis connection, running only on the leader instance.
2. Each task catches its own exceptions, and a supervisor restarts failed workers with backoff.
3. Simulations, embeddings, and blocking external API calls run on dedicated thread pools (`ml/executors.py`).

### Streaming Layer

* SSE endpoints subscribe to Redis pub/sub and send the current Redis state on connect.
* The frontend uses native `EventSource` reconnection.
* Match intelligence, momentum, counterfactual, and narrative updates stream over SSE. Social-comment samples use a polling endpoint backed by cached provider results.



## Performance Optimizations

* **Vectorized Monte Carlo simulation:** group and knockout stages run as NumPy array operations across all simulations.
* **Dedicated execution pools:** separate thread pools for simulation, embedding, and blocking I/O.
* **TTL caches with stale-on-failure:** bookmaker odds and rosters are cached and reused when the provider fails.
* **Redis caching:** tactical fingerprints, external lookups, and derived intelligence artifacts.
* **Change-aware publishing:** Redis writes and pub/sub notifications fire only when serialized match state changes.
* **Cooldowns:** fixture-level and topic-level cooldowns on simulation and LLM generation paths.
* **No-impact skip:** counterfactual events that change neither the score nor the number of players on the pitch skip simulation.
* **Shared embedding model:** one module-level `SentenceTransformer` instance.
* **Batched offline indexing:** RAG and tactical indexers encode documents in batches.
* **Single-query hybrid search:** Weaviate fuses dense and BM25 results server-side.



## Repository Structure

```
backend/
├── agents/
│   ├── briefing_agent.py            Pre-match briefing generation, Groq only, no local-first tier
│   ├── briefing_facts.py            As-of-kickoff facts a briefing may cite
│   ├── counterfactual_agent.py      Paired CRN Monte Carlo simulation and bracket-impact narration
│   ├── grounding.py                 Deterministic claim check for narration (teams, players, numbers)
│   ├── intel_graph.py               LangGraph narration graph: retrieve, generate, check, retry, template
│   ├── match_intel_agent.py         Live event/xG/tactical scoring and narrative generation
│   ├── narrative_arc_agent.py       Evidence-grounded narrative synthesis for detected spikes
│   ├── narrative_spike_detector.py  Four-source signal collection (per-hour rates, Trends lift)
│   ├── narrative_topics.py          Fixture-aware topic tracking for the Narrative Hub
│   ├── spike_scorer.py              Robust z-score surge scorer with cross-source corroboration
│   ├── langsmith_tracing.py         Standalone LangSmith @traceable
│   ├── llm_queue.py                 Priority gates for Ollama and Groq calls
│   ├── ollama_client.py             Local Ollama call with Groq fallback
│   ├── rag_indexer.py               Offline StatsBomb narrative document extraction and indexing
│   ├── tactical_agent.py            Live style descriptor and cosine match against TacticalProfiles
│   └── weaviate_client.py           Weaviate connection, collection schema, hybrid search wrapper
├── api/
│   ├── main_hybrid.py               FastAPI app and lifespan startup
│   ├── match_timeline.py            Match state as of a given event, from the event timeline
│   ├── supervisor.py                Redis leader lock and supervised workers
│   ├── tournament_state.py          Played results and live matches for the simulator
│   ├── routes/
│   │   ├── _security.py             Trigger-token dependency for debug endpoints
│   │   ├── _sse.py                  Shared Redis pub/sub-backed SSE generator
│   │   ├── briefing_routes.py       Briefing feed and trigger endpoints
│   │   ├── comment_sampler.py       Duplicate comment-sample storage, not mounted in the app
│   │   ├── counterfactual_routes.py Counterfactual feed, prediction, and live-prob endpoints
│   │   ├── group_table.py           Live group-stage standings from completed fixtures
│   │   ├── intel.py                 Intel feed and SSE stream endpoints
│   │   ├── lineups.py               Tiered lineup resolution: ESPN, API-Sports, then Zafronix projected XI
│   │   ├── match.py                 Fixture list and summary endpoints
│   │   ├── match_stream.py          Match-state SSE stream
│   │   ├── momentum.py              Momentum snapshot and SSE stream
│   │   ├── narrative.py             Spike, trending, arc, and narrative SSE endpoints
│   │   ├── narrative_comments.py    Comment sample storage and retrieval
│   │   ├── predict.py               Tournament simulation trigger, status, and result endpoints
│   │   ├── tactical.py              Tactical fingerprint endpoint and cache
│   │   └── team_form.py             Last-five-match form endpoint
│   ├── schemas/
│   │   ├── event_types.py           Shared event-type and status-code vocabulary
│   │   ├── intel_schema.py          Pydantic intel schema, defined but unused at runtime
│   │   ├── predict.py               Validated prediction response models
│   │   └── schema.py                MatchState, MatchEvent, TeamStats definitions
│   └── workers/
│       ├── briefing_worker.py       300s kickoff-window scan and briefing trigger
│       ├── counterfactual_worker.py 30s trigger detection and simulation dispatch
│       ├── intel_worker.py          30s event/momentum scoring and narration dispatch
│       ├── match_producer.py        30s ESPN poll with snapshot fallback, MatchState build
│       ├── momentum_worker.py       30s logistic momentum inference per fixture
│       ├── narrative_worker.py      60s signal aggregation, anomaly scoring, arc synthesis
│       ├── prediction_worker.py     1800s background tournament-simulation refresh
│       └── tactical_worker.py       120s tactical fingerprint refresh
├── eval/
│   ├── eval_anomaly_threshold.py    Surge-detector eval on simulated topic-days
│   ├── eval_crn_variance.py         CRN variance-reduction eval
│   ├── eval_generation_ragas.py     RAGAS faithfulness eval (generate / judge stages)
│   ├── eval_inplay_calibration.py   In-play W/D/L calibration eval
│   └── eval_retrieval.py            Retrieval eval against the live NarrativeArcs index
├── feeds/
│   ├── build_snapshot.py            Builds the committed WC 2026 snapshot
│   ├── espn.py                      ESPN public API client and pure parsers
│   └── snapshot.py                  Read side of the snapshot
├── ml/
│   ├── backtest_elo_wdl.py          Elo calibration backtest against real WC 2018/2022 results
│   ├── elo_ratings.py               Point-in-time World Football Elo from martj42 results
│   ├── embedding_model.py           Shared all-MiniLM-L6-v2 instance, loaded once
│   ├── executors.py                 Dedicated thread pools for simulation, embedding, and I/O
│   ├── fit_shot_xg.py               Fits and validates the shot xG model
│   ├── in_play.py                   Prior-calibrated Poisson in-play W/D/L model and Elo delta conversion
│   ├── momentum_features.py         Momentum features shared by trainer and live model
│   ├── momentum_model.py            Stateless logistic momentum inference
│   ├── momentum_trainer.py          Offline momentum coefficient training and validation gate
│   ├── odds_api_client.py           Odds API client with TTL cache and stale-on-failure fallback
│   ├── prior_builder.py             Elo-to-WDL and exact Shin de-vig probability construction
│   ├── shot_xg.py                   Shot-level model xG for ESPN plays
│   ├── schemas/
│   │   └── momentum_schema.py       Pydantic momentum schema, defined but unused at runtime
│   ├── statsbomb.py                 Shared StatsBomb parsing constants and helpers
│   ├── tactical_indexer.py          Offline PPDA feature extraction and TacticalProfiles indexing
│   ├── team_names.py                Team name alias mapping across three naming systems
│   ├── tournament_sim.py            Vectorized Monte Carlo group and knockout simulator
│   ├── wc2026_format.py             FIFA WC 2026 knockout format (matches 73-104)
│   └── wc_2026_config.py            Real 48-team field, groups and Elo, loaded from the snapshot
├── kg/
│   ├── schema.py                    Node/relationship constants + idempotent constraint DDL
│   ├── neo4j_client.py              Degrade-on-failure driver wrapper (mirrors weaviate_client.py)
│   └── graph_builder.py             Static graph + head-to-head edge population, CLI entrypoint
├── monitoring/
│   └── metrics.py                   Prometheus Counter/Histogram/Gauge objects for the workers + SSE
├── tests/                           pytest suite (math models, runtime, producer, snapshot, KG schema)
├── data/wc2026/                     Committed WC 2026 snapshot (fallback feed)
├── *_report.json                    Eval reports the Evaluation Results tables are built from
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt             pytest, ruff, RAGAS (offline eval only)

monitoring/                          Prometheus scrape config + Grafana provisioning/dashboards

.github/workflows/ci.yml             ruff lint gate + pytest + Docker build check on push/PR to main

data/                                README demo GIF and screenshots

ui/
├── app/
│   ├── layout.tsx                   Root layout, navigation, theme initialization
│   ├── page.tsx                     Match Center fixture list
│   ├── match/[id]/page.tsx          Live match detail page
│   ├── narrative/page.tsx           Narrative Hub page
│   └── predict/page.tsx             Tournament predictor page
├── components/
│   ├── Flag.tsx, NavBar.tsx, ThemeToggle.tsx, theme-provider.tsx   Shared UI chrome
│   ├── match/                       Score, stats, momentum, tactical, counterfactual, lineup panels
│   ├── narrative/CommentBubbles.tsx Auto-scrolling live comment sample row
│   └── predict/                     Group, bracket-impact, and probability chart panels
├── hooks/
│   ├── useCounterfactualStream.ts, useIntelStream.ts, useMatchStream.ts,
│   │   useMomentumStream.ts, useNarrativeStream.ts   One SSE hook per Redis pub/sub channel
│   ├── useMatchBriefing.ts, useMatchPrediction.ts, usePredictStream.ts, useTactical.ts
│   │                                Polling hooks for non-SSE endpoints
│   └── useTheme.ts                  Light/dark theme state
├── lib/api.ts, flag.ts, via.ts
├── types/match.ts, predict.ts       TypeScript mirrors of the backend Pydantic schemas
└── package.json, next.config.js, tsconfig.json

docker-compose.yml     Redis 7 + Weaviate 1.27 + Neo4j 5 + Prometheus + Grafana + API,
                       single API replica; UI runs separately
weaviate_data/         Bind-mounted Weaviate volume, runtime data, git-ignored (rebuilt by the indexers)
neo4j_data/            Bind-mounted Neo4j volume, runtime data, git-ignored (rebuilt by the graph builder)
```



## Setup

### Prerequisites

* Docker (for Redis 7, Weaviate 1.27, Neo4j 5, Prometheus, and Grafana)
* Python 3.12
* Node.js (for the Next.js UI)
* Ollama (optional, for the local-first LLM tier: `mistral:7b-instruct-q4_K_M`)

### Install and Run

```bash
# secrets: docker compose requires the four values marked required
cp .env.example .env

# infrastructure
docker compose up -d redis weaviate neo4j prometheus grafana

# backend (outside compose, set REDIS_URL in .env with the Redis password)
cd backend
pip install -r requirements.txt
ollama pull mistral:7b-instruct-q4_K_M   # optional
uvicorn api.main_hybrid:app --reload --port 8000

# frontend
cd ../ui
npm install
npm run dev
```

**Environment**

* `GROQ_API_KEY` enables the cloud LLM tier and the briefing agent.
* The other keys in `.env.example` are optional; without them the system uses Elo instead of market odds, mock social signals, and fewer lineup tiers.
* Without `TRIGGER_TOKEN` the debug trigger endpoints are unauthenticated.

**Local dashboards**

* Grafana: `localhost:3001` (dashboard: **PitchPulse → PitchPulse — API & Worker Health**)
* Prometheus: `localhost:9090`
* Neo4j Browser: `localhost:7474`

### Populate the Knowledge Base (offline, one time)

`weaviate_data/` is not committed. Build the stores once:

```bash
python -m agents.rag_indexer        # NarrativeArcs, every StatsBomb WC 2018/2022 match (348 docs)
python -m ml.tactical_indexer       # TacticalProfiles, also auto-runs on first empty-collection startup
python -m kg.graph_builder          # Neo4j: teams/groups/bracket + head-to-head edges, also auto-runs on first empty-graph startup
```

### Run the Tests and Elo Calibration Backtest

```bash
cd backend
pip install -r requirements-dev.txt
PYTHONPATH=. pytest tests/
PYTHONPATH=. python ml/backtest_elo_wdl.py --json report.json
```

## Tech Stack

| Category | Stack |
|:---|:---|
| Languages | Python 3.12, TypeScript |
| ML | NumPy (vectorized simulator, logistic shot-xG and momentum fits), `sentence-transformers` (`all-MiniLM-L6-v2`) on PyTorch |
| Backend | FastAPI, Uvicorn, `sse-starlette`, Pydantic v2, `httpx`, `redis.asyncio`, `weaviate-client`, `neo4j`, `prometheus-fastapi-instrumentator` |
| Frontend | Next.js 14 App Router, React 18, `next-themes` |
| Data | Redis 7, Weaviate 1.27, Neo4j 5 |
| Observability | Prometheus, Grafana, LangSmith (agent/LLM tracing, standalone SDK) |
| Orchestration | LangGraph (match-intel narration graph) |
| Infra | Docker Compose, `python-dotenv`, `python:3.12-slim` base image, `gcc` (build-time, scientific Python wheels) |
| Dev / Testing / CI | `ruff`, `pytest`, `fakeredis`, GitHub Actions, RAGAS (offline eval) |
| External data | ESPN public API, StatsBomb Open Data, martj42 results, The Odds API, API-Sports, Zafronix, Mastodon, Bluesky, Google Trends (`pytrends`), Wikipedia REST |
| AI | Ollama (`mistral:7b-instruct-q4_K_M`), Groq (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile` on 429) |

## Limitations

* StatsBomb Open Data contains World Cup matches from 2018 and 2022.
* ESPN's public API is undocumented and keyless. ESPN publishes no xG; all xG comes from `ml/shot_xg.py`.
* Match state updates at the 30-second poll interval.
* Social and search sources are rate-limited. Unavailable sources show mock values, which are not scored.
* The surge detector is evaluated on synthetic data only.
* Counterfactuals model event-driven changes in team strength, not alternate tactics or coaching decisions.
* Raw narration scores 0.684 RAGAS faithfulness; 82% of sampled narrations contain at least one unsupported claim.
* The producer, workers, and LLM calls run on the leader instance. Extra instances add HTTP and SSE capacity only.
* The workers require a long-running host; the backend does not run on request-based serverless platforms.
