# PitchPulse: FIFA 2026 Live Football Analytics Engine



PitchPulse is a live match tracking, statistical inference, and AI narration engine built for the 2026 World Cup. It ingests bare-bones live score data and mathematically enriches it using historical statistics, vectorized Monte Carlo tournament simulations, and large language models (LLMs). The primary goal is to deliver real-time, push-driven analytical updates: live momentum, counterfactual bracket shifts, and tactical briefings, without relying on an expensive commercial live-stats feed.


![Demo GIF](data/demo.gif)

The retrieval and agent layers are **built from scratch: no LangChain, no LangGraph, no LlamaIndex, or any other RAG framework.** Every prompt, retrieval call, and fallback path is hand-written directly against the Weaviate client and the LLM providers.
The backend operates entirely on a single OS process to preserve in-memory state, scaling horizontally by running independent stacks against a shared Redis instance.



## Why Build This Project?

### The Need
- Modern football coverage produces an overwhelming amount of fragmented information: live commentary, social sentiment, tactical analysis, and tournament projections exist across disconnected platforms. Fans liek myself are left manually synthesizing these signals to answer deeper questions: Why did momentum shift? How significant was that goal? How did a single event change tournament probabilities? etc.

- While advanced analytics such as expected goals (xG), pressing intensity, and live probability models have transformed professional analysis, these capabilities remain largely inaccessible behind expensive data providers. Most free platforms expose only basic score updates, creating a gap between the analytical depth fans want and the limited information available during live matches.

- The biggest moments in football are often defined by their downstream impact, a late penalty, red card, or unexpected goal can reshape qualification probabilities across an entire tournament. Existing platforms report what happened; PitchPulse focuses on explaining what it means and what could happen next.

- PitchPulse was built to solve a critical data availability problem for live sports applications. The free, keyless live feed (`worldcup26.ir`) only exposes the score, match status, and clock. It lacks possession, shots, expected goals (xG), or any other tactical statistic. To provide an advanced analytics experience, the system relies on an architecture that lazily pairs real 2026 fixtures with structurally similar historical matches from StatsBomb's open data sets. This enables complex ML and RAG features without commercial data licenses.

### The Problem
- Building a real-time football intelligence system typically requires expensive commercial feeds containing possession, shots, xG, player tracking, and tactical events.
- The available free live feed (worldcup26.ir) provides only minimal match state: score, clock, and status.
- Without access to rich telemetry, traditional approaches cannot generate advanced live analysis.

### The Solution 
- PitchPulse solves this constraint through a custom enrichment architecture. Instead of relying on proprietary live data, it dynamically maps live 2026 fixtures to structurally similar historical World Cup matches from StatsBomb open datasets, transferring historical tactical context into sparse live events.

- The enriched match state is combined with probabilistic models, vectorized Monte Carlo simulations, retrieval-grounded LLM agents, and real-time event streaming to create a predictive football intelligence platform capable of explaining live events, generating tactical narratives, and quantifying tournament-wide consequences without requiring commercial sports data licenses.




## Index

* [Why This Project?](#why-this-project)
* [Core Features](#core-features)
* [The Stack](#the-stack)
* [Architecture at a Glance](#architecture-at-a-glance)
* [Evaluation Results](#evaluation-results)
    - [Common Random Numbers: Counterfactual Variance Reduction](#common-random-numbers--counterfactual-variance-reduction)
    - [Narrative Anomaly Detection: IsolationForest Threshold Sweep](#narrative-anomaly-detection--isolationforest-threshold-sweep)
    - [Hybrid Retrieval: Recall/Precision/MRR/NDCG @ K=5, Alpha Sweep](#hybrid-retrieval--recallprecisionmrrndcg--k5-alpha-sweep)
    - [In-Play Model Calibration: Real World Cup Matches (n=12)](#in-play-model-calibration--real-world-cup-matches-n12)
    - [Limitations](#limitations)
* [Performance](#performance)
* [Data Pipeline](#data-pipeline)
    - [Data Ingestion Architecture](#data-ingestion-architecture)
    - [External Data Sources](#external-data-sources)
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
    - [Grounded Generation](#grounded-generation)
* [Real-Time Intelligence Runtime](#real-time-intelligence-runtime)
    - [Worker Architecture](#worker-architecture)
    - [Redis State Layer](#redis-state-layer)
    - [Event Sourcing (Partial Implementation)](#event-sourcing-partial-implementation)
    - [Async Execution](#async-execution)
    - [Streaming Layer](#streaming-layer)
* [Deep Dive: The Three Core Intelligence Engines](#deep-dive-the-three-core-intelligence-engines)
    - [1. Narrative Intelligence Hub](#1-narrative-intelligence-hub)
    - [2. Counterfactual What-If Engine](#2-counterfactual-what-if-engine)
    - [3. Tournament Simulation Engine](#3-tournament-simulation-engine)
* [Performance Optimizations](#performance-optimizations)
* [Repository Structure](#repository-structure)
* [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Install and Run](#install-and-run)
    - [Populate the Knowledge Base (offline, one time)](#populate-the-knowledge-base-offline-one-time)
    - [Run the Elo Calibration Backtest](#run-the-elo-calibration-backtest)
* [Tech Stack](#tech-stack)
* [Limitations](#limitations-1)



## Core Features

1. **Live Match Intelligence**
    - Runs on a 30-second worker tick against the current match state and momentum snapshot.
    - Scores every uncovered goal, card, and momentum shift for narrative relevance before deciding whether a language model call is warranted at all.
    - Low-scoring ticks return a numeric template directly; higher-scoring ticks retrieve grounding context from a vector store and generate a short reaction through local-first inference.
    - Output: a short narrative entry streamed to the match page as an SSE event, built from the same numbers used by the fallback template.
2. **The Counterfactual What-If Engine**
    - Triggers on a goal, own goal, penalty, card, or substitution event. Own-goal, penalty, and second-yellow variants are handled in code but not yet produced by the live feed. A substitution consumes the rate-limit gap without changing the resulting state.
    - Reconstructs the match state immediately before the event and runs two 20,000-run tournament simulations under one shared random seed: one before the event, one after.
    - Bounded by a 45-second minimum gap per fixture to cap compute cost.
    - Output: a probability-shift percentage, ranked per-team deltas, and a narrative explanation traceable to that single simulation pair.
3. **The Tournament Simulation Engine**
    - Takes static 48-team Elo configuration, optionally overridden by de-vigged market odds.
    - Resolves the group stage in one vectorized NumPy pass and the knockout bracket in five further vectorized passes (one per round, each fully vectorized across all requested runs), rather than sequential per-simulation loops.
    - Executes on demand: triggered explicitly, or implicitly on first read after a cache expiry.
    - Output: per-team, per-stage advancement and championship probabilities with 95% confidence intervals; no language model or retrieval step anywhere in the path.
4. **Tactical Intelligence**
    - Builds a live style descriptor from possession, shot volume, and pass accuracy, since the live feed exposes none of the pressing statistics the historical index was built from.
    - Embeds that descriptor and retrieves the closest historical pressing fingerprint by cosine similarity against the indexed StatsBomb corpus.
    - No generation step exists in this path. The best available match is always returned when the index holds any profile. Nothing is returned only when Weaviate is unreachable or the collection is empty; the interface falls back to live statistics directly.
5. **The Narrative Intelligence Hub**
    - Runs a 60-second aggregation tick across four independent social and search sources per tracked topic.
    - Scores each topic's latest reading against a per-topic Isolation Forest fit on its own 72-hour rolling window, firing a spike when the anomaly score clears a fixed threshold outside a per-topic cooldown.
    - Detected spikes and a separately-ranked trending list both pass through retrieval-grounded narrative generation that reasons explicitly from which sources moved and which didn't.
    - Output: a severity score, source attribution, and a validated narrative string.
6. **Pre-Match Briefing**
    - Generates inside a fixed kickoff window or on demand, using only the two team names and retrieved historical precedent. No live match state dependency exists in this path.
    - Generation is grounded against retrieved documents and instructed not to invent precedent beyond them.
    - This is the one path in the system that calls Groq directly rather than attempting local inference first.



## The Stack

* **Backend**: FastAPI, single-worker by design, with `redis.asyncio` as the only persistence layer. No relational database is used anywhere in the system.
* **ML**: Elo rating with Shin de-vigging, a Poisson in-play scoring model, a vectorized NumPy Monte Carlo tournament simulator, an EWMA-smoothed logistic momentum model with a separate offline training pipeline, PPDA-based tactical feature engineering, and a per-topic `IsolationForest` for social/search anomaly detection. Five distinct statistical models.
* **AI**: local-first inference via **Ollama** running `mistral:7b-instruct-q4_K_M` (accelerated locally by an RTX 3060 in this deployment), with **Groq** (`llama-3.3-70b-versatile`) as cloud fallback, and `sentence-transformers/all-MiniLM-L6-v2` for retrieval embeddings.
* **Data**: StatsBomb Open Data (real WC 2018/2022 event streams), `worldcup26.ir` (live fixture feed), The Odds API, Mastodon/Bluesky/Google Trends/Wikipedia, and Zafronix/API-Sports for lineups.
* **Frontend**: Next.js App Router, entirely client-rendered, using the native `EventSource` API for streaming. No WebSocket library. No server-side data fetching.



## Architecture at a Glance

```
worldcup26.ir + StatsBomb ──► hybrid_producer (30s poll) ──► MatchState
                                                                   │
                                                                   ⭣
                                                             Redis (only store)
        ┌───────────┬────────────┬───────────────┬────────────┬───────────┬────────────┐
        ⭣           ⭣            ⭣               ⭣            ⭣           ⭣            
   momentum     tactical      intel        counterfactual  narrative   briefing
   worker       worker        worker       worker          worker      worker
   (EWMA)       (Weaviate     (LLM+RAG)    (CRN Monte      (LLM+RAG)   (LLM+RAG)
                cosine)                    Carlo, LLM+RAG)             
        |                                                                
        │           │             │              │              │           │
        │           │             └──────┬───────┴──────────────┴───────────┘
        │           │                    ⭣
        │           │            Agent Layer (Ollama → Groq → template)
        │           │                    │
        ⭣           ⭣                    ⭣
                  FastAPI SSE + REST ──► Next.js UI
```

One Redis connection, one Uvicorn process, seven recurring `asyncio` workers, plus two one-shot startup tasks (an initial intel auto-seed and a tactical-index auto-populate check) that run once and exit. Workers read and write `MatchState` and publish to Redis pub/sub only; they do not call each other directly. Momentum and Tactical never reach the Agent Layer — momentum is pure EWMA/logistic arithmetic and Tactical has no generative step. `ScoreTracker`, the momentum state dictionary, and the `IsolationForest` rolling windows are process-local. Scaling requires running additional independent stacks against a shared Redis instance, not adding Uvicorn workers.



## Evaluation Results

Offline and semi-live evaluations of the core ML components, run against real StatsBomb data, live tournament simulations, and production query patterns. Scripts: `ml/eval_*.py`.

> **Data note:** StatsBomb's open-data World Cup coverage is limited to the 2018 and 2022 tournaments (128 total matches).

### Common Random Numbers: Counterfactual Variance Reduction

| Metric | CRN (shared seed) | Independent seeds |
|:---|---:|---:|
| Δ champ prob, mean | +0.03335 | +0.03447 |
| Δ champ prob, std | 0.00212 | 0.00469 |
| **Variance reduction** | **4.9x** | — |

Sharing a seed between the baseline and counterfactual simulation cuts estimator variance 4.9x. Independent seeding requires roughly 5x more simulations to reach the same precision on the event-impact delta.

### Narrative Anomaly Detection: IsolationForest Threshold Sweep

| Threshold | Precision | Recall | F1 | FA/topic-day |
|:---|---:|---:|---:|---:|
| −0.20 | 0.000 | 0.000 | 0.000 | 0.00 |
| −0.15 | 1.000 | 0.097 | 0.178 | 0.00 |
| **−0.10 (production)** | **0.995** | **0.601** | **0.749** | **0.01** |
| −0.05 | 0.879 | 0.964 | 0.920 | 0.41 |
| +0.00 | 0.304 | 1.000 | 0.466 | 7.06 |
| +0.05 | 0.073 | 1.000 | 0.136 | 39.17 |

The production threshold (−0.10) operates at 99.5% precision and 60.1% recall, 0.01 false alarms per topic-day. Lower thresholds trade precision for recall: −0.05 reaches 96.4% recall at 0.41 false alarms per topic-day; +0.05 reaches 100% recall at 39.17 false alarms per topic-day.

### Hybrid Retrieval: Recall/Precision/MRR/NDCG @ K=5, Alpha Sweep

Corpus: 96 docs · Queries: 36 · Encoder: `all-MiniLM-L6-v2` (production)

| Alpha | Recall@5 | Precision@5 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| 0.00 (BM25-only) | 1.00 | 0.20 | 0.766 | 0.824 |
| 0.25 | 1.00 | 0.20 | 0.789 | 0.842 |
| 0.50 | 1.00 | 0.20 | 0.845 | 0.884 |
| **0.75 (production)** | **1.00** | **0.20** | **0.944** | **0.959** |
| 1.00 (dense-only) | 1.00 | 0.20 | 0.944 | 0.959 |

Alpha=0.75 ties with pure dense retrieval (alpha=1.00) on MRR (0.944) and NDCG@5 (0.959).

### In-Play Model Calibration: Real World Cup Matches (n=12)

| Checkpoint | Log-loss | Brier |
|:---|---:|---:|
| Pre-match prior | 1.044 | 0.626 |
| Base rate | 1.099 | 0.667 |
| Minute 15 | 1.096 | 0.653 |
| Minute 45 | 0.965 | 0.554 |
| Minute 75 | 0.613 | 0.372 |

Log-loss falls monotonically from minute 15 (1.096) to minute 75 (0.613), dropping below the pre-match prior (1.044) by minute 75 reflecting the model correctly gaining information as the match progresses 



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
 worldcup26.ir            StatsBomb Open Data           The Odds API
 (score, status, clock)   (WC 2018/2022 events)         (bookmaker odds)
        │                          │                          │
        ⭣                          ⭣                          ⭣
 hybrid_producer.py     rag_indexer.py / tactical_indexer.py   odds_api_client.py
 (30s poll)              momentum_trainer.py / backtest_elo    (on demand, 900s TTL cache)
        │               (offline, on demand per proxy match)          │
        ⭣                          ⭣                          ⭣
 status normalize        event replay, feature extraction     consensus average
 scorer parse            narrative + PPDA document build       Shin de-vig
 Elo-distance proxy match
        │                          │                          │
        ⭣                          ⭣                          ⭣
   MatchState / TeamStats    Weaviate (NarrativeArcs,     prior_builder.py
   MatchEvent                TacticalProfiles)            (W/D/L probabilities)
        │                                                       │
        └───────────────────────┬───────────────────────────────┘
                                 ⭣
                              Redis
                    match:{id}:state, match:{id}:momentum,
                    match:{id}:intel:*, predict:*




 Mastodon      Bluesky      Google Trends      Wikipedia
 (search)      (searchPosts)  (pytrends)       (recentchanges)
        │           │              │                │
        └───────────┴──────┬───────┴────────────────┘
                            |
              narrative_spike_detector.py (60s tick)
              per-source rate, mock fallback, four-dim vector
                            ⭣
                          Redis
              narrative:spike:*, narrative:trending:latest
```

### External Data Sources

* **Live match state**: `worldcup26.ir`, polled every 30 seconds, providing score, status, kickoff time, and scorer text. Response shape varies (a bare list, or a `games`/`matches`/`data`/`fixtures` key) and is normalized to one shape before anything downstream sees it.
* **Historical match data**: StatsBomb Open Data, competition 43, seasons `[106, 3]` (WC 2022, WC 2018). Full shot/pass/pressure/card event streams, fetched on demand and reused across five paths: live-fixture proxying, RAG narrative extraction, tactical fingerprint indexing, momentum-model training, and the StatsBomb-proxy lineup fallback. The Elo calibration backtest is a lighter, sixth consumer of the same dataset — it uses only match-level scores, not the full event streams.
* **Market data**: The Odds API, fetched on demand and cached for 900 seconds. Per-bookmaker decimal odds for head-to-head markets, averaged into one consensus price per fixture before de-vigging.
* **Lineup data**: Zafronix roster API, API-Sports fixture lineups, and a StatsBomb-proxy starting-XI fallback, resolved in that order per request, with a PPDA-estimated formation as the final fallback when none of the three return usable data.
* **Social/search signals**: Mastodon, Bluesky, Google Trends, and Wikipedia, read concurrently every 60 seconds per tracked topic. The only category with its own dedicated normalization and detection pipeline, covered below.

### External Signals

1. Mastodon search, Bluesky `searchPosts`, Google Trends via `pytrends`, and Wikipedia's recent-changes feed are read concurrently for every tracked topic on a 60-second cadence.
2. Any source that is unauthenticated, rate-limited, or unreachable substitutes a mock generator: a sine wave whose phase is deterministically seeded per topic, plus unseeded Gaussian noise. Only the baseline shape is reproducible per topic — the noise itself varies run to run.
3. Each reading is converted into a per-source activity value — a rolling 24-hour post count for Mastodon, an unwindowed search-result count for Bluesky, a Google Trends index, and a true edits-per-minute rate for Wikipedia — and unified into one four-dimensional feature vector per topic per tick, tagged `"live"` or `"mock"` per source.
4. That vector is appended to the topic's rolling 72-hour window and scored against a per-topic anomaly model, producing either a spike or an entry in the continuously-updated trending ranking.
5. Downstream consumers are the narrative agent (for retrieval-grounded generation) and the narrative API routes that serve the resulting feed to the frontend.

### Schemas

**MatchState** (Pydantic, `api/schemas/schema.py`). Rebuilt every 30 seconds by `hybrid_producer.py`. Stored at `match:{id}:state`, TTL-scoped. The canonical object every worker, route, and agent reads.

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
    stats_source: str = "unknown"
    stats_proxy_match_id: Optional[int] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

**MatchEvent** (Pydantic, nested in `MatchState.events`). Constructed directly by `hybrid_producer.py`, either from parsed scorer strings or synthesized from score deltas when no scorer name is available.

```python
class MatchEvent(BaseModel):
    elapsed: int
    extra: Optional[int] = None
    team_id: int
    team_name: str
    player_name: Optional[str] = None
    type: str
    detail: Optional[str] = None
```

**TeamStats** (Pydantic, nested in `MatchState.home_stats` / `away_stats`). Built from replayed StatsBomb proxy events, not from a live provider.

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

**MomentumState** (dataclass, `ml/momentum_model.py`, runtime state, not the Redis wire payload). Held in-process per active fixture. The Redis payload written by `momentum_worker.py` is a plain dict, not validated through the separate Pydantic `MomentumSnapshot` model defined in `ml/schemas/momentum_schema.py`, which is unused elsewhere in the codebase.

```python
@dataclass
class TeamMomentumState:
    ewma_poss: float = 50.0
    ewma_pass_acc: float = 75.0
    ewma_pressure: float = 0.1
    shot_window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SLOTS))
    last_shots_total: int = 0
    last_shots_on: int = 0

@dataclass
class MatchMomentumState:
    fixture_id: int
    home: TeamMomentumState = field(default_factory=TeamMomentumState)
    away: TeamMomentumState = field(default_factory=TeamMomentumState)
```



## The ML Core

Every prediction traces to an explicit, inspectable formula.

**Team strength to outcome probability.** Elo expectation:

$$
E_a=\frac{1}{1+10^{\frac{R_b-R_a}{400}}}
$$

Where:

- $E_a$ = expected score for Team A
- $R_a$ = Elo rating of Team A
- $R_b$ = Elo rating of Team B

Converted into a full three-outcome distribution with a rating-gap-sensitive draw model:

$$
p_{\text{draw}}=\mathrm{clip}\left(0.25e^{-\Delta R/450}+0.05,\;0.10,\;0.30\right)
$$

$$
p_{\text{win}}=(1-p_{\text{draw}})E_a
$$

$$
p_{\text{loss}}=(1-p_{\text{draw}})(1-E_a)
$$

where $\Delta R = |R_a - R_b|$ is the absolute rating gap between the two teams.

---

**Market odds** are preferred over Elo when available and de-vigged with **Shin's (1993) method**, which discounts longshot prices less aggressively than a proportional split:

$$
p_i=\frac{\sqrt{z^2+4(1-z)\left(\frac{1}{o_i}\right)^2}-z}{2(1-z)}
$$

where

$$
z=\frac{\Omega}{\Omega+2}
$$

$$
\Omega=\sum_i\frac{1}{o_i}-1
$$

- $p_i$ = fair, de-vigged probability for outcome $i$ (home / draw / away)
- $o_i$ = quoted decimal odds for outcome $i$
- $\Omega$ = overround — the sum of implied probabilities across all outcomes, minus 1
- $z$ = Shin's insider-trading parameter, derived from $\Omega$

---

**In-play win probability** updates live from score, minute, and red cards using two independent Poisson goal processes:

$$
\lambda_{\text{home}}=\max\left(0.02,\;1.3f(1+0.65\sigma)\right)\qquad\lambda_{\text{away}}=\max\left(0.02,\;1.3f(1-0.65\sigma)\right)
$$

$$
P(g_h,g_a)=\mathrm{Pois}(g_h;\lambda_{\text{home}})\cdot\mathrm{Pois}(g_a;\lambda_{\text{away}}),\qquad g_h,g_a\in[0,8]
$$

where

- $\lambda_{\text{home}}$, $\lambda_{\text{away}}$ = expected in-play goal rate for each team over the remaining match
- $f$ is the fraction of the match remaining.
- $\sigma$ is the pre-match win-probability differential.
- $g_h$, $g_a$ = candidate home/away goal counts for the remainder of the match, enumerated over $[0,8]$
- $P(g_h,g_a)$ = joint probability that the match finishes with exactly $g_h$ further home goals and $g_a$ further away goals



---

**Momentum** is EWMA-smoothed and scored by logistic regression, executing in microseconds with fully inspectable inputs:

$$
\mathrm{EWMA}_t=\alpha x_t+(1-\alpha)\mathrm{EWMA}_{t-1},\qquad\alpha=0.3
$$

$$
P(\text{goal within 5 min})=\sigma\left(\beta_0+\sum_i\beta_i\,\mathrm{feature}_i\right)+\sum_k\mathrm{bump}_k(0.8)^{\Delta t_k}
$$

- $x_t$ = raw input signal at tick $t$ (possession, pass accuracy, or pressure)
- $\mathrm{EWMA}_{t-1}$ = previous smoothed value of that signal
- $\alpha$ = smoothing factor
- $\sigma(\cdot)$ = logistic sigmoid function
- $\beta_0$ = intercept; $\beta_i$ = trained coefficient for $\mathrm{feature}_i$ (EWMA-smoothed pressure, possession, pass accuracy, match minute, score differential)
- $\mathrm{bump}_k$ = initial magnitude of the $k$-th active event bump (goal or red card)
- $\Delta t_k$ = ticks elapsed since event $k$ occurred

$\Delta t_k$ is measured in 30-second ticks (two per minute). The bump decays to roughly 64% of its initial value after one minute and under 11% after five. Goals and red cards inject this decaying additive bump directly into the output. Coefficients are trained offline using batch gradient descent on real StatsBomb matches and are promoted only when they outperform a base-rate log-loss baseline on held-out data.

---

**Tournament simulation** resolves every simulated match and knockout round for all $N$ runs simultaneously through vectorized NumPy operations rather than sequential loops. Every reported stage probability carries a 95% confidence interval:

$$
\text{margin}=1.96\sqrt{\frac{\hat{p}(1-\hat{p})}{N}}
$$

- $\hat{p}$ = simulated probability estimate for a given stage outcome (e.g. reaching the quarterfinal)
- $N$ = number of simulation runs
- $\text{margin}$ = half-width of the 95% confidence interval around $\hat{p}$

---

**Tactical identity** is derived through feature engineering and cosine retrieval rather than a trained classifier:

$$
\mathrm{PPDA}=\frac{\text{Opponent completed passes in press zone}}{\text{Defensive actions in press zone}}
$$

$$
\text{press}\_\text{intensity}=0.7\,\min\left(1,\frac{8}{\mathrm{PPDA}}\right)+0.3\,\min\left(1,\frac{\text{pressures}}{150}\right)
$$

- $\mathrm{PPDA}$ = passes per defensive action — a lower value indicates more aggressive pressing.
- $\text{pressures}$ = count of raw pressure events recorded in the press zone.

---

**Narrative anomaly detection** fits a separate `IsolationForest` (`contamination=0.05`, 100 estimators) per tracked topic, refit every 30 ticks on a rolling 72-hour window of four-source activity, scoring every new observation against it. Per-topic baselines correct for activity volume differences between heavily-followed and lightly-followed teams:

$$
\mathrm{severity}=\mathrm{clip}\left(\frac{-s-0.10}{0.5},0,1\right)
$$

where $s$ is the raw anomaly score. A spike fires when:

$$
s < -0.10
$$

outside a 300-second per-topic cooldown window.

<br />



## The Agent Layer

Every agent scores or reconstructs a deterministic result before any generation step. The language model narrates a result that has already been computed; it does not decide what happened. Four of five agents retrieve grounding context from Weaviate before narrating. The Counterfactual Agent narrates directly from its simulation output with no retrieval step. Four of five fall back to a computed-number template when generation is unavailable. The Tactical Agent has no generative step and returns nothing; the interface falls back to live stats. All five agents below are fully implemented in the current codebase; none are stubs or placeholders.

### Match Intelligence Agent

* Runs every 30 seconds against the current match state and momentum snapshot.
* Scans the event stream for uncovered goals and cards, and factors in momentum delta and xG-versus-scoreline divergence even when no discrete event has occurred.
* On quieter ticks, falls back to a periodic tactical read of live possession and pressure.
* Whichever condition fires determines the retrieval target. Event narration pulls from the goal/red-card narrative collection filtered by event type; xG-divergence narration pulls from the same collection unfiltered. Tactical narration pulls from the pressing-fingerprint collection.
* Retrieved passages are folded into a prompt built from the real scoreline, xG, and momentum figures.
* Generation runs through local-first inference whenever a key event (a goal or card) has occurred, a periodic tactical read is due, or the computed relevance score clears 0.50; ticks that satisfy none of the three return a numeric template built from the same figures.

### Counterfactual Agent

* Fires on a goal, own goal, penalty, yellow, red, second-yellow card, or substitution, subject to a 45-second minimum gap per fixture. Own-goal, penalty, and second-yellow variants are handled in code but are not currently produced by the live ingestion pipeline, which only ever emits goal/yellow/red/substitution events; a substitution consumes the gap and runs the full simulation pair but produces no state change.
* Reverses the triggering event's effect on score and card counts to recover the state immediately beforehand.
* Converts the resulting in-play probability swing into a bounded Elo adjustment.
* Runs two 20,000-run tournament simulations, one under the pre-event adjustment and one under the post-event adjustment, sharing a single deterministic seed; the only source of divergence between the two runs is the adjustment itself.
* Ranks the per-team probability differences across both runs, and hands the largest shifts, together with the in-play win-probability swing, to the same local-first generation path.
* A fallback template reconstructs the identical explanation from the same numbers if generation is unavailable.

### Tactical Agent

* Has no generative step anywhere in its path.
* Converts live possession, shot volume, and pass accuracy into a short style descriptor, a proxy for the pressing statistics the live feed does not expose.
* Embeds that descriptor and retrieves the nearest historical pressing fingerprint by cosine similarity, along with up to two runner-up matches for comparison.
* The best available match is always returned when Weaviate holds any indexed profile — there is no similarity cutoff. It returns nothing only when Weaviate is unreachable or `TacticalProfiles` is empty; the interface falls back to displaying live statistics directly.

### Narrative Intelligence Agent

* Consumes spikes and trending rows produced by the anomaly-detection layer, which aggregates four social/search sources per topic every 60 seconds and scores each topic against its own per-topic `IsolationForest`.
* Builds a query from the topic name and its activity summary, and retrieves historical precedent from the narrative collection.
* Constructs a prompt that states explicitly which sources moved and which stayed flat: one source moving alone reads as a viral post, several moving together reads as a real-world event. The prompt instructs the model to reason from that shape rather than restate volume.
* Output passes through a banned-phrase filter; a rule-based fallback performs the identical reasoning without a model whenever generation fails or the result doesn't clear validation.

### Briefing Agent

* Triggers inside a fixed pre-kickoff window, once per match status, or on demand, using only the two team names and retrieved historical precedent. No live match state feeds this path.
* Retrieves precedent from the narrative collection and is instructed not to invent beyond it.
* Falls back to quoting the first 120 characters of the retrieved passage's first line, or stating plainly that none was available, if generation fails.
* This is the one agent that calls Groq directly and never attempts local inference, an inconsistency with the other four rather than a deliberate design choice.

Three of the five agents run continuously during live play through local-first inference, keeping goal, card, and anomaly narration off network latency and a metered API by default. Every agent falls back to a template built from the same computed numbers when generation fails, producing plain language instead of an empty response or an invented statistic.



## RAG + Knowledge Infrastructure

### Knowledge Construction

1. Historical StatsBomb event streams are the raw material for both retrieval collections.
2. Goal, red-card, and momentum-shift documents are generated from historical matches as natural-language text with structured metadata (`match_id`, `competition`, `season`, `minute`, `event_type`).
3. PPDA and pressing-zone statistics are computed per team per match and written as a separate document set.
4. Both indexing scripts run offline via a manual CLI entry point. The tactical indexer additionally has an auto-index-on-empty path invoked once at application startup, self-populating a fresh deployment without a manual step.

### Embedding Pipeline

1. Each extracted document is a short, self-contained natural-language passage; the source documents require no separate chunking step.
2. Embeddings come from `sentence-transformers/all-MiniLM-L6-v2`, loaded once per consuming module as a lazy singleton, with `normalize_embeddings=True`.
3. Vectors are inserted into Weaviate alongside their structured metadata in the same write.
4. Embedding calls run on a dedicated thread pool, isolated from the main event loop.

### Hybrid Retrieval

1. Dense retrieval scores cosine similarity over the normalized sentence embeddings.
2. BM25 retrieval scores keyword overlap over the same document set.
3. One hybrid call blends both, weighted 75% toward the dense vector (`alpha=0.75`).
4. Narrative queries additionally filter by `event_type` (`goal`, `red_card`) when the calling agent already knows the kind of moment it is narrating, improving precision over vector search alone.

### Vector Database

Weaviate 1.27 is configured with no built-in vectorizer. Every insert supplies its own precomputed vector; Weaviate never computes an embedding itself.

* `NarrativeArcs` holds goal, red-card, and momentum-shift narratives, and also receives store-back writes from the narrative agent when explicitly enabled.
* `TacticalProfiles` holds PPDA and pressing fingerprints per team, per match.
* No exact indexed document count is committed anywhere in the repository; the indexing scripts cap ingestion at 500 matches by configuration, but the actual count depends on how many matches were successfully processed on a given run and is only knowable by querying a live instance.

### Grounded Generation

1. Retrieval fetches up to 5 candidates per query; 2-3 of the top-ranked documents are interpolated directly into the prompt text.
2. Every prompt also states the real computed numbers (scoreline, xG, probability shifts, activity levels) alongside the retrieved text.
3. The Narrative and Briefing agents explicitly instruct the model not to invent facts beyond what the prompt provides. The Match Intelligence Agent's prompts inline the same real numbers but do not carry this explicit instruction.
4. If the model is unavailable, a deterministic template reuses the identical numbers, producing plainer language rather than a fabricated fact.



## Real-Time Intelligence Runtime

### Worker Architecture

| Worker | Cadence | Responsibility |
|:---|:---|:---|
| Producer | 30s | Polls the live feed, reconstructs `MatchState` |
| Momentum | 30s | EWMA and logistic inference |
| Intel | 30s | Event, xG, and tactical scoring; live narration |
| Counterfactual | 30s | Trigger detection; paired Monte Carlo; bracket-impact narration |
| Tactical | 120s | Style descriptor; cosine match against history |
| Briefing | 300s | Pre-match window gating; tactical preview |
| Narrative | 60s | Social signal aggregation; anomaly scoring; arc synthesis |

Only these seven recurring workers exist in the codebase. Two one-shot startup tasks additionally run once at boot (an initial intel auto-seed and a tactical-index auto-populate check), and `predict.py` spawns short-lived on-demand tasks per simulation trigger.

### Redis State Layer

* Match state, momentum snapshots, intel feeds, counterfactual feeds, narrative spikes, and simulation results all live in Redis, the only persistence layer in the system.
* Every key is TTL-scoped: 43,200s for a pre-kickoff `MatchState`, 3,600s while live, up to 2,592,000s for completed counterfactual/briefing history.
* Five Redis pub/sub channels carry state-change notifications: `match_update`, `momentum_update`, `intel_update`, `counterfactual_update`, `narrative_spike`. Tactical and briefing workers write state but publish nothing, relying on frontend polling instead.

### Event Sourcing (Partial Implementation)

* There is no durable, replayable event log. `MatchState` is overwritten in place on every poll, not appended to.
* Replay-restart detection rehydrates score-tracking state from the last-persisted `MatchState` on worker restart, preventing an already-shipped goal event from re-firing.
* The intel and counterfactual workers separately detect when the live match minute has rewound relative to their cached progress, and explicitly flush their per-fixture caches rather than mixing old and new narratives.
* This is state-reconstruction-on-restart, not event sourcing in the CQRS sense.

### Async Execution

1. All seven workers, plus two one-shot startup tasks, are `asyncio` tasks sharing one event loop and one Redis connection; none run in a separate OS process.
2. Each worker's tick is wrapped in a try/except block, so one fixture's malformed state or one worker's exception never halts processing elsewhere.
3. CPU-bound and blocking work is kept off the event loop via four dedicated thread pools: tournament simulation, counterfactual simulation pairs, embedding, and blocking HTTP calls (Google Trends).

### Streaming Layer

* SSE channels (`match_update`, `momentum_update`, `intel_update`, `counterfactual_update`, `narrative_spike`) subscribe to their Redis pub/sub channel first, then perform an initial key read, delivering current state immediately to a client connecting mid-match, and carry a periodic heartbeat distinguishing inactivity from a dropped connection.
* Every frontend hook relies on the browser's native `EventSource` reconnect logic for failure recovery.
* The Mastodon/social data path is not part of any SSE channel. Comment samples captured during the 60-second social-signal tick are cached in-process and pushed into a plain Redis list (600s TTL, 19-item cap, deduplicated by permalink or text hash). The frontend reads this by polling a REST endpoint every 15 seconds. No comment-streaming SSE route exists anywhere in the codebase. The sample cache populates only on live (non-mock) provider reads; without social API credentials, the comment UI displays nothing while anomaly detection continues operating on mock signal data.



## Deep Dive: The Three Core Intelligence Engines

These three subsystems are treated as first-class reasoning and simulation systems, not simple LLM agents.

### 1. Narrative Intelligence Hub

A 60-second worker aggregates Mastodon, Bluesky, Google Trends, and Wikipedia activity into per-topic feature vectors for anomaly detection, decoupled from `MatchState` and the live-match workers. Multi-source activity shape, not single-value volume, distinguishes a real-world event from a single viral post, feeding both the anomaly feed and the continuously-updated trending view.

Four sources are read concurrently every 60 seconds, per tracked topic:

1. Mastodon authenticated full-text search: a single 401 disables the source immediately, and a separate 5-consecutive-failure counter (any exception type, not only authentication failures) also falls back to mock.
2. Bluesky session-token authenticated search, re-authenticating once on a token expiry before falling back to mock.
3. Google Trends via `pytrends`, executed on a blocking I/O thread pool since the library is not async-native.
4. Wikipedia recent-changes edit velocity, used as a proxy for reference and news activity.

Each source's mock fallback substitutes a sine wave whose phase is deterministically seeded per topic, plus unseeded Gaussian noise and occasional injected spikes. Only the baseline shape is reproducible per topic, not the exact output.

Detection pipeline:

1. Every 60 seconds, all four sources are read concurrently for every tracked topic.
2. Each reading is appended to that topic's own rolling 72-hour window (4,320 ticks).
3. Every 30th tick, a fresh `IsolationForest` (`contamination=0.05`, 100 estimators) is refit on that topic's window alone. Models are never shared or pooled across topics; baseline activity varies by roughly an order of magnitude between a heavily-followed team and a lightly-followed one.
4. Every tick, the newest observation is scored against the cached model.
5. A score below `-0.10`, outside a 300-second per-topic cooldown, triggers a spike.

A source is marked as "driving" a spike when its current value exceeds 1.2× its own rolling mean and clears an absolute activity floor. Both conditions must hold. Spikes are rate-limited by the score threshold and a 300-second cooldown. The trending list is a continuously refreshed ranking of every tracked topic by activity lift, computed independently of spike status.

A selected spike or trending topic generates a query from its topic name and activity summary, embedded and matched against the narrative collection for historical precedent. The prompt states which sources moved and which stayed flat: concentration in one source indicates a viral post, movement across multiple sources indicates a real-world event. A banned-phrase list and few-shot examples constrain the prompt; a post-generation check discards and re-templates any response containing banned phrasing. Each execution produces an anomaly score between 0 and 1, source attribution, and a retrieval-grounded narrative.

### 2. Counterfactual What-If Engine

* Tournament impact of a goal or red card is not derivable from the scoreline alone: an early goal in a low-stakes group match and a late winner in a decisive fixture carry different implications for 48 teams' championship odds. The tournament simulator alone reports current odds, not the marginal effect of a specific moment.
* Match events are translated into tournament-wide probability shifts through paired Monte Carlo simulation, powering the match page's "What If?" tab and the bracket-impact feed on the predictor page.
* A trigger event (goal, own goal, penalty, yellow, red, second-yellow card, or substitution) is detected in the match event stream, and the state immediately before it is reconstructed by reversing its effect on score and card counts. A substitution has no score/card effect to reverse; the pipeline runs without producing a divergence.
* In-play win probability is computed for both the pre- and post-event states, and the difference is converted into a bounded Elo adjustment.
* Two full tournament simulations, 20,000 runs each by default, are executed: one under the pre-event adjustment, one under the post-event adjustment. The divergence in each team's championship probability across the two runs is the reported tournament impact.
* Both simulations share one deterministic seed derived from the fixture, minute, event type, and team, producing identical random draws across both runs except for the strength adjustment. A shared deterministic seed isolates the event's effect from Monte Carlo sampling noise.
* Independently seeded runs require substantially more simulations to distinguish an event's effect from sampling variance. When the computed adjustment is negligible, the second simulation is skipped.

```
Δp_i = p_i^after - p_i^before      path_shift = min(1, Σ|Δp_i| / 2)
```

* The model perturbs team strength rather than replaying alternate match events. It measures the effect of an estimated strength change, not the outcome of an exact goal replayed in an alternate match.

### 3. Tournament Simulation Engine

* Tournament probabilities are computed by simulating the complete group and knockout stages for all 48 teams across 11 outcome stages. Third-place advancement and cross-group knockout draws have no closed-form solution, requiring full bracket simulation.
* The engine exposes tournament probabilities through the prediction API and serves as the simulation backend for the counterfactual pipeline, with no LLM or retrieval dependency anywhere in the path.
* Every simulated group match, for all `N` requested runs, is resolved in one vectorized NumPy pass rather than sequential loops.
* Group order ties are broken with a composite key combining points and a noise term; knockout rounds reuse one precomputed team-advancement matrix across all rounds via array indexing.
* Measured directly rather than estimated: 50,000 simulations complete in 0.70 seconds, and 10,000 complete in 0.13 seconds.
* The engine's own internal documentation estimates roughly 8 seconds for 50,000 runs. The measured figure is about 10 times faster on current hardware, a stale estimate rather than a correctness issue.
* Vectorized execution reduces tournament simulation latency to 0.70 s for 50,000 runs, enabling live inference for both the tournament predictor and the counterfactual engine's on-demand, in-match simulation requirements.



## Performance Optimizations

* **Vectorized simulation.** `tournament_sim.py` resolves every group match in one NumPy pass and each of the five knockout rounds in a further vectorized pass (cumulative probability draws, `np.add.at`, `np.maximum.at`, fancy indexing into a precomputed advancement matrix), instead of looping per simulation. Vectorized execution keeps 50,000 runs at 0.70 seconds instead of scaling linearly with simulated matches.
* **Isolated thread pools instead of one shared pool.** `ml/executors.py` defines four dedicated `ThreadPoolExecutor`s (`SIM_EXECUTOR`, `CF_SIM_EXECUTOR`, `EMBED_EXECUTOR`, `IO_EXECUTOR`). A burst of counterfactual simulations cannot delay a tournament prediction request, and CPU-bound embedding never blocks the asyncio event loop handling SSE connections.
* **TTL caching with stale-on-failure fallback.** `odds_api_client.py` caches bookmaker odds for 900 seconds and serves the last successful snapshot if a refresh fails, rather than blocking every simulation or prediction request on a fresh external call.
* **Redis-level caching for expensive external lookups.** `lineups.py` caches Wikipedia player photos for 7 days and Zafronix rosters for 6 hours; `tactical.py` caches computed tactical fingerprints for 600 seconds, avoiding repeated similarity searches on every page load.
* **Change-detection before every write and publish.** `hybrid_producer.py`'s `_persist_if_changed` compares serialized `MatchState` payloads and writes to Redis and publishes to `match_update` only when the state actually changed, keeping idle fixtures from generating SSE traffic every 30 seconds.
* **Per-fixture and per-topic rate limiting on expensive generation paths.** The counterfactual agent enforces a 45-second minimum gap per fixture before running a new simulation pair; the narrative detector enforces a 300-second cooldown per topic before re-firing a spike; the match intelligence agent skips its LLM call entirely below a relevance-score threshold. Each avoids a Monte Carlo run or an inference call on a redundant or low-value tick.
* **Common Random Numbers doubles as a compute optimization.** When a counterfactual event's computed Elo adjustment is negligible, the second (post-event) simulation is skipped entirely and the baseline result is reused, avoiding a second 20,000-run simulation for events that would not move the outcome.
* **Lazy, shared model loading.** Every agent that embeds text loads `all-MiniLM-L6-v2` once as a module-level singleton on first use, rather than reloading it per request.
* **Batched embedding during offline indexing.** `rag_indexer.py` and `tactical_indexer.py` encode documents in batches of 50 rather than one at a time, reducing per-call model overhead across roughly 500 matches.
* **Process-local result caching alongside Redis.** `predict.py` mirrors the latest tournament prediction in a process-local variable, skipping a Redis round-trip on repeated reads of an already-cached result.
* **Single combined hybrid retrieval call.** Weaviate's `hybrid_search` blends dense and BM25 scoring in one query per agent call, instead of running two separate retrieval passes and merging results client-side.



## Repository Structure

```
backend/
├── agents/
│   ├── briefing_agent.py            Pre-match briefing generation, Groq only, no local-first tier
│   ├── counterfactual_agent.py      Paired CRN Monte Carlo simulation and bracket-impact narration
│   ├── match_intel_agent.py         Live event/xG/tactical scoring and narrative generation
│   ├── narrative_arc_agent.py       Evidence-grounded narrative synthesis for detected spikes
│   ├── narrative_spike_detector.py  Four-source signal aggregation and per-topic IsolationForest
│   ├── narrative_topics.py          Fixture-aware dynamic topic tracking, defined but unused
│   ├── ollama_client.py             Local Ollama call with Groq fallback, shared by three
│   │                                agents (match intel, counterfactual, narrative arc)
│   ├── rag_indexer.py               Offline StatsBomb narrative document extraction and indexing
│   ├── tactical_agent.py            Live style descriptor and cosine match against TacticalProfiles
│   └── weaviate_client.py           Weaviate connection, collection schema, hybrid search wrapper
├── api/
│   ├── main_hybrid.py               FastAPI app, lifespan worker orchestration, single-process guard
│   ├── routes/
│   │   ├── _security.py             Trigger-token dependency for debug endpoints
│   │   ├── _sse.py                  Shared Redis pub/sub-backed SSE generator
│   │   ├── briefing_routes.py       Briefing feed and trigger endpoints
│   │   ├── comment_sampler.py       Duplicate comment-sample storage, not mounted in the app
│   │   ├── counterfactual_routes.py Counterfactual feed, prediction, and live-prob endpoints
│   │   ├── group_table.py           Live group-stage standings from completed fixtures
│   │   ├── intel.py                 Intel feed and SSE stream endpoints
│   │   ├── lineups.py                Tiered lineup resolution across four providers
│   │   ├── match.py                 Fixture list and summary endpoints
│   │   ├── match_stream.py           Match-state SSE stream
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
│       ├── hybrid_producer.py       30s live-feed poll and MatchState reconstruction
│       ├── intel_worker.py          30s event/momentum scoring and narration dispatch
│       ├── momentum_worker.py       30s EWMA and logistic inference per fixture
│       ├── narrative_worker.py      60s signal aggregation, anomaly scoring, arc synthesis
│       └── tactical_worker.py       120s tactical fingerprint refresh
├── ml/
│   ├── backtest_elo_wdl.py          Elo calibration backtest against real WC 2018/2022 results
│   ├── executors.py                 Dedicated thread pools for simulation, embedding, and I/O
│   ├── in_play.py                   Poisson in-play W/D/L model and Elo delta conversion
│   ├── momentum_model.py            Online EWMA and logistic momentum inference
│   ├── momentum_trainer.py          Offline momentum coefficient training and validation gate
│   ├── odds_api_client.py           Odds API client with TTL cache and stale-on-failure fallback
│   ├── prior_builder.py             Elo-to-WDL and Shin de-vig probability construction
│   ├── schemas/
│   │   └── momentum_schema.py       Pydantic momentum schema, defined but unused at runtime
│   ├── statsbomb.py                 Shared StatsBomb parsing constants and helpers
│   ├── tactical_indexer.py          Offline PPDA feature extraction and TacticalProfiles indexing
│   ├── team_names.py                Team name alias mapping across three naming systems
│   ├── tournament_sim.py            Vectorized Monte Carlo group and knockout simulator
│   └── wc_2026_config.py            Static 48-team Elo configuration, groups, and bracket
├── Dockerfile
└── requirements.txt

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
├── lib/flag.ts
├── types/match.ts, predict.ts       TypeScript mirrors of the backend Pydantic schemas
└── package.json, next.config.js, tsconfig.json

docker-compose.yml     Redis 7 + Weaviate 1.27 + API, single replica; UI runs separately
weaviate_data/         Bind-mounted Weaviate volume, runtime data, not source
```



## Setup

### Prerequisites

* Docker (for Redis 7 and Weaviate 1.27)
* Python 3.12
* Node.js (for the Next.js UI)
* Ollama, optional, required only to engage the local-first LLM tier (`mistral:7b-instruct-q4_K_M`)

### Install and Run

```bash
# infrastructure
docker compose up -d redis weaviate

# backend
cd backend
pip install -r requirements.txt
ollama pull mistral:7b-instruct-q4_K_M   # optional
uvicorn api.main_hybrid:app --reload --port 8000

# frontend
cd ui
npm install
npm run dev
```

### Populate the Knowledge Base (offline, one time)

```bash
python -m agents.rag_indexer        # NarrativeArcs, ~500 StatsBomb matches
python -m ml.tactical_indexer       # TacticalProfiles, also auto-runs on first empty-collection startup
```

### Run the Elo Calibration Backtest

```bash
PYTHONPATH=. python ml/backtest_elo_wdl.py --json report.json
```

## Tech Stack

| Category | Stack |
|:---|:---|
| Languages | Python 3.12, TypeScript |
| ML | NumPy, scikit-learn (`IsolationForest`), `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Backend | FastAPI, Uvicorn, `sse-starlette`, Pydantic v2, `httpx`, `redis.asyncio`, `weaviate-client` |
| Frontend | Next.js 14 App Router, React 18, `next-themes` |
| Data | Redis 7, Weaviate 1.27 |
| Infra | Docker Compose, `python-dotenv`, `python:3.12-slim` base image, `gcc` (build-time, scientific Python wheels) |
| Dev / Testing | `pytest`, `ruff` |
| External data | StatsBomb Open Data, `worldcup26.ir`, The Odds API, API-Sports, Zafronix, Mastodon, Bluesky, Google Trends (`pytrends`), Wikipedia REST |
| AI | Ollama (`mistral:7b-instruct-q4_K_M`), Groq (`llama-3.3-70b-versatile`) |


## Limitations

* **Tournament outcomes are inherently stochastic.** The simulator reports probability distributions with 95% confidence intervals, not deterministic forecasts. The Elo backtest measured on this codebase loses to a naive base-rate baseline overall, and only beats it after the ratings warm up across roughly half a match sequence.

* **The architecture is difficult to host on free-tier platforms.** Seven long-running background workers and in-memory EWMA/IsolationForest state require a persistently running process, not the request-scoped, auto-scaling functions most free serverless tiers provide.

* **Live state updates are bounded by a 30-second poll interval against a free, keyless feed with no push mechanism.** Events inside that window are not visible until the next poll cycle.

* **Social/search signal reads run against free or unauthenticated-fallback APIs with their own rate limits and timeouts.** Any of the four sources can silently degrade to a mock signal generator; the system tags this internally as live or mock but does not surface it in the UI.

* **Every "live" statistic for a 2026 fixture is a StatsBomb proxy from a real 2018 or 2022 match**, selected by Elo-distance similarity, not the actual game being played. Possession, shots, and xG shown for a live 2026 match describe a historically similar match, not the one on the pitch.

* **A hybrid, keyless-first data approach was chosen over a paid live-stats provider.** API-Sports and comparable commercial feeds gate meaningful request volume and full statistical coverage behind a paid plan. `API_SPORTS_KEY` is optional throughout the codebase and used only as a lineup fallback, never as the primary live data source. StatsBomb Open Data combined with the free `worldcup26.ir` feed avoids that dependency entirely, at the cost of the proxy-statistics limitation above.

* **Counterfactual analysis perturbs team strength rather than replaying an alternate version of the match.** It does not model the effect of a different tactical decision or substitution.

* **Retrieval-grounded generation reduces but does not eliminate hallucination risk.** A banned-phrase filter and prompt-level grounding constrain output; no independent faithfulness or groundedness score is computed for any generated narrative.

* **The single-process architecture is a scalability ceiling, not only a design preference.** Horizontal scale requires running independent stacks against shared Redis and Weaviate instances, not adding workers to one process.
