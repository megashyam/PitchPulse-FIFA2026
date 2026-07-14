# PitchPulse: WC2026 Live Football Analytics Engine

***

PitchPulse is a live match tracking, statistical inference, and AI narration engine built for the 2026 World Cup. It ingests bare-bones live score data and mathematically enriches it using historical statistics, vectorized Monte Carlo tournament simulations, and large language models (LLMs). The primary technical goal is to deliver real-time, push-driven analytical updates: live momentum, counterfactual bracket shifts, and tactical briefings, without relying on an expensive commercial live-stats feed. The retrieval and agent layers are **built from scratch: no LangChain, no LangGraph, no LlamaIndex, or any other RAG framework.** Every prompt, retrieval call, and fallback path is hand-written directly against the Weaviate client and the LLM providers.

![Demo GIF](data/demo.gif)

The backend operates entirely on a single OS process to preserve in-memory state, scaling horizontally via Redis pub/sub.

## Why This Project?

PitchPulse was built to solve a critical data availability problem for live sports applications. The free, keyless live feed (`worldcup26.ir`) only exposes the score, match status, and clock. It lacks possession, shots, expected goals (xG), or any other tactical statistic. To provide an advanced analytics experience, the system relies on an architecture that lazily pairs real 2026 fixtures with structurally similar historical matches from StatsBomb's open data sets. This enables complex ML and RAG features without commercial data licenses.

###

***

## Core Features

1. **Live Match Intelligence**

   * Runs on a 30-second worker tick against the current match state and momentum snapshot.

   * Scores every uncovered goal, card, and momentum shift for narrative relevance before deciding whether a language model call is warranted at all.

   * Low-scoring ticks return a numeric template directly; higher-scoring ticks retrieve grounding context from a vector store and generate a short reaction through local-first inference.

   * Output: a short narrative entry streamed to the match page as an SSE event, built from the same numbers used by the fallback template.

2. **The Counterfactual What-If Engine**

   * Triggers on any goal, own goal, penalty, or card event.

   * Reconstructs the match state immediately before the event and runs two 20,000-run tournament simulations under one shared random seed: one before the event, one after.

   * Bounded by a 45-second minimum gap per fixture to cap compute cost.

   * Output: a probability-shift percentage, ranked per-team deltas, and a narrative explanation traceable to that single simulation pair.

3. **The Tournament Simulation Engine**

   * Takes the static 48-team Elo configuration, optionally overridden by de-vigged market odds.

   * Resolves the entire group stage and knockout bracket for every requested run in one vectorized NumPy pass rather than sequential loops.

   * Executes on demand: triggered explicitly, or implicitly on first read after a cache expiry.

   * Output: per-team, per-stage advancement and championship probabilities with 95% confidence intervals; no language model or retrieval step anywhere in the path.

4. **Tactical Intelligence**

   * Builds a live style descriptor from possession, shot volume, and pass accuracy, since the live feed exposes none of the pressing statistics the historical index was built from.

   * Embeds that descriptor and retrieves the closest historical pressing fingerprint by cosine similarity against the indexed StatsBomb corpus.

   * No generation step exists in this path. The match, or nothing if no candidate clears the similarity threshold, is returned directly.

5. **The Narrative Intelligence Hub**

   * Runs a 60-second aggregation tick across four independent social and search sources per tracked topic.

   * Scores each topic's latest reading against a per-topic Isolation Forest fit on its own 72-hour rolling window, firing a spike when the anomaly score clears a fixed threshold outside a per-topic cooldown.

   * Detected spikes and a separately-ranked trending list both pass through retrieval-grounded narrative generation that reasons explicitly from which sources moved and which didn't.

   * Output: a severity score, source attribution, and a validated narrative string.

6. **Pre-Match Briefing**

   * Generates inside a fixed kickoff window or on demand, using only the two team names and retrieved historical precedent. No live match state dependency exists in this path.

   * Generation is grounded against retrieved documents and instructed not to invent precedent beyond them.

   * This is the one path in the system that calls Groq directly rather than attempting local inference first.

***

## The Stack

* **Backend**: FastAPI, single-worker by design, with `redis.asyncio` as the only persistence layer. No relational database is used anywhere in the system.

* **ML**: Elo rating with Shin de-vigging, a Poisson in-play scoring model, a vectorized NumPy Monte Carlo tournament simulator, an EWMA-smoothed logistic momentum model with a separate offline training pipeline, PPDA-based tactical feature engineering, and a per-topic `IsolationForest` for social/search anomaly detection. Five distinct statistical models.

* **AI**: local-first inference via **Ollama** running `mistral:7b-instruct-q4_K_M` (accelerated locally by an RTX 3060 in this deployment), with **Groq** (`llama-3.3-70b-versatile`) as cloud fallback, and `sentence-transformers/all-MiniLM-L6-v2` for retrieval embeddings.

* **Data**: StatsBomb Open Data (real WC 2018/2022 event streams), `worldcup26.ir` (live fixture feed), The Odds API, Mastodon/Bluesky/Google Trends/Wikipedia, and Zafronix/API-Sports for lineups.

* **Frontend**: Next.js App Router, entirely client-rendered, using the native `EventSource` API for streaming. No WebSocket library. No server-side data fetching.

***

## Architecture at a Glance

```
worldcup26.ir + StatsBomb ──► hybrid_producer (30s poll) ──► MatchState
                                                                   │
                                                                   ▼
                                                             Redis (only store)
                          ┌──────────┬───────────┬──────────────┬───────────┐
                          ▼          ▼           ▼              ▼           ▼
                     momentum    intel      counterfactual   tactical   narrative
                     worker      worker     worker           worker     worker
                     (EWMA)      (LLM+RAG)  (CRN Monte Carlo) (Weaviate)  (IsolationForest)
                          │          │           │              │           │
                          └──────────┴─────┬─────┴──────────────┴───────────┘
                                            ▼
                                    Agent Layer (Ollama → Groq → template)
                                            │
                                            ▼
                              FastAPI SSE + REST ──► Next.js UI
```

One Redis connection, one Uvicorn process, seven `asyncio` tasks. Workers read and write `MatchState` and publish to Redis pub/sub only; they do not call each other directly. `ScoreTracker`, the momentum state dictionary, and the `IsolationForest` rolling windows are process-local. Scaling requires running additional independent stacks against a shared Redis instance, not adding Uvicorn workers.

***

## Data Pipeline

### Data Ingestion Architecture

```
 worldcup26.ir            StatsBomb Open Data           The Odds API
 (score, status, clock)   (WC 2018/2022 events)         (bookmaker odds)
        │                          │                          │
        ▼                          ▼                          ▼
 hybrid_producer.py     rag_indexer.py / tactical_indexer.py   odds_api_client.py
 (30s poll)              momentum_trainer.py / backtest_elo    (on demand, 900s TTL cache)
        │               (offline, on demand per proxy match)          │
        ▼                          ▼                          ▼
 status normalize        event replay, feature extraction     consensus average
 scorer parse            narrative + PPDA document build       Shin de-vig
 Elo-distance proxy match
        │                          │                          │
        ▼                          ▼                          ▼
   MatchState / TeamStats    Weaviate (NarrativeArcs,     prior_builder.py
   MatchEvent                TacticalProfiles)            (W/D/L probabilities)
        │                                                       │
        └───────────────────────┬───────────────────────────────┘
                                 ▼
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
                            ▼
                          Redis
              narrative:spike:*, narrative:trending:latest
```

### External Data Sources

* **Live match state**: `worldcup26.ir`, polled every 30 seconds, providing score, status, kickoff time, and scorer text. Response shape varies (a bare list, or a `games`/`matches`/`data`/`fixtures` key) and is normalized to one shape before anything downstream sees it.

* **Historical match data**: StatsBomb Open Data, competition 43, seasons `[106, 3]` (WC 2022, WC 2018). Full shot/pass/pressure/card event streams, fetched on demand and reused across four different paths: live-fixture proxying, RAG narrative extraction, tactical fingerprint indexing, and momentum-model training/backtesting.

* **Market data**: The Odds API, fetched on demand and cached for 900 seconds. Per-bookmaker decimal odds for head-to-head markets, averaged into one consensus price per fixture before de-vigging.

* **Lineup data**: Zafronix roster API, API-Sports fixture lineups, and a StatsBomb-proxy starting-XI fallback, resolved in that order per request, with a PPDA-estimated formation as the final fallback when none of the three return usable data.

* **Social/search signals**: Mastodon, Bluesky, Google Trends, and Wikipedia, read concurrently every 60 seconds per tracked topic. The only category with its own dedicated normalization and detection pipeline, covered below.

### External Signals

1. Mastodon search, Bluesky `searchPosts`, Google Trends via `pytrends`, and Wikipedia's recent-changes feed are read concurrently for every tracked topic on a 60-second cadence.
2. Any source that is unauthenticated, rate-limited, or unreachable substitutes a deterministic mock generator: a phase-shifted sine wave plus Gaussian noise, seeded per topic, maintaining pipeline operation without live credentials.
3. Each reading is converted into a per-source activity rate (posts/min, mentions/min, a trends index, edits/min) and unified into one four-dimensional feature vector per topic per tick, tagged `"live"` or `"mock"` per source.
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
    home_score: int = 0
    home_stats: TeamStats = Field(default_factory=TeamStats)
    away_id: int = 0
    away_name: str = ""
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

***

## The ML Core

Every prediction traces to an explicit, inspectable formula.

**Team strength to outcome probability.** Elo expectation:

```
E_a = 1 / (1 + 10^((R_b - R_a) / 400))
```

Converted into a full three-outcome distribution with a rating-gap-sensitive draw model:

```
p_draw = clip(0.25·e^(-ΔR/450) + 0.05, 0.10, 0.30)
p_win  = (1 - p_draw)·E_a
p_loss = (1 - p_draw)·(1 - E_a)
```

**Market odds** are preferred over Elo when available and de-vigged with **Shin's (1993) method**, which discounts longshot prices less aggressively than a proportional split:

```
p_i = (√(z² + 4(1-z)·(1/o_i)²) - z) / (2(1-z))     where z = Ω/(Ω+2),  Ω = Σ(1/o_i) - 1
```

**In-play win probability** updates live from score, minute, and red cards using two independent Poisson goal processes:

```
λ_home = max(0.02, 1.3f·(1+0.65σ))      λ_away = max(0.02, 1.3f·(1-0.65σ))
P(g_h, g_a) = Pois(g_h; λ_home) · Pois(g_a; λ_away),     g_h, g_a ∈ [0, 8]
```

`f` is the fraction of the match remaining and `σ` is the pre-match win-probability differential. A red card multiplies the offending team's rate by 0.72 and the opponent's by 1.12.

**Momentum** is EWMA-smoothed and scored by logistic regression, executing in microseconds with fully inspectable inputs:

```
ewma_x = α·x_t + (1-α)·ewma_x_prev,     α = 0.3
P(goal within 5 min) = σ(β₀ + Σβᵢ·featureᵢ) + Σ bump_k·(0.8)^Δt_k
```

Goals and red cards inject a decaying additive bump directly into the output. Coefficients are trained offline by batch gradient descent on real StatsBomb matches. New coefficients ship only when they beat a base-rate log-loss baseline on held-out data.

**Tournament simulation** resolves every simulated match and knockout round for all `N` runs simultaneously through vectorized NumPy operations, not sequential loops. Every reported stage probability carries a 95% confidence interval:

```
margin = 1.96·√(p̂(1-p̂)/N)
```

**Tactical identity** is feature engineering plus cosine retrieval, not a trained classifier:

```
PPDA = opponent completed passes in press zone / defensive actions in press zone
press_intensity = 0.7·min(1, 8/PPDA) + 0.3·min(1, pressures/150)
```

**Narrative anomaly detection** fits a separate `IsolationForest` (`contamination=0.05`, 100 estimators) per tracked topic, refit every 30 ticks on a rolling 72-hour window of four-source activity, scoring every new observation against it. Per-topic baselines correct for activity volume that varies by roughly an order of magnitude between a heavily-followed team and a lightly-followed one:

```
severity = clip((-s - 0.10) / 0.5, 0, 1)
```

`s` is the raw anomaly score. A spike fires when `s` falls below `-0.10`, outside a 300-second per-topic cooldown.

<br />

***

## The Agent Layer

Every agent follows the same execution order: deterministic scoring first, retrieval second, LLM narration last, template fallback always available. The language model narrates a result that has already been computed. It does not decide what happened. All five agents below are fully implemented in the current codebase; none are stubs or placeholders.

### Match Intelligence Agent

* Runs every 30 seconds against the current match state and momentum snapshot.

* Scans the event stream for uncovered goals and cards, and factors in momentum delta and xG-versus-scoreline divergence even when no discrete event has occurred.

* On quieter ticks, falls back to a periodic tactical read of live possession and pressure.

* Whichever condition fires determines the retrieval target. Event or xG narration pulls from the goal/red-card narrative collection filtered by event type. Tactical narration pulls from the pressing-fingerprint collection.

* Retrieved passages are folded into a prompt built from the real scoreline, xG, and momentum figures.

* Generation runs through local-first inference only when the computed relevance score clears a threshold; ticks below threshold return a numeric template built from the same figures.

### Counterfactual Agent

* Fires on a goal, own goal, penalty, or yellow/red/second-yellow card, subject to a 45-second minimum gap per fixture.

* Reverses the triggering event's effect on score and card counts to recover the state immediately beforehand.

* Converts the resulting in-play probability swing into a bounded Elo adjustment.

* Runs two 20,000-run tournament simulations, one under the pre-event adjustment and one under the post-event adjustment, sharing a single deterministic seed; the only source of divergence between the two runs is the adjustment itself.

* Ranks the per-team probability differences across both runs, and hands the largest shifts, together with the in-play win-probability swing, to the same local-first generation path.

* A fallback template reconstructs the identical explanation from the same numbers if generation is unavailable.

### Tactical Agent

* Has no generative step anywhere in its path.

* Converts live possession, shot volume, and pass accuracy into a short style descriptor, a proxy for the pressing statistics the live feed does not expose.

* Embeds that descriptor and retrieves the nearest historical pressing fingerprint by cosine similarity, along with up to two runner-up matches for comparison.

* If nothing clears the similarity threshold it returns nothing, and the interface falls back to displaying live statistics directly.

### Narrative Intelligence Agent

* Consumes spikes and trending rows produced by the anomaly-detection layer, which aggregates four social/search sources per topic every 60 seconds and scores each topic against its own per-topic `IsolationForest`.

* Builds a query from the topic name and its activity summary, and retrieves historical precedent from the narrative collection.

* Constructs a prompt that states explicitly which sources moved and which stayed flat: one source moving alone reads as a viral post, several moving together reads as a real-world event. The prompt instructs the model to reason from that shape rather than restate volume.

* Output passes through a banned-phrase filter; a rule-based fallback performs the identical reasoning without a model whenever generation fails or the result doesn't clear validation.

### Briefing Agent

* Triggers inside a fixed pre-kickoff window, once per match status, or on demand, using only the two team names and retrieved historical precedent. No live match state feeds this path.

* Retrieves precedent from the narrative collection and is instructed not to invent beyond it.

* Falls back to quoting the retrieved passage's first line verbatim, or stating plainly that none was available, if generation fails.

* This is the one agent that calls Groq directly and never attempts local inference, an inconsistency with the other four rather than a deliberate design choice.

Three of the five agents run continuously during live play through local-first inference, keeping goal, card, and anomaly narration off network latency and a metered API by default. Every agent falls back to a template built from the same computed numbers when generation fails, producing plain language instead of an empty response or an invented statistic.

***

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

1. Up to 3-5 retrieved documents are interpolated directly into the prompt text.
2. Every prompt also states the real computed numbers (scoreline, xG, probability shifts, activity levels) alongside the retrieved text.
3. The model is explicitly instructed not to invent facts beyond what the prompt provides.
4. If the model is unavailable, a deterministic template reuses the identical numbers, producing plainer language rather than a fabricated fact.

***

## Real-Time Intelligence Runtime

### Worker Architecture

| Worker         | Cadence | Responsibility                                                  |
| -------------- | ------- | --------------------------------------------------------------- |
| Producer       | 30s     | Polls the live feed, reconstructs `MatchState`                  |
| Momentum       | 30s     | EWMA and logistic inference                                     |
| Intel          | 30s     | Event, xG, and tactical scoring; live narration                 |
| Counterfactual | 30s     | Trigger detection; paired Monte Carlo; bracket-impact narration |
| Tactical       | 120s    | Style descriptor; cosine match against history                  |
| Briefing       | 300s    | Pre-match window gating; tactical preview                       |
| Narrative      | 60s     | Social signal aggregation; anomaly scoring; arc synthesis       |

Only these seven workers exist in the codebase; no additional background processes are launched.

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

1. All seven workers are `asyncio` tasks sharing one event loop and one Redis connection; none run in a separate OS process.
2. Each worker's tick is wrapped in a try/except block, so one fixture's malformed state or one worker's exception never halts processing elsewhere.
3. CPU-bound and blocking work is kept off the event loop via four dedicated thread pools: tournament simulation, counterfactual simulation pairs, embedding, and blocking HTTP calls (Google Trends).

### Streaming Layer

* SSE channels (`match_update`, `momentum_update`, `intel_update`, `counterfactual_update`, `narrative_spike`) perform an initial Redis key read before subscribing, delivering current state immediately to a client connecting mid-match, and carry a periodic heartbeat distinguishing inactivity from a dropped connection.

* Every frontend hook relies on the browser's native `EventSource` reconnect logic for failure recovery.

* The Mastodon/social data path is not part of any SSE channel. Comment samples captured during the 60-second social-signal tick are cached in-process and pushed into a plain Redis list (600s TTL, 19-item cap, deduplicated by permalink or text hash). The frontend reads this by polling a REST endpoint every 15 seconds. No comment-streaming SSE route exists anywhere in the codebase. The sample cache populates only on live (non-mock) provider reads; without social API credentials, the comment UI displays nothing while anomaly detection continues operating on mock signal data.

***

## Deep Dive: The Three Core Intelligence Engines

These three subsystems are treated as first-class reasoning and simulation systems, not simple LLM agents.

### 1. Narrative Intelligence Hub

A 60-second worker aggregates Mastodon, Bluesky, Google Trends, and Wikipedia activity into per-topic feature vectors for anomaly detection, decoupled from `MatchState` and the live-match workers. Multi-source activity shape, not single-value volume, distinguishes a real-world event from a single viral post, feeding both the anomaly feed and the continuously-updated trending view.

Four sources are read concurrently every 60 seconds, per tracked topic:

1. Mastodon authenticated full-text search, disabling itself after repeated authentication failures and falling back to a deterministic mock generator.
2. Bluesky session-token authenticated search, re-authenticating once on a token expiry before falling back to mock.
3. Google Trends via `pytrends`, executed on a blocking I/O thread pool since the library is not async-native.
4. Wikipedia recent-changes edit velocity, used as a proxy for reference and news activity.

Each source's mock fallback substitutes a phase-shifted sine wave, Gaussian noise, and occasional injected spikes, seeded per topic, maintaining detector operation without live credentials.

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

* A trigger event (goal, own goal, penalty, or yellow/red/second-yellow card) is detected in the match event stream, and the state immediately before it is reconstructed by reversing its effect on score and card counts.

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

***

## Performance

What is actually measured, against this codebase directly:

| Measurement                                      | Result            |
| ------------------------------------------------ | ----------------- |
| Vectorized Tournament simulation, 50,000 runs    | 0.70s             |
| Vectorized  Tournament simulation, 10,000 runs   | 0.13s             |
| Counterfactual simulation pair, 20,000 runs each | 0.25s             |
| Live momentum inference                          | \~11.3µs per call |

No formal load-testing, SSE-latency-under-concurrency benchmark, retrieval-quality evaluation, or agent-faithfulness evaluation exists in this codebase. This absence is stated directly rather than replaced with an invented figure.

***

## Performance Optimizations

* **Vectorized simulation.** `tournament_sim.py` resolves every group match and knockout round for all `N` requested runs in one NumPy pass (cumulative probability draws, `np.add.at`, `np.maximum.at`, fancy indexing into a precomputed advancement matrix), instead of looping per simulation. Vectorized execution keeps 50,000 runs at 0.70 seconds instead of scaling linearly with simulated matches.

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

* <br />

***

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
│   ├── ollama_client.py             Local Ollama call with Groq fallback, shared by four agents
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

***

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

| Category      | Stack                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Languages     | Python 3.12, TypeScript                                                                                                                 |
| ML            | NumPy, scikit-learn (`IsolationForest`), `sentence-transformers` (`all-MiniLM-L6-v2`)                                                   |
| Backend       | FastAPI, Uvicorn, `sse-starlette`, Pydantic v2, `httpx`, `redis.asyncio`, `weaviate-client`                                             |
| Frontend      | Next.js 14 App Router, React 18, `next-themes`                                                                                          |
| Data          | Redis 7, Weaviate 1.27                                                                                                                  |
| Infra         | Docker Compose, `python-dotenv`, `python:3.12-slim` base image, `gcc` (build-time, scientific Python wheels)                            |
| Dev / Testing | `pytest`, `ruff`                                                                                                                        |
| External data | StatsBomb Open Data, `worldcup26.ir`, The Odds API, API-Sports, Zafronix, Mastodon, Bluesky, Google Trends (`pytrends`), Wikipedia REST |
| AI            | Ollama (`mistral:7b-instruct-q4_K_M`), Groq (`llama-3.3-70b-versatile`)                                                                 |

# Limitations

* **Tournament outcomes are inherently stochastic.** The simulator reports probability distributions with 95% confidence intervals, not deterministic forecasts. The Elo backtest measured on this codebase loses to a naive base-rate baseline overall, and only beats it after the ratings warm up across roughly half a match sequence.

* **Live state updates are bounded by a 30-second poll interval against a free, keyless feed with no push mechanism.** Events inside that window are not visible until the next poll cycle.

* **Social/search signal reads run against free or unauthenticated-fallback APIs with their own rate limits and timeouts.** Any of the four sources can silently degrade to a mock signal generator; the system tags this internally as live or mock but does not surface it in the UI.

* **Every "live" statistic for a 2026 fixture is a StatsBomb proxy from a real 2018 or 2022 match**, selected by Elo-distance similarity, not the actual game being played. Possession, shots, and xG shown for a live 2026 match describe a historically similar match, not the one on the pitch.

* **A hybrid, keyless-first data approach was chosen over a paid live-stats provider.** API-Sports and comparable commercial feeds gate meaningful request volume and full statistical coverage behind a paid plan. `API_SPORTS_KEY` is optional throughout the codebase and used only as a lineup fallback, never as the primary live data source. StatsBomb Open Data combined with the free `worldcup26.ir` feed avoids that dependency entirely, at the cost of the proxy-statistics limitation above.

* **Counterfactual analysis perturbs team strength rather than replaying an alternate version of the match.** It does not model the effect of a different tactical decision or substitution.

* **Retrieval-grounded generation reduces but does not eliminate hallucination risk.** A banned-phrase filter and prompt-level grounding constrain output; no independent faithfulness or groundedness score is computed for any generated narrative.

* **The single-process architecture is a scalability ceiling, not only a design preference.** Horizontal scale requires running independent stacks against shared Redis and Weaviate instances, not adding workers to one process.

* **The architecture is difficult to host on free-tier platforms.** Seven long-running background workers and in-memory EWMA/IsolationForest state require a persistently running process, not the request-scoped, auto-scaling functions most free serverless tiers provide. A free tier that idles out or restarts a container on inactivity silently resets all momentum and narrative-detection state. Weaviate additionally requires a persistent volume and a non-trivial memory footprint for its vector index, which most free database tiers do not offer. Local LLM inference through Ollama requires meaningful compute (GPU acceleration in this deployment) unavailable on free web hosting, making the Groq cloud fallback the only practical LLM path in a free deployment.
