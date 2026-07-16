"""
StatsBomb-based narrative arc indexing pipeline.

Extracts tactical and momentum events from historical football matches,
converts structured event data into natural-language narrative documents,
generates semantic embeddings, and stores them in Weaviate for retrieval.

Pipeline stages:
    - Fetch match and event data from StatsBomb Open Data.
    - Detect narrative events including goals, red cards, and momentum shifts.
    - Generate context-rich textual representations of match situations.
    - Encode documents using SentenceTransformer embeddings.
    - Batch index vectors and metadata into Weaviate.

The resulting NarrativeArcs collection provides historical football context
for retrieval-augmented generation systems and live match narrative agents.
"""

import argparse
import asyncio
import logging
from collections import defaultdict, deque
from pathlib import Path

import httpx
from sentence_transformers import SentenceTransformer

from agents.weaviate_client import (
    get_weaviate_client,
    WEAVIATE_HOST,
    WEAVIATE_PORT,
    WEAVIATE_GRPC_PORT,
)
import weaviate
import weaviate.classes as wvc

from ml.statsbomb import sort_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43
SEASON_IDS = [106, 3]  # 2022, 2018
MAX_MATCHES = 500
ALPHA = 0.3
WINDOW = 15


async def fetch_json(client: httpx.AsyncClient, url: str):
    """
    Fetch and decode JSON data from an asynchronous HTTP endpoint.

    Args:
        client: Shared asynchronous HTTP client.
        url: Remote JSON resource URL.

    Returns:
        Parsed JSON response.

    Raises:
        httpx.HTTPStatusError: If the request returns an unsuccessful status.
    """
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _build_goal_doc(
    home: str,
    away: str,
    minute: int,
    home_score: int,
    away_score: int,
    scorer_team: str,
    home_poss: float,
    home_shots: int,
    home_xg: float,
    away_poss: float,
    away_shots: int,
    away_xg: float,
    competition: str,
    season: str,
) -> str:
    """
    Create a narrative document describing a goal event.

    Combines match state, score transition, possession context, shooting
    statistics, and expected goals information into a retrieval-friendly
    textual representation.

    Returns:
        str: Natural-language goal event description for embedding.
    """
    score_before = (
        f"{home_score-1 if scorer_team==home else home_score}"
        f"-{away_score-1 if scorer_team==away else away_score}"
    )

    dom = home if home_poss >= 50 else away
    dom_poss = home_poss if home_poss >= 50 else away_poss

    return (
        f"{competition} {season} · {home} vs {away} · Minute {minute} · Goal\n"
        f"Situation: Score was {score_before}, {scorer_team} scored to make it {home_score}-{away_score}.\n"
        f"Tactical context: {dom} dominated with {dom_poss:.0f}% possession. "
        f"{home} shots: {home_shots}, xG: {home_xg:.2f}. "
        f"{away} shots: {away_shots}, xG: {away_xg:.2f}.\n"
        f"Pattern: Goal came during sustained {dom} pressure phase."
    )


def _build_red_card_doc(
    home: str,
    away: str,
    minute: int,
    home_score: int,
    away_score: int,
    card_team: str,
    home_poss: float,
    competition: str,
    season: str,
) -> str:
    """
    Create a narrative document describing a red card event.

    Captures match state, numerical disadvantage, remaining match context,
    and potential tactical impact.

    Returns:
        str: Natural-language red card event description for embedding.
    """
    remaining = home if card_team == away else away
    return (
        f"{competition} {season} · {home} vs {away} · Minute {minute} · Red Card\n"
        f"Situation: Score {home_score}-{away_score}, {card_team} reduced to 10 men.\n"
        f"Tactical context: {home} possession {home_poss:.0f}%, {away} {100-home_poss:.0f}%. "
        f"{remaining} gained numerical advantage with {90-minute} minutes remaining.\n"
        f"Pattern: Man disadvantage mid-match typically shifts momentum and bracket implications."
    )


def _build_momentum_doc(
    home: str,
    away: str,
    start_min: int,
    end_min: int,
    home_score: int,
    away_score: int,
    shifting_team: str,
    poss_before: float,
    poss_after: float,
    competition: str,
    season: str,
) -> str:
    """
    Create a narrative document describing a tactical momentum shift.

    Represents changes in territorial control using possession movement over
    time and converts statistical changes into a semantic match narrative.

    Returns:
        str: Natural-language momentum shift description for embedding.
    """
    delta = abs(poss_after - poss_before)
    return (
        f"{competition} {season} · {home} vs {away} · Minutes {start_min}-{end_min} · Momentum Shift\n"
        f"Situation: Score {home_score}-{away_score}, {shifting_team} momentum shift.\n"
        f"Tactical context: {shifting_team} possession moved from {poss_before:.0f}% "
        f"to {poss_after:.0f}% — a {delta:.0f}-point swing over {end_min-start_min} minutes.\n"
        f"Pattern: Sustained territorial gain suggesting tactical or fatigue-driven phase change."
    )


def extract_documents(
    events: list,
    home: str,
    away: str,
    competition: str,
    season: str,
    match_id: str,
) -> list:
    """
    Extract retrieval documents from a StatsBomb event stream.

    Processes chronological match events and identifies high-value narrative
    moments such as goals, dismissals, and momentum changes. Each detected
    event is converted into a document with associated match metadata.

    Args:
        events: Raw StatsBomb event list.
        home: Home team name.
        away: Away team name.
        competition: Competition identifier.
        season: Season identifier.
        match_id: Unique match identifier.

    Returns:
        list: Narrative document dictionaries containing:
            - content
            - match_id
            - competition
            - season
            - minute
            - event_type
    """
    evs = sort_events(events)

    h = defaultdict(float)
    a = defaultdict(float)
    home_score = away_score = 0
    documents = []

    h_shot_win = deque(maxlen=WINDOW)
    a_shot_win = deque(maxlen=WINDOW)
    h_poss_win = deque(maxlen=WINDOW)
    a_poss_win = deque(maxlen=WINDOW)

    prev_h_shots = prev_a_shots = 0
    last_momentum_min = 0
    h_ewma_poss = a_ewma_poss = 50.0

    for ev in evs:
        team_name = (ev.get("team") or {}).get("name", "")
        etype = (ev.get("type") or {}).get("name", "")
        minute = ev.get("minute", 0)

        if team_name not in (home, away):
            continue

        s = h if team_name == home else a
        is_home = team_name == home

        s["ev_total"] += 1
        if etype == "Pass":
            s["passes"] += 1
            if ev.get("pass", {}).get("outcome") is None:
                s["passes_acc"] += 1
        elif etype == "Shot":
            s["shots"] += 1
            outcome = (ev.get("shot", {}).get("outcome") or {}).get("name", "")
            xg = ev.get("shot", {}).get("statsbomb_xg") or 0
            s["xg"] += float(xg)
            if outcome in ("Saved", "Saved to Post", "Goal"):
                s["shots_on"] += 1
            if outcome == "Goal":
                if is_home:
                    home_score += 1
                else:
                    away_score += 1

                total_ev = h["ev_total"] + a["ev_total"]
                hp = h["ev_total"] / total_ev * 100 if total_ev else 50.0

                documents.append(
                    {
                        "content": _build_goal_doc(
                            home,
                            away,
                            minute,
                            home_score,
                            away_score,
                            team_name,
                            hp,
                            int(h["shots"]),
                            round(h["xg"], 2),
                            100 - hp,
                            int(a["shots"]),
                            round(a["xg"], 2),
                            competition,
                            season,
                        ),
                        "match_id": match_id,
                        "competition": competition,
                        "season": season,
                        "minute": minute,
                        "event_type": "goal",
                    }
                )

        elif etype == "Foul Committed":
            card = (ev.get("foul_committed", {}).get("card") or {}).get(
                "name", ""
            ) or ""
            if "Red" in card:
                total_ev = h["ev_total"] + a["ev_total"]
                hp = h["ev_total"] / total_ev * 100 if total_ev else 50.0
                documents.append(
                    {
                        "content": _build_red_card_doc(
                            home,
                            away,
                            minute,
                            home_score,
                            away_score,
                            team_name,
                            hp,
                            competition,
                            season,
                        ),
                        "match_id": match_id,
                        "competition": competition,
                        "season": season,
                        "minute": minute,
                        "event_type": "red_card",
                    }
                )

        if minute % 5 == 0 and minute > last_momentum_min:
            last_momentum_min = minute
            total_ev = h["ev_total"] + a["ev_total"]
            hp_now = h["ev_total"] / total_ev * 100 if total_ev else 50.0
            ap_now = 100 - hp_now

            new_h_ewma = ALPHA * hp_now + (1 - ALPHA) * h_ewma_poss
            new_a_ewma = ALPHA * ap_now + (1 - ALPHA) * a_ewma_poss

            delta_h = abs(new_h_ewma - h_ewma_poss)
            delta_a = abs(new_a_ewma - a_ewma_poss)

            if max(delta_h, delta_a) > 8.0 and minute > 10:
                shifting = home if delta_h > delta_a else away
                poss_b = h_ewma_poss if shifting == home else a_ewma_poss
                poss_a = new_h_ewma if shifting == home else new_a_ewma
                documents.append(
                    {
                        "content": _build_momentum_doc(
                            home,
                            away,
                            max(1, minute - 5),
                            minute,
                            home_score,
                            away_score,
                            shifting,
                            poss_b,
                            poss_a,
                            competition,
                            season,
                        ),
                        "match_id": match_id,
                        "competition": competition,
                        "season": season,
                        "minute": minute,
                        "event_type": "momentum_shift",
                    }
                )

            h_ewma_poss = new_h_ewma
            a_ewma_poss = new_a_ewma

    return documents


def _get_count() -> int:
    """
    Retrieve the current NarrativeArcs document count from Weaviate.

    Uses a direct GraphQL request to avoid requiring a full client query path.

    Returns:
        int: Number of indexed narrative documents.
        Returns zero when the query fails.
    """
    import urllib.request, json

    try:
        q = json.dumps({"query": "{Aggregate{NarrativeArcs{meta{count}}}}"})
        req = urllib.request.Request(
            f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}/v1/graphql",
            data=q.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            return data["data"]["Aggregate"]["NarrativeArcs"][0]["meta"]["count"]
    except Exception as exc:
        log.warning(f"count failed: {exc}")
        return 0


async def main(check_only: bool = False) -> None:
    """
    Execute the complete narrative indexing workflow.

    Initializes Weaviate, loads historical match data, extracts narrative
    events, generates embeddings, and batch inserts vectors with metadata into
    the NarrativeArcs collection.

    Args:
        check_only: If True, only reports the current index size without
            rebuilding the collection.

    Workflow:
        1. Validate/create Weaviate collection.
        2. Download StatsBomb matches and events.
        3. Generate narrative documents.
        4. Create semantic embeddings.
        5. Batch insert documents and vectors.
    """

    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT
    )

    try:
        log.info(f"Weaviate ready: {client.is_ready()}")

        if not client.collections.exists("NarrativeArcs"):
            client.collections.create(
                name="NarrativeArcs",
                properties=[
                    wvc.config.Property(
                        name="content", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="match_id", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="competition", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="season", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="minute", data_type=wvc.config.DataType.INT
                    ),
                    wvc.config.Property(
                        name="event_type", data_type=wvc.config.DataType.TEXT
                    ),
                ],
            )
            log.info("Created collection NarrativeArcs")

        if check_only:
            log.info(f"NarrativeArcs document count: {_get_count()}")
            return

        existing = _get_count()
        if existing > 0:
            ans = (
                input(f"Collection already has {existing} documents. Re-index? (y/N): ")
                .strip()
                .lower()
            )
            if ans != "y":
                return

            log.info(f"Dropping {existing} existing documents for a clean rebuild...")
            client.collections.delete("NarrativeArcs")
            client.collections.create(
                name="NarrativeArcs",
                properties=[
                    wvc.config.Property(
                        name="content", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="match_id", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="competition", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="season", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="minute", data_type=wvc.config.DataType.INT
                    ),
                    wvc.config.Property(
                        name="event_type", data_type=wvc.config.DataType.TEXT
                    ),
                ],
            )

        log.info("Loading all-MiniLM-L6-v2 embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embedding model ready")

        all_docs = []

        async with httpx.AsyncClient(
            headers={"User-Agent": "wc2026-rag-indexer/1.0"},
            follow_redirects=True,
        ) as http:
            matches = []
            for sid in SEASON_IDS:
                try:
                    ms = await fetch_json(
                        http, f"{SB_BASE}/matches/{COMPETITION_ID}/{sid}.json"
                    )
                    matches.extend(ms)
                    log.info(f"Season {sid}: {len(ms)} matches")
                except Exception as e:
                    log.warning(f"Season {sid} failed: {e}")

            log.info(f"Processing up to {MAX_MATCHES} of {len(matches)} matches...")
            for i, m in enumerate(matches[:MAX_MATCHES]):
                match_id = str(m["match_id"])
                home = m["home_team"]["home_team_name"]
                away = m["away_team"]["away_team_name"]
                season = str(m.get("season", {}).get("season_name", ""))
                comp = m.get("competition", {}).get("competition_name", "WC")
                try:
                    events = await fetch_json(http, f"{SB_BASE}/events/{match_id}.json")
                    docs = extract_documents(events, home, away, comp, season, match_id)
                    all_docs.extend(docs)
                    log.info(f"  [{i+1:2d}] {home} vs {away}: {len(docs)} docs")
                except Exception as e:
                    log.warning(f"  [{i+1:2d}] {home} vs {away}: failed — {e}")

        log.info(f"\nTotal documents to index: {len(all_docs)}")
        counts = defaultdict(int)
        for d in all_docs:
            counts[d["event_type"]] += 1
        for k, v in counts.items():
            log.info(f"  {k}: {v}")

        log.info("\nEmbedding and inserting into Weaviate...")
        col = client.collections.get("NarrativeArcs")
        inserted = 0
        batch_size = 50

        with col.batch.dynamic() as batch:
            for i in range(0, len(all_docs), batch_size):
                chunk = all_docs[i : i + batch_size]
                texts = [d["content"] for d in chunk]
                vectors = model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                for doc, vec in zip(chunk, vectors):
                    batch.add_object(
                        properties={
                            "content": doc["content"],
                            "match_id": doc["match_id"],
                            "competition": doc["competition"],
                            "season": doc["season"],
                            "minute": doc["minute"],
                            "event_type": doc["event_type"],
                        },
                        vector=vec.tolist(),
                    )
                    inserted += 1
                log.info(f"  Queued {inserted}/{len(all_docs)}")

        final_count = _get_count()
        log.info(f"\nDone — {final_count} documents confirmed in Weaviate")

    finally:
        client.close()
        log.info("Connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Just report document count, don't re-index",
    )
    args = parser.parse_args()
    asyncio.run(main(check_only=args.check))
