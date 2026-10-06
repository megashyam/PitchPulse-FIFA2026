"""
NarrativeArcs corpus builder and Weaviate indexer.

Builds goal and red-card situation docs from every StatsBomb WC 2018/2022
match. Own goals and second-yellow / "Bad Behaviour" reds are counted
(ml.statsbomb.card_from_event); "minutes remaining" is clamped at zero.

Each doc carries `situation` metadata (acting team's game state before the
event, minute band). It isn't indexed; the retrieval eval uses it as the
relevance label.

    PYTHONPATH=. python agents/rag_indexer.py            # rebuild (prompts if non-empty)
    PYTHONPATH=. python agents/rag_indexer.py --yes      # rebuild without prompting
    PYTHONPATH=. python agents/rag_indexer.py --check    # count only
"""

import argparse
import logging
from collections import Counter

import httpx

from agents.weaviate_client import NARRATIVE_ARCS, get_weaviate_client
from ml.embedding_model import get_embed_model
from ml.statsbomb import COMPETITION_ID, SEASON_IDS, card_from_event, load, sort_events

log = logging.getLogger(__name__)

def minute_band(minute: int) -> str:
    if minute <= 30:
        return "0-30"
    if minute <= 60:
        return "31-60"
    if minute <= 90:
        return "61-90"
    return "90+"


def game_state(own: int, other: int) -> str:
    return "leading" if own > other else "trailing" if own < other else "level"


def _build_goal_doc(
    home: str,
    away: str,
    minute: int,
    home_score: int,
    away_score: int,
    scorer_team: str,
    own_goal: bool,
    home_poss: float,
    home_shots: int,
    home_xg: float,
    away_shots: int,
    away_xg: float,
    competition: str,
    season: str,
) -> str:
    before_h = home_score - (scorer_team == home)
    before_a = away_score - (scorer_team == away)
    dom = home if home_poss >= 50 else away
    dom_poss = max(home_poss, 100 - home_poss)
    how = f"{scorer_team} benefited from an own goal" if own_goal else f"{scorer_team} scored"
    return (
        f"{competition} {season} · {home} vs {away} · Minute {minute} · Goal\n"
        f"Situation: Score was {before_h}-{before_a}, {how} to make it {home_score}-{away_score}.\n"
        f"Match so far: {dom} had {dom_poss:.0f}% of the events. "
        f"{home} shots: {home_shots}, xG: {home_xg:.2f}. "
        f"{away} shots: {away_shots}, xG: {away_xg:.2f}."
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
    other = home if card_team == away else away
    return (
        f"{competition} {season} · {home} vs {away} · Minute {minute} · Red Card\n"
        f"Situation: Score {home_score}-{away_score}, {card_team} reduced to 10 men.\n"
        f"Match so far: {home} {home_poss:.0f}% of the events, {away} {100 - home_poss:.0f}%. "
        f"{other} had the extra player with {max(0, 90 - minute)} minutes of regulation left."
    )


def extract_documents(
    events: list,
    home: str,
    away: str,
    competition: str,
    season: str,
    match_id: str,
) -> list:
    """Docs from one StatsBomb event stream.

    Each doc: {content, match_id, competition, season, minute, event_type,
    situation}.
    """
    counts = {home: Counter(), away: Counter()}
    score = {home: 0, away: 0}
    docs = []

    def home_poss() -> float:
        total = counts[home]["ev"] + counts[away]["ev"]
        return counts[home]["ev"] / total * 100 if total else 50.0

    def doc(content: str, minute: int, event_type: str, acting: str, before: tuple) -> dict:
        own, other = before if acting == home else before[::-1]
        return {
            "content": content,
            "match_id": match_id,
            "competition": competition,
            "season": season,
            "minute": minute,
            "event_type": event_type,
            "situation": {
                "acting_team": acting,
                "state_before": game_state(own, other),
                "minute_band": minute_band(minute),
            },
        }

    for ev in sort_events(events):  # drops the penalty shootout
        team = (ev.get("team") or {}).get("name", "")
        if team not in counts:
            continue
        etype = (ev.get("type") or {}).get("name", "")
        minute = ev.get("minute", 0)
        c = counts[team]
        c["ev"] += 1

        goal, own_goal = False, False
        if etype == "Shot":
            shot = ev.get("shot") or {}
            c["shots"] += 1
            c["xg"] += float(shot.get("statsbomb_xg") or 0)
            goal = (shot.get("outcome") or {}).get("name") == "Goal"
        elif etype == "Own Goal For":  # recorded on the benefiting team
            goal = own_goal = True

        if goal:
            before = (score[home], score[away])
            score[team] += 1
            content = _build_goal_doc(
                home, away, minute, score[home], score[away], team, own_goal,
                home_poss(), counts[home]["shots"], counts[home]["xg"],
                counts[away]["shots"], counts[away]["xg"], competition, season,
            )
            docs.append(doc(content, minute, "goal", team, before))
        elif card_from_event(ev) == "red":
            before = (score[home], score[away])
            content = _build_red_card_doc(
                home, away, minute, score[home], score[away], team, home_poss(),
                competition, season,
            )
            docs.append(doc(content, minute, "red_card", team, before))

    return docs


def build_corpus() -> list[dict]:
    """Every doc from every WC 2018/2022 match (cached StatsBomb downloads)."""
    docs: list[dict] = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for sid in SEASON_IDS:
            for m in load(f"matches/{COMPETITION_ID}/{sid}.json", client):
                events = load(f"events/{m['match_id']}.json", client)
                docs.extend(
                    extract_documents(
                        events,
                        m["home_team"]["home_team_name"],
                        m["away_team"]["away_team_name"],
                        m.get("competition", {}).get("competition_name", "WC"),
                        str(m.get("season", {}).get("season_name", "")),
                        str(m["match_id"]),
                    )
                )
    return docs


def main(check_only: bool = False, assume_yes: bool = False) -> None:
    wv = get_weaviate_client()
    try:
        if not wv.ready:
            log.error("Weaviate not ready")
            return
        existing = wv.get_count(NARRATIVE_ARCS)
        if check_only:
            log.info(f"{NARRATIVE_ARCS} document count: {existing}")
            return
        if existing > 0 and not assume_yes:
            ans = input(f"Collection has {existing} documents. Re-index? (y/N): ")
            if ans.strip().lower() != "y":
                return
        # Re-index means replace, not append.
        wv.recreate(NARRATIVE_ARCS)

        docs = build_corpus()
        log.info(f"Documents: {len(docs)} {dict(Counter(d['event_type'] for d in docs))}")
        vectors = get_embed_model().encode(
            [d["content"] for d in docs], normalize_embeddings=True, show_progress_bar=False
        )
        keys = ("content", "match_id", "competition", "season", "minute", "event_type")
        wv.insert_many(NARRATIVE_ARCS, [{k: d[k] for k in keys} for d in docs], vectors)
        log.info(f"Done — {wv.get_count(NARRATIVE_ARCS)} documents in Weaviate")
    finally:
        wv.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report document count only")
    parser.add_argument("--yes", action="store_true", help="re-index without prompting")
    args = parser.parse_args()
    main(check_only=args.check, assume_yes=args.yes)
