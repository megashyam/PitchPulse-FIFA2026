"""
RAG corpus generator tests.

Covers own goals, second-yellow and Bad Behaviour reds, the absence of
momentum and boilerplate docs, and situation labels.
"""

from agents.rag_indexer import extract_documents


def _ev(i, period, minute, team, etype, **extra):
    return {"index": i, "period": period, "minute": minute, "team": {"name": team}, "type": {"name": etype}, **extra}


def _shot(i, minute, team, outcome, xg=0.1):
    return _ev(i, 1 if minute < 45 else 2, minute, team, "Shot", shot={"outcome": {"name": outcome}, "statsbomb_xg": xg})


EVENTS = [
    _ev(1, 1, 3, "A", "Pass", **{"pass": {}}),
    _shot(2, 10, "A", "Goal", 0.4),
    _ev(3, 1, 30, "A", "Own Goal For"),  # own goal by B, credited to A
    _ev(4, 1, 30, "B", "Own Goal Against"),
    _ev(5, 2, 55, "B", "Foul Committed", foul_committed={"card": {"name": "Second Yellow"}}),
    _ev(6, 2, 93, "A", "Bad Behaviour", bad_behaviour={"card": {"name": "Red Card"}}),
    _shot(7, 94, "B", "Goal", 0.2),
    _ev(8, 5, 120, "A", "Shot", shot={"outcome": {"name": "Goal"}}),  # shootout: ignored
]


def test_corpus_documents():
    docs = extract_documents(EVENTS, "A", "B", "FIFA World Cup", "2022", "1")
    kinds = [(d["event_type"], d["minute"]) for d in docs]
    assert kinds == [("goal", 10), ("goal", 30), ("red_card", 55), ("red_card", 93), ("goal", 94)]

    og = docs[1]
    assert "own goal" in og["content"] and "to make it 2-0" in og["content"]
    assert og["situation"] == {"acting_team": "A", "state_before": "leading", "minute_band": "0-30"}

    assert docs[2]["situation"]["acting_team"] == "B"
    assert docs[2]["situation"]["state_before"] == "trailing"
    assert "0 minutes of regulation left" in docs[3]["content"]
    assert docs[3]["situation"]["minute_band"] == "90+"
    assert "to make it 2-1" in docs[4]["content"]

    assert all("Pattern:" not in d["content"] for d in docs)
    assert all(d["event_type"] != "momentum_shift" for d in docs)
