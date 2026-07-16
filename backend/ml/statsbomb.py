"""
Shared StatsBomb data utilities used across the football intelligence pipeline.

This module provides the canonical source for:

    StatsBomb Dataset
            |
            |
    ----------------------
    |         |          |
 Producer  Trainer  Indexers
    |         |          |
    ----------------------
            |
      Consistent Features


Core responsibilities:
    - Maintain shared StatsBomb dataset configuration.
    - Normalize event parsing across training and inference pipelines.
    - Provide shot-quality and goalkeeper event classifiers.
    - Handle card extraction from different StatsBomb event schemas.
    - Ensure identical preprocessing logic between ML training and serving.


Design goal:

Centralizing these utilities prevents feature drift between:

    Training Pipeline
            |
            |
     Momentum Model
            |
            |
    Runtime Inference


The module acts as the data contract layer between raw StatsBomb
events and downstream ML systems.
"""

from __future__ import annotations

from typing import Optional

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43  # FIFA World Cup
SEASON_IDS = [106, 3]  # WC 2022, WC 2018


ON_TARGET_SHOT_OUTCOMES = {
    "goal",
    "saved",
    "saved to post",
    "saved twice",
}

GK_SAVE_OUTCOMES = {
    "touched out",
    "success",
    "in play safe",
    "collected twice",
    "success in play",
    "success out",
}


def shot_is_on_target(outcome_name: str) -> bool:
    """
    Determine whether a shot outcome represents an on-target attempt.

    StatsBomb contains multiple naming variations for goalkeeper saves and
    post saves, so outcomes are normalized before lookup.

    Args:
        outcome_name:
            Raw StatsBomb shot outcome label.

    Returns:
        True if the shot counts as on target, otherwise False.
    """
    return (outcome_name or "").strip().lower() in ON_TARGET_SHOT_OUTCOMES


def gk_is_save(outcome_name: str) -> bool:
    """
    Determine whether an event represents a goalkeeper save action.

    Handles multiple StatsBomb goalkeeper outcome labels by normalizing
    event names before classification.

    Args:
        outcome_name:
            Raw StatsBomb goalkeeper outcome label.

    Returns:
        True if the outcome represents a goalkeeper save, otherwise False.
    """
    return (outcome_name or "").strip().lower() in GK_SAVE_OUTCOMES


def card_from_event(ev: dict) -> Optional[str]:
    """
    Extract disciplinary card information from a StatsBomb event.

    StatsBomb stores cards under different event containers depending on
    the event type. This function normalizes those variations into a simple
    downstream representation.

    Handles:
        - Foul Committed events
        - Bad Behaviour events
        - Second yellow → red conversion

    Args:
        ev:
            Raw StatsBomb event dictionary.

    Returns:
        "red":
            Player/team received a sending off.

        "yellow":
            Yellow card event.

        None:
            Event does not contain a recognized card.
    """
    etype = (ev.get("type") or {}).get("name", "")
    if etype == "Foul Committed":
        card = (ev.get("foul_committed", {}).get("card") or {}).get("name")
    elif etype == "Bad Behaviour":
        card = (ev.get("bad_behaviour", {}).get("card") or {}).get("name")
    else:
        return None
    if not card:
        return None
    if "Red" in card or "Second Yellow" in card:
        return "red"
    if "Yellow" in card:
        return "yellow"
    return None


def sort_events(events: list[dict]) -> list[dict]:
    """
    Sort StatsBomb events into chronological match order.

    Ensures every downstream consumer processes events using the same
    ordering, preventing subtle differences between training and inference
    feature generation.

    Sorting priority:
        1. Match period
        2. Match minute
        3. Original event index

    Args:
        events:
            Unordered list of raw StatsBomb event dictionaries.

    Returns:
        Chronologically ordered list of StatsBomb events.
    """
    return sorted(
        (e for e in events if e.get("period", 1) < 5),
        key=lambda e: (e.get("period", 1), e.get("minute", 0), e.get("index", 0)),
    )
