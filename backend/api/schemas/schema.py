"""
Pydantic models for match state.

    MatchState.stats_source       "espn" | "unavailable"
    MatchState.elapsed_estimated  True when the minute was synthesised from
                                  a status string (UI renders "≈30'")
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


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


class MatchEvent(BaseModel):
    elapsed: int
    extra: Optional[int] = None
    team_id: int
    team_name: str
    player_name: Optional[str] = None
    type: str
    detail: Optional[str] = None
    # Provenance: "espn" (real feed) | "synthesised" (from a score delta).
    source: str = "espn"


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
    elapsed_extra: Optional[int] = None  # stoppage minutes ("90'+4'" → 4)
    elapsed_estimated: bool = False  # minute synthesised from status string
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

    # Provenance of home_stats/away_stats: "espn" (real match stats; xG is
    # our shot model over real shots, see ml/shot_xg.py) | "unavailable".
    stats_source: str = "unavailable"
    home_pens: Optional[int] = None
    away_pens: Optional[int] = None

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
