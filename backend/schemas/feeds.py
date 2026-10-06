"""Shapes of the JSON the workers write to Redis (TypedDicts: no runtime
effect, the stored JSON is unchanged), plus the CF Change row."""

from typing import List, NamedTuple, NotRequired, Optional, TypedDict


class MomentumSide(TypedDict):
    momentum_score: float  # share of the two sides' goal probability
    goal_prob_5min: float
    xg_15min: float
    shots_15min: int
    xg_total: float


class Momentum(TypedDict):
    """match:{id}:momentum (ml.momentum_model)."""
    fixture_id: int
    home_name: str
    away_name: str
    elapsed: int
    window_min: int
    stats_source: str
    home: MomentumSide
    away: MomentumSide
    updated_at: str


class WinProb(TypedDict):
    home: float
    draw: float
    away: float


class WinProbShift(TypedDict):
    team: str  # side that benefited from the event
    p_before: float
    p_after: float


class IntelEntry(TypedDict):
    """One match:{id}:intel:feed entry."""
    fixture_id: int
    minute: int
    narration_type: str  # xg_divergence | tactical | event_reaction
    narrative: str
    score: float
    rag_docs_used: int
    rag_collection: str
    via: str
    updated_at: str
    event_sig: NotRequired[str]  # event and FT-summary entries only
    win_prob_shift: NotRequired[Optional[WinProbShift]]


class Change(NamedTuple):
    """One team's champion-probability move between two CF brackets."""
    team: str
    before: float
    after: float
    delta: float
    se: Optional[float]  # paired CRN SE; None when hits weren't kept


class TopChange(TypedDict):
    team: str
    before: float
    after: float
    delta: float
    delta_se: Optional[float]


class CfEntry(TypedDict):
    """One match:{id}:counterfactual:feed entry."""
    fixture_id: int
    minute: int
    extra: Optional[int]  # stoppage-time sub-minute; part of the coverage sig
    event_type: str
    event_team: str
    event_team_id: int
    path_shift_pct: float
    top_changes: List[TopChange]
    narrative: str
    via: str
    conditioned: bool  # fixture is in the simulated tournament
    match_win_prob_before: float
    match_win_prob_after: float
    n_sims: int
    elapsed_s: float
    updated_at: str
