from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class StageProbability(BaseModel):
    p: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0


class TeamPrediction(BaseModel):
    name: str
    group: str
    elo: float
    fifa_rank: int

    # ── Group stage finishing position ────────────────────────────────────
    group_exit: StageProbability = Field(default_factory=StageProbability)
    group_first: StageProbability = Field(default_factory=StageProbability)
    group_second: StageProbability = Field(default_factory=StageProbability)
    group_third: StageProbability = Field(default_factory=StageProbability)
    group_fourth: StageProbability = Field(default_factory=StageProbability)

    # ── Knockout stages ───────────────────────────────────────────────────
    r32: StageProbability = Field(default_factory=StageProbability)
    r16: StageProbability = Field(default_factory=StageProbability)
    qf: StageProbability = Field(default_factory=StageProbability)
    sf: StageProbability = Field(default_factory=StageProbability)
    final: StageProbability = Field(default_factory=StageProbability)
    champion: StageProbability = Field(default_factory=StageProbability)

    @classmethod
    def from_result(cls, r) -> "TeamPrediction":
        def sp(stage: str) -> StageProbability:
            p = r.probs.get(stage, 0.0)
            lo, hi = r.ci_95.get(stage, (0.0, 0.0))
            return StageProbability(p=p, ci_lo=lo, ci_hi=hi)

        return cls(
            name=r.name,
            group=r.group,
            elo=r.elo,
            fifa_rank=r.fifa_rank,
            group_exit=sp("group_exit"),
            group_first=sp("group_first"),
            group_second=sp("group_second"),
            group_third=sp("group_third"),
            group_fourth=sp("group_fourth"),
            r32=sp("r32"),
            r16=sp("r16"),
            qf=sp("qf"),
            sf=sp("sf"),
            final=sp("final"),
            champion=sp("champion"),
        )


class TournamentPrediction(BaseModel):
    sim_id: str
    n_sims: int
    elapsed_s: float
    run_at: datetime
    teams: List[TeamPrediction]
    status: str = "complete"


class SimStatus(BaseModel):
    status: str
    sim_id: Optional[str] = None
    started_at: Optional[datetime] = None
    error: Optional[str] = None


class SimTriggerResponse(BaseModel):
    accepted: bool
    message: str
    sim_id: str
