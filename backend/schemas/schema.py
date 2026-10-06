"""
Pydantic models for match state and the ESPN feed records behind it.
Raw payloads are validated in schemas/espn_schema.py and converted
here with `from_espn`.

    MatchState.stats_source       "espn" | "unavailable"
    MatchState.elapsed_estimated  True when the minute was synthesised from
                                  a status string (UI renders "≈30'")
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from schemas.espn_schema import (
    HOSTS,
    KEY_EVENT_TYPES,
    LINE_ORDER,
    NOT_PLAYED,
    POS_LINE,
    POS_X,
    STAGES,
    STAT_KEYS,
    EspnEvent,
    EspnKeyEvent,
    EspnRoster,
    EspnRosterPlayer,
    EspnStandingsEntry,
    EspnStatus,
    EspnTeamBoxscore,
)
from ml.shot_xg import parse_clock
from ml.team_names import canonical

# ESPN team id → (side 1|2, canonical team name)
TeamIds = dict[str, tuple[int, str]]


def _float(v: str) -> float:
    try:
        return float(v or 0)
    except ValueError:
        return 0.0


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
    tackles: int = 0
    interceptions: int = 0
    offsides: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    goalkeeper_saves: int = 0
    expected_goals: float = 0.0

    @classmethod
    def from_espn(cls, raw: EspnTeamBoxscore, xg: float) -> "TeamStats":
        """`xg` comes from our shot model; ESPN has no xG."""
        v = {STAT_KEYS[s.name]: _float(s.display_value) for s in raw.statistics if s.name in STAT_KEYS}
        total, on = int(v.get("shots_total", 0)), int(v.get("shots_on_goal", 0))
        passes, accurate = int(v.get("passes_total", 0)), int(v.get("passes_accurate", 0))
        return cls(
            possession=v.get("possession", 0.0),
            shots_total=total,
            shots_on_goal=on,
            shots_off_goal=max(0, total - on),
            passes_total=passes,
            passes_accurate=accurate,
            pass_accuracy=round(accurate / passes * 100, 1) if passes else 0.0,
            corner_kicks=int(v.get("corner_kicks", 0)),
            fouls=int(v.get("fouls", 0)),
            tackles=int(v.get("tackles", 0)),
            interceptions=int(v.get("interceptions", 0)),
            offsides=int(v.get("offsides", 0)),
            yellow_cards=int(v.get("yellow_cards", 0)),
            red_cards=int(v.get("red_cards", 0)),
            goalkeeper_saves=int(v.get("goalkeeper_saves", 0)),
            expected_goals=round(xg, 2),
        )

    def ppda_full(self, opp: "TeamStats") -> float:
        """Whole-pitch PPDA: opponent completed passes per tackle,
        interception or foul (live feeds have no event locations)."""
        return opp.passes_accurate / max(1, self.tackles + self.interceptions + self.fouls)


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

    @classmethod
    def from_espn(cls, raw: EspnKeyEvent, ids: TeamIds) -> Optional["MatchEvent"]:
        """None for untracked event types, shootout kicks and unknown teams."""
        etype = KEY_EVENT_TYPES.get(raw.type.text)
        team = ids.get(raw.team.id)
        if etype is None or raw.shootout or team is None:
            return None
        side, name = team
        if etype == "own_goal":
            # ESPN credits the benefiting side; downstream expects the
            # conceding (own-goal scorer's) side.
            side = 3 - side
            name = next(n for s, n in ids.values() if s == side)
        minute, extra = parse_clock(raw.clock.display_value)
        players = [p.athlete.display_name for p in raw.participants]
        detail = None
        if etype == "substitution" and len(players) > 1:
            detail = f"replaces {players[1] or ''}"
        return cls(
            elapsed=minute,
            extra=extra,
            team_id=side,
            team_name=name,
            player_name=players[0] if players else None,
            type=etype,
            detail=detail,
            source="espn",
        )


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


class Fixture(BaseModel):
    """One scoreboard fixture; the shape of tournament.json "fixtures"."""

    fixture_id: int
    date: Optional[str] = None  # ISO kickoff, UTC
    stage: str = "group"  # group | r32 | r16 | qf | sf | 3rd | final
    group: Optional[str] = None
    round: str = ""
    status: str = "NS"
    elapsed: Optional[int] = None
    elapsed_extra: Optional[int] = None
    home_name: str
    away_name: str
    home_espn_id: str = ""
    away_espn_id: str = ""
    home_logo: str = ""
    away_logo: str = ""
    home_score: Optional[int] = None  # None until kickoff
    away_score: Optional[int] = None
    home_pens: Optional[int] = None
    away_pens: Optional[int] = None
    winner: Optional[str] = None
    venue: str = ""
    # Host nations play at home in their own country; everything else is neutral.
    host_side: Optional[str] = None  # home | away
    neutral: bool = True

    @property
    def kickoff(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.date.replace("Z", "+00:00")) if self.date else None
        except ValueError:
            return None

    def espn_ids(self) -> TeamIds:
        return {
            self.home_espn_id: (1, self.home_name),
            self.away_espn_id: (2, self.away_name),
        }

    @classmethod
    def from_espn(cls, raw: EspnEvent) -> "Fixture":
        comp = raw.competitions[0]
        home, away = comp.side("home"), comp.side("away")
        home_name = canonical(home.team.display_name)
        away_name = canonical(away.team.display_name)

        stage, label = STAGES.get(raw.season.slug, ("group", "Group Stage"))
        group = comp.group.name.replace("Group", "").strip() or None
        status, minute, extra = (raw.status or comp.status or EspnStatus()).decode()
        started = status not in NOT_PLAYED

        country = comp.venue.address.country
        host_side = (
            "home" if HOSTS.get(home_name) == country
            else "away" if HOSTS.get(away_name) == country
            else None
        )
        winner = None
        if raw.status and raw.status.type.completed:
            winner = home_name if home.winner else away_name if away.winner else None

        return cls(
            fixture_id=raw.id,
            date=raw.date,
            stage=stage,
            group=group,
            round=f"Group Stage - Group {group}" if stage == "group" and group else label,
            status=status,
            elapsed=minute,
            elapsed_extra=extra,
            home_name=home_name,
            away_name=away_name,
            home_espn_id=home.team.id,
            away_espn_id=away.team.id,
            home_logo=home.team.logo,
            away_logo=away.team.logo,
            home_score=home.score if started else None,
            away_score=away.score if started else None,
            home_pens=home.shootout_score,
            away_pens=away.shootout_score,
            winner=winner,
            venue=comp.venue.full_name,
            host_side=host_side,
            neutral=host_side is None,
        )


class StandingRow(BaseModel):
    team: str
    espn_id: str = ""
    rank: int = 0
    points: int = 0
    gd: int = 0
    gf: int = 0
    played: int = 0

    @classmethod
    def from_espn(cls, raw: EspnStandingsEntry) -> "StandingRow":
        return cls(
            team=canonical(raw.team.display_name),
            espn_id=raw.team.id,
            rank=raw.stat("rank"),
            points=raw.stat("points"),
            gd=raw.stat("pointDifferential"),
            gf=raw.stat("pointsFor"),
            played=raw.stat("gamesPlayed"),
        )


class LineupPlayer(BaseModel):
    number: int = 0
    name: str = ""
    position: str = ""
    line: str = "M"  # G | D | M | F
    grid: str = ""
    captain: bool = False
    subbed_in: bool = False
    subbed_out: bool = False
    photo: Optional[str] = None

    @classmethod
    def from_espn(cls, raw: EspnRosterPlayer) -> "LineupPlayer":
        pos = raw.position.abbreviation
        return cls(
            number=raw.jersey or 0,
            name=raw.athlete.display_name or "",
            position=pos,
            line=POS_LINE.get(pos, "M"),
            grid=raw.formation_place,
            captain=raw.captain,
            subbed_in=raw.subbed_in,
            subbed_out=raw.subbed_out,
        )


class Lineup(BaseModel):
    team: str
    formation: str = ""
    startingXI: list[LineupPlayer] = Field(default_factory=list)
    substitutes: list[LineupPlayer] = Field(default_factory=list)
    coach: Optional[str] = None

    @classmethod
    def from_espn(cls, raw: EspnRoster, team: str) -> "Lineup":
        starters = sorted(
            (LineupPlayer.from_espn(p) for p in raw.roster if p.starter),
            key=lambda p: (LINE_ORDER.get(p.line, 2), POS_X.get(p.position, 2)),
        )
        return cls(
            team=team,
            formation=raw.formation,
            startingXI=starters,
            substitutes=[LineupPlayer.from_espn(p) for p in raw.roster if not p.starter],
        )


class MatchDetail(BaseModel):
    """Stats, events, lineups and shots from one ESPN summary; the shape of
    data/wc2026/matches/{fixture}.json."""

    home_stats: TeamStats = Field(default_factory=TeamStats)
    away_stats: TeamStats = Field(default_factory=TeamStats)
    events: list[MatchEvent] = Field(default_factory=list)
    lineups: Optional[dict[str, Lineup]] = None  # {"home": ..., "away": ...}
    shots: list[dict] = Field(default_factory=list)  # ml.shot_xg.Shot fields + "side"
