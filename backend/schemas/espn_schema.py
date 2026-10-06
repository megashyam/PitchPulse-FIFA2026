"""
Pydantic models for the raw ESPN soccer API payloads, plus ESPN's code
tables. Only the fields we read are declared; the rest is ignored.

    EspnEvent      scoreboard  events[]
    EspnStandings  standings   (children = groups)
    EspnSummary    summary     boxscore, keyEvents, rosters, commentary

Domain models are built from these in schemas/schema.py (`from_espn`).
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from ml.shot_xg import parse_clock


HOSTS = {"United States": "USA", "Mexico": "Mexico", "Canada": "Canada"}

# Season slug → (stage code, round label)
STAGES = {
    "group-stage": ("group", "Group Stage"),
    "round-of-32": ("r32", "Round of 32"),
    "round-of-16": ("r16", "Round of 16"),
    "quarterfinals": ("qf", "Quarter-final"),
    "semifinals": ("sf", "Semi-final"),
    "3rd-place-match": ("3rd", "3rd Place"),
    "final": ("final", "Final"),
}

STATUS = {
    "STATUS_SCHEDULED": "NS",
    "STATUS_FIRST_HALF": "1H",
    "STATUS_HALFTIME": "HT",
    "STATUS_SECOND_HALF": "2H",
    "STATUS_END_OF_REGULATION": "ET",
    "STATUS_OVERTIME": "ET",
    "STATUS_FIRST_HALF_EXTRA_TIME": "ET",
    "STATUS_HALFTIME_ET": "ET",
    "STATUS_SECOND_HALF_EXTRA_TIME": "ET",
    "STATUS_END_OF_EXTRATIME": "P",
    "STATUS_SHOOTOUT": "P",
    "STATUS_FULL_TIME": "FT",
    "STATUS_FINAL": "FT",
    "STATUS_FINAL_AET": "AET",
    "STATUS_FINAL_PEN": "PEN",
    "STATUS_POSTPONED": "PST",
    "STATUS_CANCELED": "CANC",
    "STATUS_ABANDONED": "ABD",
    "STATUS_SUSPENDED": "SUSP",
    "STATUS_DELAYED": "NS",
}
PERIOD_STATUS = {1: "1H", 2: "2H", 3: "ET", 4: "ET", 5: "P"}
NOT_PLAYED = frozenset({"NS", "PST", "CANC"})

STATUS_LONG = {
    "NS": "Not Started",
    "1H": "First Half",
    "HT": "Halftime",
    "2H": "Second Half",
    "ET": "Extra Time",
    "P": "Penalties",
    "FT": "Match Finished",
    "AET": "After Extra Time",
    "PEN": "Penalties Finished",
    "PST": "Postponed",
    "CANC": "Cancelled",
    "ABD": "Abandoned",
    "SUSP": "Suspended",
}

# keyEvents type.text → MatchEvent.type
KEY_EVENT_TYPES = {
    "Goal": "goal",
    "Goal - Header": "goal",
    "Goal - Free-kick": "goal",
    "Goal - Volley": "goal",
    "Penalty - Scored": "penalty_goal",
    "Own Goal": "own_goal",
    "Yellow Card": "yellow",
    "Red Card": "red",
    "VAR - (Red) Card Upgrade": "red",
    "Substitution": "substitution",
    "Penalty - Missed": "penalty_missed",
    "Penalty - Saved": "penalty_missed",
}

# boxscore statistics name → TeamStats field
STAT_KEYS = {
    "possessionPct": "possession",
    "totalShots": "shots_total",
    "shotsOnTarget": "shots_on_goal",
    "totalPasses": "passes_total",
    "accuratePasses": "passes_accurate",
    "wonCorners": "corner_kicks",
    "foulsCommitted": "fouls",
    "totalTackles": "tackles",  # attempted, like StatsBomb Duel/Tackle
    "interceptions": "interceptions",
    "offsides": "offsides",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "saves": "goalkeeper_saves",
}

# Roster position abbreviation → line (G/D/M/F), and left→right order within it.
LINE_ORDER = {"G": 0, "D": 1, "M": 2, "F": 3}
POS_LINE = {
    "G": "G",
    "SW": "D", "CD": "D", "CD-L": "D", "CD-R": "D", "LB": "D", "RB": "D", "LWB": "D", "RWB": "D",
    "DM": "M", "CM": "M", "CM-L": "M", "CM-R": "M", "LM": "M", "RM": "M", "M": "M",
    "AM": "M", "AM-L": "M", "AM-R": "M",
    "F": "F", "CF": "F", "CF-L": "F", "CF-R": "F", "LF": "F", "RF": "F", "RCF": "F", "LCF": "F",
}
POS_X = {
    "LB": 0, "LWB": 0, "LM": 0, "LF": 0, "AM-L": 0,
    "CD-L": 1, "CM-L": 1, "CF-L": 1,
    "CD": 2, "SW": 2, "DM": 2, "CM": 2, "AM": 2, "F": 2, "CF": 2, "M": 2, "G": 2,
    "CD-R": 3, "CM-R": 3, "CF-R": 3, "RCF": 3,
    "RB": 4, "RWB": 4, "RM": 4, "RF": 4, "AM-R": 4,
}


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ESPN sends scores/jerseys as strings ("3"); unparseable → None.
OptInt = Annotated[Optional[int], BeforeValidator(_int_or_none)]


class EspnModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,  # displayName ↔ display_name
        populate_by_name=True,
        coerce_numbers_to_str=True,  # ids arrive as int or str
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        # ESPN sends null for absent objects; let field defaults apply.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class EspnTeam(EspnModel):
    id: str = ""
    display_name: str = ""
    logo: str = ""


class EspnAthlete(EspnModel):
    display_name: Optional[str] = None


class EspnStatusType(EspnModel):
    name: str = ""
    state: str = ""  # pre | in | post
    completed: bool = False


class EspnStatus(EspnModel):
    type: EspnStatusType = EspnStatusType()
    period: OptInt = None
    clock: Optional[float] = None  # seconds
    display_clock: str = ""  # "90'+4'"

    def decode(self) -> tuple[str, Optional[int], Optional[int]]:
        """→ (status code, minute, stoppage) from the real clock
        ("90'+4'" → 90, 4); minute is None when not started."""
        code = STATUS.get(self.type.name)
        if code is None:
            if self.type.state == "pre":
                code = "NS"
            elif self.type.state == "post":
                code = "FT"
            else:
                code = PERIOD_STATUS.get(self.period or 1, "1H")
        if code in NOT_PLAYED:
            return code, None, None
        minute, extra = parse_clock(self.display_clock)
        if not minute and self.clock:
            minute = int(self.clock // 60) + 1
        if code == "HT":
            minute = 45
        return code, minute or None, extra


class EspnCompetitor(EspnModel):
    home_away: str = ""
    team: EspnTeam
    score: OptInt = None
    shootout_score: OptInt = None
    winner: bool = False


class EspnAddress(EspnModel):
    country: str = ""


class EspnVenue(EspnModel):
    full_name: str = ""
    address: EspnAddress = EspnAddress()


class EspnGroup(EspnModel):
    name: str = ""  # "Group A"


class EspnCompetition(EspnModel):
    competitors: list[EspnCompetitor]
    venue: EspnVenue = EspnVenue()
    group: EspnGroup = EspnGroup()
    status: Optional[EspnStatus] = None

    @model_validator(mode="after")
    def _has_both_sides(self):
        if not {"home", "away"} <= {c.home_away for c in self.competitors}:
            raise ValueError("competition needs a home and an away competitor")
        return self

    def side(self, home_away: str) -> EspnCompetitor:
        return next(c for c in self.competitors if c.home_away == home_away)


class EspnSeason(EspnModel):
    slug: str = ""  # "group-stage", "round-of-32", ...


class EspnEvent(EspnModel):
    id: int
    date: Optional[str] = None
    season: EspnSeason = EspnSeason()
    status: Optional[EspnStatus] = None
    competitions: list[EspnCompetition] = Field(min_length=1)


class EspnStat(EspnModel):
    name: str = ""
    value: Optional[float] = None


class EspnStandingsEntry(EspnModel):
    team: EspnTeam
    stats: list[EspnStat] = []

    def stat(self, name: str) -> int:
        return int(next((s.value for s in self.stats if s.name == name), None) or 0)


class EspnStandingsTable(EspnModel):
    entries: list[EspnStandingsEntry] = []


class EspnStandingsGroup(EspnModel):
    name: str = ""  # "Group A"
    standings: EspnStandingsTable = EspnStandingsTable()


class EspnStandings(EspnModel):
    children: list[EspnStandingsGroup] = []


class EspnStatEntry(EspnModel):
    name: str = ""
    display_value: str = ""


class EspnTeamBoxscore(EspnModel):
    team: EspnTeam = EspnTeam()
    statistics: list[EspnStatEntry] = []


class EspnBoxscore(EspnModel):
    teams: list[EspnTeamBoxscore] = []


class EspnText(EspnModel):
    text: str = ""


class EspnClock(EspnModel):
    display_value: str = ""


class EspnParticipant(EspnModel):
    athlete: EspnAthlete = EspnAthlete()


class EspnKeyEvent(EspnModel):
    type: EspnText = EspnText()
    team: EspnTeam = EspnTeam()
    clock: EspnClock = EspnClock()
    participants: list[EspnParticipant] = []
    shootout: bool = False


class EspnPosition(EspnModel):
    abbreviation: str = ""


class EspnRosterPlayer(EspnModel):
    starter: bool = False
    jersey: OptInt = None
    position: EspnPosition = EspnPosition()
    formation_place: str = ""
    captain: bool = False
    subbed_in: bool = False
    subbed_out: bool = False
    athlete: EspnAthlete = EspnAthlete()


class EspnRoster(EspnModel):
    team: EspnTeam = EspnTeam()
    formation: str = ""
    roster: list[EspnRosterPlayer] = []


class EspnSummary(EspnModel):
    boxscore: EspnBoxscore = EspnBoxscore()
    key_events: list[EspnKeyEvent] = []
    rosters: list[EspnRoster] = []
    commentary: list[dict] = []  # parsed by ml.shot_xg
