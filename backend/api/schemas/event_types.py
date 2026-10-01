"""
MatchEvent.type vocabulary and match status sets.
"""

GOAL_TYPES = frozenset({"goal", "own_goal", "penalty_goal"})
RED_TYPES = frozenset({"red", "yellow_red"})
SIGNIFICANT_TYPES = GOAL_TYPES | RED_TYPES

# Everything that can trigger a counterfactual cycle. Yellows/subs resolve to
# an exact-zero bracket shift by construction (see counterfactual_agent), but
# remain triggers so both live and completed matches surface an honest
# "no impact" entry alongside the goals/reds that actually move the bracket.
TRIGGER_TYPES = SIGNIFICANT_TYPES | frozenset({"yellow", "substitution"})

LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P"})
COMPLETED_STATUSES = frozenset({"FT", "AET", "PEN"})
