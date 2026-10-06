"""
MatchEvent.type vocabulary and match status sets.
"""

GOAL_TYPES = frozenset({"goal", "own_goal", "penalty_goal"})
RED_TYPES = frozenset({"red", "yellow_red"})
SIGNIFICANT_TYPES = GOAL_TYPES | RED_TYPES

# triggers for a counterfactual cycle; yellows and subs give an exact-zero bracket shift
# (see counterfactual_agent) but stay triggers, so each match shows a "no impact" entry
# next to the goals/reds that move the bracket
TRIGGER_TYPES = SIGNIFICANT_TYPES | frozenset({"yellow", "substitution"})

LIVE_STATUSES = frozenset({"1H", "HT", "2H", "ET", "P"})
IN_PLAY = LIVE_STATUSES - {"HT"}  # ball in play: colour narration runs
COMPLETED_STATUSES = frozenset({"FT", "AET", "PEN"})
