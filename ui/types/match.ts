// types/match.ts
// Mirrors Python MatchState / TeamStats / MatchEvent from schemas/schema.py

export interface TeamStats {
  possession: number
  shots_total: number
  shots_on_goal: number
  shots_off_goal: number
  passes_total: number
  passes_accurate: number
  pass_accuracy: number
  corner_kicks: number
  fouls: number
  offsides: number
  yellow_cards: number
  red_cards: number
  goalkeeper_saves: number
  expected_goals: number
}

export interface MatchEvent {
  elapsed: number
  extra: number | null
  team_id: number
  team_name: string
  player_name: string | null
  type: string
  detail: string | null
  source?: "espn" | "synthesised"
}

export interface MatchState {
  fixture_id: number
  league_id: number
  season: number
  round: string
  venue: string
  referee: string
  status_short: string
  status_long: string
  elapsed: number | null
  elapsed_extra?: number | null  // stoppage minutes ("90'+4'")
  kickoff_time: string | null
  home_id: number
  home_name: string
  home_logo: string
  home_score: number
  home_stats: TeamStats
  away_id: number
  away_name: string
  away_logo: string
  away_score: number
  away_stats: TeamStats
  events: MatchEvent[]
  stats_source?: "espn" | "unavailable"
  home_pens?: number | null
  away_pens?: number | null
  updated_at: string
}