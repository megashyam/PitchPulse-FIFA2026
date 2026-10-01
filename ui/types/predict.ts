// types/predict.ts — updated with group finishing position fields

interface StageProbability {
    p: number   // probability 0..1
    ci_lo: number   // 95% CI lower
    ci_hi: number   // 95% CI upper
}

export interface TeamPrediction {
    name: string
    group: string
    elo: number
    fifa_rank: number

    // Group stage finishing position
    group_exit: StageProbability  // eliminated in group
    group_first: StageProbability  // finishes 1st
    group_second: StageProbability  // finishes 2nd
    group_third: StageProbability  // finishes 3rd (eligible for best-thirds)
    group_fourth: StageProbability  // finishes 4th (always eliminated)

    // Knockout stages
    r32: StageProbability  // advances from group (= 1st + 2nd + best_third)
    r16: StageProbability
    qf: StageProbability
    sf: StageProbability
    final: StageProbability
    champion: StageProbability
}

export interface TournamentPrediction {
    sim_id: string
    n_sims: number
    elapsed_s: number
    run_at: string
    teams: TeamPrediction[]
    status: "complete" | "running" | "error"
}

export interface SimStatus {
    status: "idle" | "running" | "complete" | "error"
    sim_id?: string
    started_at?: string
    error?: string
}

export const STAGES = ["r32", "r16", "qf", "sf", "final", "champion"] as const
export type Stage = (typeof STAGES)[number]

export const STAGE_LABELS: Record<Stage, string> = {
    r32: "R32",
    r16: "R16",
    qf: "QF",
    sf: "SF",
    final: "Final",
    champion: "Champion",
}