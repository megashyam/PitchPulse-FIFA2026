"use client"
// components/predict/Top8Chart.tsx

import { useMemo } from "react"
import type { TeamPrediction } from "@/types/predict"

const CHART_H = 130   // px — total chart height

export function Top8Chart({ teams }: { teams: TeamPrediction[] }) {
    const top8 = useMemo(
        () => [...teams].sort((a, b) => b.champion.p - a.champion.p).slice(0, 8),
        [teams]
    )
    const maxP = top8[0]?.champion.p ?? 1

    return (
        <div className="pred-card">
            <div className="pred-card-top">
                <span className="src src-novel">Novel · tournament win distribution</span>
                <span className="pred-card-label">MC output · 95% CI shown</span>
            </div>
            <div className="pred-card-title">Win probability distribution — top 8</div>

            {/* Vertical bar chart */}
            <div className="pred-top8-chart">
                {top8.map((team, idx) => {
                    const p = team.champion.p
                    const barH = Math.max(2, (p / maxP) * CHART_H)
                    const abbr = team.name.slice(0, 3).toUpperCase()
                    const bg = idx === 0
                        ? "var(--c-goal)"
                        : `rgba(168,85,247,${0.9 - idx * 0.08})`

                    return (
                        <div key={team.name} className="pred-top8-col">
                            <div className="pred-top8-bar-wrap">
                                <div
                                    className="pred-top8-bar"
                                    style={{ height: barH, background: bg }}
                                />
                            </div>
                            <span className={`pred-top8-abbr${idx === 0 ? " leader" : ""}`}>
                                {abbr}
                            </span>
                        </div>
                    )
                })}
            </div>

            <p className="pred-card-note">
                Probability distributions, not point estimates. Shows uncertainty from novel format.
            </p>
        </div>
    )
}