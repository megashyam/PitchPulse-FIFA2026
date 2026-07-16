"use client"

import type { MatchState } from "@/types/match"

export function XGCard({ state }: { state: MatchState }) {
    const homeXG = state.home_stats.expected_goals ?? 0
    const awayXG = state.away_stats.expected_goals ?? 0
    const diff = homeXG - awayXG

    return (
        <div>
            <div className="stats-section-header">Expected Goals (xG)</div>
            <div style={{
                display: "grid", gridTemplateColumns: "1fr auto 1fr",
                alignItems: "center", padding: "3px 14px 5px", gap: 4,
            }}>
                <span style={{ fontSize: ".7rem", fontWeight: 700, color: "var(--home)" }}>
                    {homeXG > 0 ? homeXG.toFixed(2) : "—"}
                </span>
                <span style={{
                    fontSize: ".58rem", textAlign: "center",
                    color: Math.abs(diff) < 0.1 ? "var(--text-3)"
                        : diff > 0 ? "var(--accent)"
                            : "var(--away)"
                }}>
                    {homeXG > 0 || awayXG > 0
                        ? `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}`
                        : "—"}
                </span>
                <span style={{ fontSize: ".7rem", fontWeight: 700, color: "var(--away)", textAlign: "right" }}>
                    {awayXG > 0 ? awayXG.toFixed(2) : "—"}
                </span>
            </div>
        </div>
    )
}