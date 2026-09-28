"use client"
// components/predict/OutcomeBar.tsx
// StageBar     — single probability bar, used in TournamentTable
// MatchOutcomeBar — stacked W/D/L bar for the per-match view

import React from "react"

// ── StageBar ───────────────────────────────────────────────────────────────

interface StageBarProps {
    p: number        // probability 0..1
    ci_lo: number
    ci_hi: number
    fill?: string    // CSS color string — defaults to var(--border-light)
    showCI?: boolean
}

export function StageBar({ p, ci_lo, ci_hi, fill = "var(--border-light)", showCI = false }: StageBarProps) {
    const pct = (p * 100).toFixed(1)

    return (
        <div className="p-bar-wrap">
            <div className="p-bar-track">
                {showCI && (
                    <div
                        className="p-bar-ci"
                        style={{
                            left: `${(ci_lo * 100).toFixed(2)}%`,
                            width: `${((ci_hi - ci_lo) * 100).toFixed(2)}%`,
                        }}
                    />
                )}
                <div
                    className="p-bar-fill"
                    style={{ width: `${pct}%`, background: fill }}
                />
            </div>
            <span className="p-bar-val">
                {pct}%
                {showCI && (
                    <span style={{ display: "block", fontSize: ".55rem", color: "var(--text-3)" }}>
                        {(ci_lo * 100).toFixed(1)}–{(ci_hi * 100).toFixed(1)}
                    </span>
                )}
            </span>
        </div>
    )
}

// ── MatchOutcomeBar ─────────────────────────────────────────────────────────

interface MatchOutcomeBarProps {
    homeTeam: string
    awayTeam: string
    pHome: number
    pDraw: number
    pAway: number
}

export function MatchOutcomeBar({ homeTeam, awayTeam, pHome, pDraw, pAway }: MatchOutcomeBarProps) {
    const segs = [
        { label: homeTeam, p: pHome, bg: "var(--home)", color: "#fff" },
        { label: "Draw", p: pDraw, bg: "var(--bg-4)", color: "var(--text-2)" },
        { label: awayTeam, p: pAway, bg: "var(--away)", color: "#fff" },
    ]

    return (
        <div className="outcome-bar-wrap">
            <div className="outcome-bar-stacked">
                {segs.map(({ label, p, bg, color }) => (
                    <div
                        key={label}
                        className="outcome-seg"
                        style={{ width: `${(p * 100).toFixed(2)}%`, background: bg, color }}
                    >
                        {p > 0.08 ? `${(p * 100).toFixed(0)}%` : ""}
                    </div>
                ))}
            </div>
            <div className="outcome-legend">
                <span style={{ color: "var(--home)" }}>{homeTeam}: {(pHome * 100).toFixed(1)}%</span>
                <span>Draw: {(pDraw * 100).toFixed(1)}%</span>
                <span style={{ color: "var(--away)" }}>{awayTeam}: {(pAway * 100).toFixed(1)}%</span>
            </div>
        </div>
    )
}