"use client"
// components/predict/TournamentTable.tsx

import React, { useState, useMemo } from "react"
import { TeamPrediction, Stage, STAGES, STAGE_LABELS } from "@/types/predict"

// Color per stage for the active sort column's fill bar
const STAGE_FILL: Record<Stage, string> = {
    r32: "#3b82f6",
    r16: "#3b82f6",
    qf: "#7c3aed",
    sf: "#f59e0b",
    final: "#f59e0b",
    champion: "#22c55e",
}

interface Props {
    teams: TeamPrediction[]
    highlightStage?: Stage
}

export function TournamentTable({ teams, highlightStage = "champion" }: Props) {
    const [sortStage, setSortStage] = useState<Stage>(highlightStage)
    const [groupFilter, setGroupFilter] = useState("All")
    const [showCI, setShowCI] = useState(false)

    // Sorted by selected stage descending
    const sorted = useMemo(
        () => [...teams].sort((a, b) => b[sortStage].p - a[sortStage].p),
        [teams, sortStage]
    )

    // Unique group letters for filter pills
    const groups = useMemo(() => {
        const gs = [...new Set(teams.map(t => t.group))].sort()
        return ["All", ...gs]
    }, [teams])

    // Apply group filter
    const visible = useMemo(
        () => groupFilter === "All" ? sorted : sorted.filter(t => t.group === groupFilter),
        [sorted, groupFilter]
    )

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

            {/* Controls row */}
            <div className="predict-filters">
                <div className="filter-pills">
                    {groups.map(g => (
                        <button
                            key={g}
                            onClick={() => setGroupFilter(g)}
                            className={`filter-pill${groupFilter === g ? " active" : ""}`}
                        >
                            {g}
                        </button>
                    ))}
                </div>
                <label className="ci-toggle">
                    <input
                        type="checkbox"
                        checked={showCI}
                        onChange={e => setShowCI(e.target.checked)}
                    />
                    Show 95% CI
                </label>
            </div>

            {/* Table */}
            <div className="tournament-wrap">
                <table className="tournament-table">
                    <thead>
                        <tr>
                            <th style={{ textAlign: "right" }}>#</th>
                            <th style={{ textAlign: "left" }}>Team</th>
                            <th>Grp</th>
                            <th style={{ textAlign: "right" }}>Elo</th>
                            {STAGES.map(stage => (
                                <th
                                    key={stage}
                                    className={`sortable p-bar-cell${sortStage === stage ? " sort-active" : ""}`}
                                    onClick={() => setSortStage(stage)}
                                    style={{ textAlign: "right" }}
                                >
                                    {STAGE_LABELS[stage]}{sortStage === stage ? " ▾" : ""}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {visible.map((team, idx) => (
                            <TeamRow
                                key={team.name}
                                rank={idx + 1}
                                team={team}
                                sortStage={sortStage}
                                showCI={showCI}
                            />
                        ))}
                    </tbody>
                </table>
            </div>

            <p style={{ fontSize: ".65rem", fontFamily: "var(--font-mono)", color: "var(--text-3)", textAlign: "right" }}>
                Sorted by {STAGE_LABELS[sortStage]} · click any column header to re-sort
            </p>
        </div>
    )
}

// ── Individual row ────────────────────────────────────────────────────────

interface RowProps {
    rank: number
    team: TeamPrediction
    sortStage: Stage
    showCI: boolean
}

function TeamRow({ rank, team, sortStage, showCI }: RowProps) {
    return (
        <tr>
            <td className="td-rank">{rank}</td>
            <td className="td-name">{team.name}</td>
            <td className="td-group">
                <span className="group-badge">{team.group}</span>
            </td>
            <td className="td-elo">{team.elo.toFixed(0)}</td>

            {STAGES.map(stage => {
                const sp = team[stage]
                const isSortCol = stage === sortStage
                const fill = isSortCol ? STAGE_FILL[stage] : "var(--border-light)"
                const pct = (sp.p * 100).toFixed(1)

                return (
                    <td key={stage} className="p-bar-cell">
                        <div className="p-bar-wrap">
                            <div className="p-bar-track">
                                {showCI && isSortCol && (
                                    <div
                                        className="p-bar-ci"
                                        style={{
                                            left: `${(sp.ci_lo * 100).toFixed(2)}%`,
                                            width: `${((sp.ci_hi - sp.ci_lo) * 100).toFixed(2)}%`,
                                        }}
                                    />
                                )}
                                <div
                                    className="p-bar-fill"
                                    style={{ width: `${pct}%`, background: fill }}
                                />
                            </div>
                            <span className={`p-bar-val${isSortCol ? " highlight" : ""}`}>
                                {pct}%
                            </span>
                        </div>
                    </td>
                )
            })}
        </tr>
    )
}