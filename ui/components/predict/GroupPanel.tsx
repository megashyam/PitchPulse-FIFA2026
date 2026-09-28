"use client"
// components/predict/GroupPanel.tsx

import { useEffect, useMemo, useState } from "react"
import type { TeamPrediction } from "@/types/predict"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export function GroupPanel({ teams }: { teams: TeamPrediction[] }) {
    // Build group map sorted by 1st-place probability
    const groupMap = useMemo(() => {
        const map = new Map<string, TeamPrediction[]>()
        for (const t of teams) {
            if (!map.has(t.group)) map.set(t.group, [])
            map.get(t.group)!.push(t)
        }
        map.forEach((ts, g) =>
            map.set(g, [...ts].sort((a, b) =>
                (b.group_first?.p ?? b.r32?.p ?? 0) - (a.group_first?.p ?? a.r32?.p ?? 0)
            ))
        )
        return map
    }, [teams])

    const groupKeys = useMemo(() => [...groupMap.keys()].sort(), [groupMap])

    // Auto-detect active match group
    const [active, setActive] = useState("")
    useEffect(() => {
        if (!teams.length) return
        async function detect() {
            try {
                const r = await fetch(`${API}/matches`)
                if (!r.ok) return
                const { fixtures: ids } = await r.json()
                if (!ids?.length) return
                const state = await fetch(`${API}/matches/${ids[0]}`).then(r => r.json())
                const found = teams.find(t => t.name === state.home_name)
                setActive(found?.group ?? groupKeys[0] ?? "")
            } catch {
                setActive(groupKeys[0] ?? "")
            }
        }
        detect()
    }, [teams, groupKeys])

    const groupTeams = groupMap.get(active) ?? []

    // Build flat rows: 1st × all teams, 2nd × all teams, WC win × all teams
    const rows = useMemo(() => {
        const out: { label: string; p: number; kind: "first" | "second" | "wc" }[] = []
        groupTeams.forEach(t => out.push({ label: `${t.name} 1st`, p: t.group_first?.p ?? 0, kind: "first" }))
        groupTeams.forEach(t => out.push({ label: `${t.name} 2nd`, p: t.group_second?.p ?? 0, kind: "second" }))
        groupTeams.forEach(t => out.push({ label: `${t.name} win WC`, p: t.champion?.p ?? 0, kind: "wc" }))
        return out
    }, [groupTeams])

    const maxFirst = Math.max(...groupTeams.map(t => t.group_first?.p ?? 0), 0.01)
    const maxSecond = Math.max(...groupTeams.map(t => t.group_second?.p ?? 0), 0.01)
    const maxWC = Math.max(...groupTeams.map(t => t.champion?.p ?? 0), 0.01)

    const maxFor = (kind: "first" | "second" | "wc") =>
        kind === "first" ? maxFirst : kind === "second" ? maxSecond : maxWC

    const colorFor = (kind: "first" | "second" | "wc") =>
        kind === "first" ? "var(--home)" : kind === "second" ? "var(--text-3)" : "var(--amber)"

    return (
        <div className="pred-card">
            <div className="pred-card-top">
                <span className="src src-novel">Novel · 48-team Monte Carlo</span>
                <span className="pred-card-label">50k runs · numpy · CPU ~8s</span>
            </div>

            {/* Group selector pills */}
            <div className="pred-group-pills">
                {groupKeys.map(g => (
                    <button
                        key={g}
                        className={`pred-group-pill${active === g ? " active" : ""}`}
                        onClick={() => setActive(g)}
                    >
                        {g}
                    </button>
                ))}
            </div>

            {active && (
                <div className="pred-group-subtitle">Group {active} — advancement probabilities</div>
            )}

            <div className="pred-group-rows">
                {rows.map((row, i) => (
                    <div key={i} className="pred-group-row">
                        <span className="pred-group-row-label">{row.label}</span>
                        <div className="pred-group-row-track">
                            <div
                                className="pred-group-row-fill"
                                style={{
                                    width: row.p > 0 ? `${((row.p / maxFor(row.kind)) * 100).toFixed(1)}%` : "0%",
                                    background: colorFor(row.kind),
                                }}
                            />
                        </div>
                        <span className="pred-group-row-pct" style={{ color: colorFor(row.kind) }}>
                            {row.p > 0 ? `${(row.p * 100).toFixed(0)}%` : "—"}
                        </span>
                    </div>
                ))}
            </div>

            <p className="pred-card-note">
                First model to simulate 48-team format + 3rd-place advancement rule. Re-runs after every result.
            </p>
        </div>
    )
}