"use client"
// components/match/GroupTable.tsx
// FotMob-style group standings table

import { useEffect, useState } from "react"
import { Flag } from "@/components/Flag"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface TableRow {
    pos: number; name: string; p: number; w: number; d: number; l: number
    gf: number; ga: number; gd: number; pts: number
}
interface GroupTableData { group: string; table: TableRow[] }

interface Props { fixtureId: string; highlightTeams: [string, string] }

export function GroupTable({ fixtureId, highlightTeams }: Props) {
    const [data, setData] = useState<GroupTableData | null>(null)
    const [err, setErr] = useState(false)

    useEffect(() => {
        fetch(`${API}/matches/${fixtureId}/group-table`)
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(setData)
            .catch(() => setErr(true))
    }, [fixtureId])

    if (err || !data) return null

    return (
        <div className="fm-group-table">
            <div className="fm-gt-header">
                <span style={{ fontSize: "1.1rem" }}>🏆</span>
                <span className="fm-gt-title">Group {data.group}</span>
            </div>


            <div className="fm-gt-cols">
                <div className="fm-gt-col-label">#</div>
                <div className="fm-gt-col-label left">Team</div>
                <div className="fm-gt-col-label">PL</div>
                <div className="fm-gt-col-label">W</div>
                <div className="fm-gt-col-label">D</div>
                <div className="fm-gt-col-label">L</div>
                <div className="fm-gt-col-label">+/-</div>
                <div className="fm-gt-col-label">GD</div>
                <div className="fm-gt-col-label">PTS</div>
            </div>

            {data.table.map(row => (
                <div
                    key={row.name}
                    className={`fm-gt-row${highlightTeams.includes(row.name) ? " highlighted" : ""}`}
                >
                    <div className="fm-gt-pos">{row.pos}</div>
                    <div className="fm-gt-team">
                        <Flag team={row.name} size="sm" />
                        <span className="fm-gt-team-name">{row.name}</span>
                    </div>
                    <div className="fm-gt-val">{row.p}</div>
                    <div className="fm-gt-val">{row.w}</div>
                    <div className="fm-gt-val">{row.d}</div>
                    <div className="fm-gt-val">{row.l}</div>
                    <div className="fm-gt-val">{row.gf}-{row.ga}</div>
                    <div className="fm-gt-val">{row.gd >= 0 ? `+${row.gd}` : row.gd}</div>
                    <div className="fm-gt-pts">{row.pts}</div>
                </div>
            ))}
        </div>
    )
}