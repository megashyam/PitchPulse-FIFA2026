"use client"


import { useEffect, useState } from "react"
import { Flag } from "@/components/Flag"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface FormResult {
    opponent: string
    home_score: number
    away_score: number
    result: "W" | "D" | "L"
    was_home: boolean
    date?: string
}
interface TeamForm { team: string; form: FormResult[] }
interface FormData { home: TeamForm; away: TeamForm }

const RESULT_STYLES = {
    W: { bg: "#16a34a", color: "#fff" },
    D: { bg: "var(--bg-4)", color: "var(--text-2)" },
    L: { bg: "#dc2626", color: "#fff" },
}

function FormRow({ result }: { result: FormResult }) {
    const { bg, color } = RESULT_STYLES[result.result]
    const homeScore = result.was_home ? result.home_score : result.away_score
    const awayScore = result.was_home ? result.away_score : result.home_score
    return (
        <div style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "6px 0", borderBottom: "1px solid var(--border)"
        }}>
            <span style={{
                width: 22, height: 22, borderRadius: 4,
                background: bg, color, display: "flex", alignItems: "center",
                justifyContent: "center", fontFamily: "var(--font-mono)",
                fontSize: ".68rem", fontWeight: 700, flexShrink: 0
            }}>
                {result.result}
            </span>
            <Flag team={result.opponent} size="sm" />
            <span style={{
                fontSize: ".74rem", color: "var(--text-2)", flex: 1,
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis"
            }}>
                {result.opponent.length > 12 ? result.opponent.slice(0, 11) + "." : result.opponent}
            </span>
            <span style={{
                fontFamily: "var(--font-mono)", fontSize: ".72rem",
                fontWeight: 600, color: "var(--text-1)", flexShrink: 0
            }}>
                {homeScore}–{awayScore}
            </span>
        </div>
    )
}

interface Props { fixtureId: string; homeTeam: string; awayTeam: string }

export function TeamFormCard({ fixtureId, homeTeam, awayTeam }: Props) {
    const [data, setData] = useState<FormData | null>(null)

    useEffect(() => {
        fetch(`${API}/matches/${fixtureId}/team-form`)
            .then(r => r.ok ? r.json() : null)
            .then(setData)
            .catch(() => { })
    }, [fixtureId])

    if (!data) return null
    if (!data.home.form.length && !data.away.form.length) return null

    return (
        <div style={{ borderTop: "1px solid var(--border-bright)" }}>
            <div style={{
                padding: "10px 14px 8px", borderBottom: "1px solid var(--border)",
                display: "flex", alignItems: "center", gap: 6
            }}>
                <div style={{ width: 3, height: 10, background: "var(--amber)", borderRadius: 2 }} />
                <span style={{
                    fontFamily: "var(--font-mono)", fontSize: ".6rem",
                    textTransform: "uppercase", letterSpacing: ".1em", color: "var(--text-3)"
                }}>
                    Recent Form
                </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 0 }}>

                <div style={{ padding: "10px 12px", borderRight: "1px solid var(--border)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8 }}>
                        <Flag team={homeTeam} size="sm" />
                        <span style={{ fontSize: ".76rem", fontWeight: 700, color: "var(--home)" }}>
                            {homeTeam.length > 12 ? homeTeam.slice(0, 11) + "." : homeTeam}
                        </span>
                    </div>
                    {data.home.form.length === 0 ? (
                        <div style={{ fontSize: ".7rem", color: "var(--text-3)", fontStyle: "italic" }}>
                            No recent results
                        </div>
                    ) : (
                        data.home.form.slice(0, 5).map((r, i) => (
                            <FormRow key={i} result={r} />
                        ))
                    )}
                </div>

                <div style={{ padding: "10px 12px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8 }}>
                        <Flag team={awayTeam} size="sm" />
                        <span style={{ fontSize: ".76rem", fontWeight: 700, color: "var(--away)" }}>
                            {awayTeam.length > 12 ? awayTeam.slice(0, 11) + "." : awayTeam}
                        </span>
                    </div>
                    {data.away.form.length === 0 ? (
                        <div style={{ fontSize: ".7rem", color: "var(--text-3)", fontStyle: "italic" }}>
                            No recent results
                        </div>
                    ) : (
                        data.away.form.slice(0, 5).map((r, i) => (
                            <FormRow key={i} result={r} />
                        ))
                    )}
                </div>

            </div>
        </div>
    )
}