"use client"
// components/match/PreMatchBriefingCard.tsx
// Polls /briefing/trigger every 60s; the backend generates a new briefing
// only when match status changes, so most polls are no-ops. Renders the
// full feed, newest first.

import { useEffect, useState, useRef } from "react"
import { triggerHeaders } from "@/lib/api"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const POLL_MS = 60_000

interface BriefingEntry {
    fixture_id: number
    home_name: string
    away_name: string
    match_status: string
    briefing: string
    model: string
    generated_at: string
}

interface Props { fixtureId: string }

export function PreMatchBriefingCard({ fixtureId }: Props) {
    const [entries, setEntries] = useState<BriefingEntry[]>([])
    const [loading, setLoading] = useState(true)
    const [triggering, setTriggering] = useState(false)
    const hasTriggeredOnce = useRef(false)

    const fetchFeed = async () => {
        try {
            const r = await fetch(`${API}/matches/${fixtureId}/briefing/feed`)
            if (r.ok) {
                const data = await r.json()
                setEntries(data.briefings ?? [])
            }
        } catch { }
    }

    const triggerAndRefresh = async () => {
        setTriggering(true)
        try {
            await fetch(`${API}/matches/${fixtureId}/briefing/trigger`, { headers: triggerHeaders() })
            await fetchFeed()
        } catch { } finally {
            setTriggering(false)
        }
    }

    useEffect(() => {
        if (!fixtureId) return
        let mounted = true

        const init = async () => {
            setLoading(true)
            await fetchFeed()
            if (mounted) setLoading(false)
            if (!hasTriggeredOnce.current) {
                hasTriggeredOnce.current = true
                await triggerAndRefresh()
            }
        }
        init()

        const interval = setInterval(() => {
            if (mounted) triggerAndRefresh()
        }, POLL_MS)

        return () => { mounted = false; clearInterval(interval) }
    }, [fixtureId])

    if (loading) return (
        <div style={{ padding: "20px 14px", textAlign: "center" }}>
            <div className="spinner" style={{ width: 20, height: 20, margin: "0 auto 10px" }} />
            <div style={{ fontSize: ".78rem", color: "var(--text-3)" }}>Loading briefings…</div>
        </div>
    )

    if (entries.length === 0) return (
        <div style={{ padding: "20px 14px", textAlign: "center" }}>
            <div style={{ fontSize: ".8rem", fontWeight: 600, color: "var(--text-1)", marginBottom: 6 }}>
                {triggering ? "Generating first briefing…" : "No briefing yet"}
            </div>
            <div style={{ fontSize: ".72rem", color: "var(--text-3)" }}>
                Regenerates automatically at kickoff, half-time, and full-time
            </div>
        </div>
    )

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: 10 }}>
            {entries.map((entry) => {
                const genTime = new Date(entry.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                return (
                    <div key={`${entry.match_status}-${entry.generated_at}`} style={{
                        padding: 16,
                        background: "var(--glass-bg-inner)",
                        border: "1px solid var(--glass-border-inner)",
                        borderRadius: "var(--r-md)",
                    }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                            <span style={{
                                fontFamily: "var(--font-mono)", fontSize: ".56rem", textTransform: "none",
                                letterSpacing: "normal", color: "var(--accent)", background: "var(--accent-dim)",
                                padding: "2px 8px", borderRadius: 10,
                            }}>
                                {entry.match_status}
                            </span>
                            <span style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", color: "var(--text-2)" }}>
                                {genTime} · {entry.model}
                            </span>
                        </div>
                        <p style={{ fontSize: ".78rem", color: "var(--text-1)", fontStyle: "italic", lineHeight: 1.55, margin: 0 }}>
                            "{entry.briefing}"
                        </p>
                    </div>
                )
            })}
        </div>
    )
}