"use client"
// components/match/PreMatchBriefingCard.tsx
// v2 — polls /briefing/trigger periodically (every 60s) instead of
// firing once on mount. Since the backend only actually generates a
// NEW briefing when match status has changed (see briefing_routes.py
// v3), this is safe to poll continuously — it's a no-op most of the
// time and naturally produces a fresh entry at kickoff, half-time,
// and full-time without needing a separate backend worker loop.
// Renders the full feed (newest first) instead of a single static block.

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
        <div style={{ display: "flex", flexDirection: "column" }}>
            {entries.map((entry, i) => {
                const genTime = new Date(entry.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                return (
                    <div key={`${entry.match_status}-${entry.generated_at}`} style={{
                        padding: "12px 14px",
                        borderBottom: i < entries.length - 1 ? "1px solid var(--border)" : "none",
                    }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                            <span style={{
                                fontFamily: "var(--font-mono)", fontSize: ".56rem", textTransform: "uppercase",
                                letterSpacing: ".08em", color: "var(--accent)", background: "var(--accent-dim)",
                                padding: "2px 8px", borderRadius: 10,
                            }}>
                                {entry.match_status}
                            </span>
                            <span style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", color: "var(--text-3)" }}>
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