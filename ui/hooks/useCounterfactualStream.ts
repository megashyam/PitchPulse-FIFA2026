"use client"
// hooks/useCounterfactualStream.ts
// REST /feed for history plus SSE for new results, deduped by
// (minute, type, team) signature. Completed matches (FT/AET/PEN) skip SSE.
//
// `calculating` is set by the "counterfactual_calculating" SSE event when a
// simulation starts, and cleared when a matching result lands or after a
// safety timeout.

import { useEffect, useRef, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const COMPLETED = new Set(["FT", "AET", "PEN"])
const CALCULATING_TIMEOUT_MS = 40_000 // safety net — sims normally finish in ~8-16s

interface CfChange {
    team: string
    before: number
    after: number
    delta: number
}

export interface CfResult {
    fixture_id: number
    minute: number
    event_type: string
    event_team: string
    path_shift_pct: number
    top_changes: CfChange[]
    narrative: string
    via?: string
    n_sims: number
    elapsed_s: number
    updated_at: string
}

export interface CfCalculating {
    minute: number
    event_type: string
    event_team: string
}

interface UseCfStreamResult {
    results: CfResult[]   // newest first, deduped
    calculating: CfCalculating | null
    isWaiting: boolean
    error: string | null
}

function sigOf(r: { minute: number; event_type: string; event_team: string }): string {
    return `${r.minute}:${r.event_type}:${r.event_team}`
}

function dedupe(list: CfResult[]): CfResult[] {
    const seen = new Set<string>()
    const out: CfResult[] = []
    for (const r of list) {
        const s = sigOf(r)
        if (seen.has(s)) continue
        seen.add(s)
        out.push(r)
    }
    return out
}

export function useCounterfactualStream(
    fixtureId: string,
    statusShort?: string,
): UseCfStreamResult {
    const [results, setResults] = useState<CfResult[]>([])
    const [calculating, setCalculating] = useState<CfCalculating | null>(null)
    const [isWaiting, setWaiting] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const calcTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const isCompleted = statusShort ? COMPLETED.has(statusShort) : false

    useEffect(() => {
        if (!fixtureId) return
        let mounted = true

        // Initial REST fetch — this is the full history for the match so far
        fetch(`${API}/matches/${fixtureId}/counterfactual/feed`)
            .then(r => (r.ok ? r.json() : null))
            .then((data: { entries: CfResult[] } | null) => {
                if (mounted && data?.entries?.length) {
                    setResults(dedupe(data.entries))
                }
            })
            .catch(() => { })

        // Completed matches: no live updates left to stream — REST fetch above
        // is the complete and final history. Skip SSE entirely.
        if (isCompleted) return () => { mounted = false }

        // Live/upcoming matches: subscribe for real-time updates
        const es = new EventSource(`${API}/matches/${fixtureId}/counterfactual/stream`)

        es.addEventListener("counterfactual_calculating", (e: MessageEvent) => {
            if (!mounted) return
            try {
                const info = JSON.parse(e.data) as CfCalculating & { fixture_id: number }
                setCalculating({ minute: info.minute, event_type: info.event_type, event_team: info.event_team })
                setWaiting(false)

                if (calcTimeoutRef.current) clearTimeout(calcTimeoutRef.current)
                calcTimeoutRef.current = setTimeout(() => {
                    if (mounted) setCalculating(null)
                }, CALCULATING_TIMEOUT_MS)
            } catch {
                // ignore malformed calculating pings — worst case the loader never shows for this one
            }
        })

        es.addEventListener("counterfactual_update", (e: MessageEvent) => {
            if (!mounted) return
            try {
                const result = JSON.parse(e.data) as CfResult
                setResults(prev => dedupe([result, ...prev]).slice(0, 20))
                setWaiting(false)
                setError(null)

                // Clear the loader once the matching (or any newer) result lands.
                setCalculating(prev => {
                    if (!prev) return prev
                    if (sigOf(prev) === sigOf(result) || result.minute >= prev.minute) {
                        if (calcTimeoutRef.current) clearTimeout(calcTimeoutRef.current)
                        return null
                    }
                    return prev
                })
            } catch {
                setError("Failed to parse counterfactual update")
            }
        })

        es.addEventListener("waiting", () => {
            if (mounted) setWaiting(true)
        })

        es.onerror = () => { }

        return () => {
            mounted = false
            if (calcTimeoutRef.current) clearTimeout(calcTimeoutRef.current)
            es.close()
        }
    }, [fixtureId, isCompleted])

    return { results, calculating, isWaiting, error }
}
