"use client"
// hooks/useIntelStream.ts
// SSE subscription + 30s REST poll for live match intel narratives.
//
// Takes the match `statusShort` and:
//   - resolves isWaiting=false after the first fetch completes;
//   - exposes phase: "loading" | "streaming" | "idle" for the component;
//   - opens SSE only for live matches.

import { useEffect, useRef, useState, useCallback } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const POLL_INTERVAL = 30_000

const LIVE = new Set(["1H", "HT", "2H", "ET", "P"])
const COMPLETED = new Set(["FT", "AET", "PEN"])

export interface IntelEntry {
    fixture_id: number
    minute: number
    narration_type: "tactical" | "event_reaction" | "xg_divergence"
    narrative: string
    score: number
    rag_docs_used: number
    via: "mistral" | "groq" | "template"
    updated_at: string
    event_sig?: string
}

type IntelPhase = "loading" | "streaming" | "idle_notlive" | "idle_nodata"

interface UseIntelStreamResult {
    entries: IntelEntry[]
    isWaiting: boolean       // kept for back-compat; true only during first load
    phase: IntelPhase
    error: string | null
}

function dedupeKey(e: IntelEntry): string {
    return e.event_sig ?? `min:${e.minute}:${e.narration_type}`
}

function sortedDeduped(entries: IntelEntry[]): IntelEntry[] {
    const byKey = new Map<string, IntelEntry>()
    for (const e of entries) {
        const key = dedupeKey(e)
        const existing = byKey.get(key)
        if (!existing || e.updated_at > existing.updated_at) byKey.set(key, e)
    }
    return Array.from(byKey.values())
        .sort((a, b) => b.minute - a.minute)
        .slice(0, 30)
}

export function useIntelStream(
    fixtureId: string,
    statusShort?: string,
): UseIntelStreamResult {
    const [entries, setEntries] = useState<IntelEntry[]>([])
    const [firstLoadDone, setFirstLoadDone] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const mountedRef = useRef(true)
    // Entry count for the completed-match poll check, kept out of the
    // subscription effect's dependencies.
    const entriesCountRef = useRef(0)

    const isLive = statusShort ? LIVE.has(statusShort) : false
    const isCompleted = statusShort ? COMPLETED.has(statusShort) : false

    const fetchFeed = useCallback(async () => {
        if (!fixtureId) return
        try {
            const r = await fetch(`${API}/matches/${fixtureId}/intel`)
            console.log("[intel v2] fetch status", r.status, "for", fixtureId)
            if (r.ok) {
                const data: { entries?: IntelEntry[] } = await r.json()
                console.log("[intel v2] got entries:", data?.entries?.length ?? "none")
                if (!mountedRef.current) return
                const list = Array.isArray(data?.entries) ? data.entries : []
                const deduped = sortedDeduped(list)
                setEntries(deduped)
                entriesCountRef.current = deduped.length
                setError(null)
            }
            // 404 (no data) is a normal state, not an error — just fall through.
        } catch (e) {
            console.log("[intel v2] fetch error", e)
        } finally {
            if (mountedRef.current) setFirstLoadDone(true)
        }
    }, [fixtureId])

    useEffect(() => {
        if (!fixtureId) return
        mountedRef.current = true
        setFirstLoadDone(false)
        entriesCountRef.current = 0

        // Always do one fetch — a completed match may have stored history.
        fetchFeed()

        // LIVE matches: poll + SSE indefinitely.
        // COMPLETED matches: poll every 30s for the first few tries (the FT
        // wrap-up lands within ~30s of full time), then every 2 min.
        // NS matches: one fetch, no polling.
        if (!isLive && !isCompleted) {
            return () => { mountedRef.current = false }
        }

        let ftPolls = 0
        const FT_FAST_POLLS = 6       // ~3 min at the normal 30s cadence...
        const FT_SLOW_INTERVAL_MS = 120_000  // ...then every 2 min after that, forever
        let poll: ReturnType<typeof setInterval> = setInterval(tick, POLL_INTERVAL)

        function tick() {
            if (isCompleted) {
                if (entriesCountRef.current > 0) {
                    clearInterval(poll)
                    return
                }
                ftPolls += 1
                if (ftPolls === FT_FAST_POLLS) {
                    // Switch to the slower indefinite cadence instead of
                    // stopping — re-create the interval at the new period.
                    clearInterval(poll)
                    poll = setInterval(tick, FT_SLOW_INTERVAL_MS)
                }
            }
            fetchFeed()
        }

        // SSE only makes sense for live matches.
        let es: EventSource | null = null
        if (isLive) {
            es = new EventSource(`${API}/matches/${fixtureId}/intel/stream`)
            es.addEventListener("intel_update", () => {
                if (mountedRef.current) fetchFeed()
            })
            es.onerror = () => { }
        }

        return () => {
            mountedRef.current = false
            clearInterval(poll)
            if (es) es.close()
        }
        // entries.length is deliberately not a dependency: re-subscribe only
        // when the fixture or its live/completed status changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [fixtureId, isLive, isCompleted, fetchFeed])

    let phase: IntelPhase
    if (entries.length > 0) phase = "streaming"        // have data → always show it
    else if (!firstLoadDone) phase = "loading"
    else if (isLive) phase = "idle_nodata"             // live but nothing narratable yet
    else if (isCompleted) phase = "idle_nodata"        // finished, no stored intel
    else phase = "idle_notlive"                        // NS / pre-match

    return {
        entries,
        isWaiting: !firstLoadDone,
        phase,
        error,
    }
}