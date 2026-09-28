"use client"
// hooks/useMomentumStream.ts
// SSE subscription for live momentum data.
// Mirrors the pattern of useMatchStream.ts — REST fetch for immediate
// display then SSE for live updates every ~30s.
//
// A 200 {"status": "not_started"} from the REST endpoint is normalized to
// `momentum: null`. No SSE stream is opened before kickoff; pass
// `statusShort` so the hook knows to wait.

import { useEffect, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const LIVE = new Set(["1H", "HT", "2H", "ET", "P"])

export interface TeamMomentumData {
    momentum_score: number   // 0-1 relative (home + away always = 1)
    goal_prob_5min: number   // absolute P(goal in next 5 min)
    xg_15min: number     // model xG from shots in the last 15 minutes
    shots_15min: number
    xg_total: number
}

export interface MomentumSnapshot {
    fixture_id: number
    home_name: string
    away_name: string
    elapsed: number | null
    home: TeamMomentumData
    away: TeamMomentumData
    updated_at: string
}

interface UseMomentumStreamResult {
    momentum: MomentumSnapshot | null
    isWaiting: boolean
    error: string | null
}

function normalize(data: any): MomentumSnapshot | null {
    if (!data || data.status === "not_started") return null
    return data as MomentumSnapshot
}

export function useMomentumStream(
    fixtureId: string,
    statusShort?: string,
): UseMomentumStreamResult {
    const [momentum, setMomentum] = useState<MomentumSnapshot | null>(null)
    const [isWaiting, setWaiting] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // If status isn't known yet, don't assume NS — fetch once regardless so
    // a live/completed match with real data doesn't wait on a status prop
    // that hasn't arrived from a parent yet.
    const knownNotLive = statusShort !== undefined && !LIVE.has(statusShort)

    useEffect(() => {
        if (!fixtureId) return
        let mounted = true

        fetch(`${API}/matches/${fixtureId}/momentum`)
            .then(r => (r.ok ? r.json() : null))
            .then(data => {
                if (mounted) setMomentum(normalize(data))
            })
            .catch(() => { })

        // Nothing will ever arrive on the stream for a match that isn't
        // live — skip opening the connection entirely rather than leaving
        // an idle EventSource sitting on the "waiting" heartbeat.
        if (knownNotLive) return () => { mounted = false }

        const es = new EventSource(`${API}/matches/${fixtureId}/momentum/stream`)

        es.addEventListener("momentum_update", (e: MessageEvent) => {
            if (!mounted) return
            try {
                setMomentum(JSON.parse(e.data) as MomentumSnapshot)
                setWaiting(false)
                setError(null)
            } catch {
                setError("Failed to parse momentum update")
            }
        })

        es.addEventListener("waiting", () => {
            if (mounted) setWaiting(true)
        })

        es.onerror = () => { }

        return () => {
            mounted = false
            es.close()
        }
    }, [fixtureId, knownNotLive])

    return { momentum, isWaiting, error }
}