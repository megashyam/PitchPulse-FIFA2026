"use client"
// hooks/useNarrativeStream.ts
// Feeds the tournament-wide "trending now" hub: fetches /narrative/trending
// (ALL topics ranked by buzz, not only alerted surges), polls every
// 20s, and re-fetches whenever the SSE stream signals a new anomaly.
//
// Spikes may carry optional `fixture_id` / `match_status` fields for the
// live/kickoff badge.

import { useEffect, useState, useCallback, useRef } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const POLL_INTERVAL = 20_000

export interface NarrativeSpike {
    spike_id: string
    topic: string
    tick: number
    severity: number
    sources: {
        mastodon: number
        bluesky: number
        trends: number
        wikipedia: number
    }
    source_names: string[]
    summary: string
    timestamp: number
    arc: string | null
    is_spike?: boolean
    // Optional fixture fields for the live/kickoff badge
    fixture_id?: number | null
    match_status?: string | null
}

interface UseNarrativeStreamResult {
    spikes: NarrativeSpike[]
    spikeCount: number
    isWarming: boolean
    isWaiting: boolean
    error: string | null
}

export function useNarrativeStream(): UseNarrativeStreamResult {
    const [spikes, setSpikes] = useState<NarrativeSpike[]>([])
    const [isWaiting, setIsWaiting] = useState(true)
    const [isWarming, setIsWarming] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const mounted = useRef(true)

    const fetchTrending = useCallback(async () => {
        try {
            const r = await fetch(`${API}/narrative/trending?limit=12`)
            if (!r.ok) return
            const data = await r.json()
            if (!mounted.current) return
            if (Array.isArray(data?.spikes)) {
                setSpikes(data.spikes)
                if (data.spikes.length) setIsWaiting(false)
                setError(null)
            }
        } catch {
            /* keep waiting; next poll retries */
        }
    }, [])

    useEffect(() => {
        mounted.current = true
        fetchTrending()
        const poll = setInterval(fetchTrending, POLL_INTERVAL)

        // SSE only nudges a refetch when a fresh anomaly lands.
        const es = new EventSource(`${API}/narrative/stream`)
        es.addEventListener("narrative_spike", () => {
            if (mounted.current) fetchTrending()
        })
        es.addEventListener("warming", () => {
            if (mounted.current) setIsWarming(true)
        })
        es.onerror = () => { }

        return () => {
            mounted.current = false
            clearInterval(poll)
            es.close()
        }
    }, [fetchTrending])

    return { spikes, spikeCount: spikes.length, isWarming, isWaiting, error }
}
