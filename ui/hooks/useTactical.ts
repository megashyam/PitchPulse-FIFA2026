"use client"
// hooks/useTactical.ts
// Fetches the tactical fingerprint match for a fixture from the
// /matches/{id}/tactical endpoint (backed by the Weaviate TacticalProfiles
// collection). Simple REST — this data changes slowly within a match
// (cached 10 min server-side), so no SSE needed.
//
// Returns null when no fingerprint exists yet (404) or the match hasn't
// started (200 {"status": "not_started"}).

import { useEffect, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export interface FingerprintMatch {
    team: string
    opponent: string
    competition: string
    season: string
    possession_gap_pp: number | null
    ppda: number
    ppda_mid_third: number
    ppda_att_third: number
    possession: number
    press_intensity: number
    content: string
}

export interface TeamFingerprint {
    team: string
    opponent: string
    live_possession: number
    match: FingerprintMatch
    alternatives: {
        team: string
        season: string
        ppda: number
        possession_gap_pp: number | null
    }[]
}

export interface TacticalData {
    fixture_id: number
    home_name: string
    away_name: string
    home: TeamFingerprint | null
    away: TeamFingerprint | null
    source: string
    generated_at: string
}

function normalize(data: any): TacticalData | null {
    if (!data || data.status === "not_started" || data.status === "pending") return null
    return data as TacticalData
}

export function useTactical(fixtureId: string) {
    const [tactical, setTactical] = useState<TacticalData | null>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        if (!fixtureId) return
        let mounted = true
        setLoading(true)

        fetch(`${API}/matches/${fixtureId}/tactical`)
            .then(r => (r.ok ? r.json() : null))
            .then(data => {
                if (mounted) {
                    setTactical(normalize(data))
                    setLoading(false)
                }
            })
            .catch(() => {
                if (mounted) setLoading(false)
            })

        // Refresh every 60s; a worker refresh or kickoff can change it before
        // the 10-min server cache expires.
        const t = setInterval(() => {
            fetch(`${API}/matches/${fixtureId}/tactical`)
                .then(r => (r.ok ? r.json() : null))
                .then(data => {
                    if (mounted) setTactical(normalize(data))
                })
                .catch(() => { })
        }, 60_000)

        return () => {
            mounted = false
            clearInterval(t)
        }
    }, [fixtureId])

    return { tactical, loading }
}