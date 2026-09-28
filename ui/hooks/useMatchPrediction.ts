"use client"
// hooks/useMatchPrediction.ts
// Fetches per-match W/D/L prediction + tournament implications.
// Simple REST fetch — no SSE needed (this data changes slowly).

import { useEffect, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export interface MatchOdds {
    p: number
    ci_lo: number
    ci_hi: number
}

export interface TeamTournament {
    name: string
    group: string
    elo: number
    r32: { p: number }
    r16: { p: number }
    qf: { p: number }
    sf: { p: number }
    final: { p: number }
    champion: { p: number }
}

export interface MatchPrediction {
    fixture_id: number
    home_name: string
    away_name: string
    status_short: string
    elapsed: number | null
    match_odds: {
        home_win: MatchOdds
        draw: MatchOdds
        away_win: MatchOdds
    }
    source: string    // "market_odds" | "elo"
    home_tournament: TeamTournament | null
    away_tournament: TeamTournament | null
}

export function useMatchPrediction(fixtureId: string) {
    const [prediction, setPrediction] = useState<MatchPrediction | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        if (!fixtureId) return
        let mounted = true
        setLoading(true)

        fetch(`${API}/matches/${fixtureId}/prediction`)
            .then(r => (r.ok ? r.json() : null))
            .then(data => {
                if (mounted) {
                    setPrediction(data)
                    setLoading(false)
                }
            })
            .catch(e => {
                if (mounted) {
                    setError(String(e))
                    setLoading(false)
                }
            })

        return () => { mounted = false }
    }, [fixtureId])

    return { prediction, loading, error }
}