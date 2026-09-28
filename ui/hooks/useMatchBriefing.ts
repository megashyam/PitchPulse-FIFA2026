"use client"
// hooks/useMatchBriefing.ts — adds refetch() for polling after auto-trigger

import { useEffect, useState, useCallback } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export interface MatchBriefing {
    fixture_id: number
    home_name: string
    away_name: string
    briefing: string
    model: string
    source: string
    generated_at: string
}

export function useMatchBriefing(fixtureId: string) {
    const [briefing, setBriefing] = useState<MatchBriefing | null>(null)
    const [loading, setLoading] = useState(true)

    const fetchBriefing = useCallback(async () => {
        if (!fixtureId) return
        try {
            const r = await fetch(`${API}/matches/${fixtureId}/briefing`)
            if (r.ok) {
                const data = await r.json()
                setBriefing(data)
            }
        } catch { }
    }, [fixtureId])

    useEffect(() => {
        setLoading(true)
        fetchBriefing().finally(() => setLoading(false))
    }, [fetchBriefing])

    return { briefing, loading, refetch: fetchBriefing }
}