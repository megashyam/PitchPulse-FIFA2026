"use client"
// components/predict/PerMatchOddsPanel.tsx

import { useEffect, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface MatchOdds {
    fixture_id: number
    home_name: string
    away_name: string
    status_short: string
    elapsed: number | null
    match_odds: {
        home_win: { p: number }
        draw: { p: number }
        away_win: { p: number }
    }
    source: string
}

function useAllMatchOdds() {
    const [matches, setMatches] = useState<MatchOdds[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        let mounted = true
        async function load() {
            try {
                const r = await fetch(`${API}/matches`)
                if (!r.ok) return
                const { fixtures: ids }: { fixtures: string[] } = await r.json()
                const results = await Promise.all(
                    ids.map(id =>
                        fetch(`${API}/matches/${id}/prediction`)
                            .then(r => r.ok ? r.json() : null)
                            .catch(() => null)
                    )
                )
                if (mounted) { setMatches(results.filter(Boolean)); setLoading(false) }
            } catch { if (mounted) setLoading(false) }
        }
        load()
        const t = setInterval(load, 30_000)
        return () => { mounted = false; clearInterval(t) }
    }, [])

    return { matches, loading }
}

export function PerMatchOddsPanel() {
    const { matches, loading } = useAllMatchOdds()

    return (
        <div className="pred-card">
            <div className="pred-card-top">
                {/* Source badge */}
                <span className="src src-api">The Odds API · market odds</span>
                <span className="pred-card-label">consumed directly</span>
            </div>
            <div className="pred-odds-subtitle">Per-match win probabilities (input to MC sim)</div>

            {loading && <p className="pred-odds-empty">Loading market data…</p>}
            {!loading && matches.length === 0 && (
                <p className="pred-odds-empty">No active fixtures. Start the mock producer.</p>
            )}

            {matches.map(m => <MatchCard key={m.fixture_id} m={m} />)}

            {matches.length > 0 && (
                <p className="pred-card-note">
                    Averaged across all available US books (DraftKings, FanDuel, BetMGM, etc).
                    Overround removed via normalisation.{" "}
                    {matches[0]?.source === "elo" && "No live odds — using Elo fallback."}
                </p>
            )}
        </div>
    )
}

function MatchCard({ m }: { m: MatchOdds }) {
    const hw = m.match_odds.home_win.p
    const d = m.match_odds.draw.p
    const aw = m.match_odds.away_win.p
    const isLive = ["1H", "HT", "2H", "ET", "P"].includes(m.status_short)

    return (
        <div className="pred-odds-match">
            <div className="pred-odds-match-hdr">
                <div className="pred-odds-teams">
                    <span className="pred-odds-team home">{m.home_name}</span>
                    <span className="pred-odds-vs">vs</span>
                    <span className="pred-odds-team away">{m.away_name}</span>
                </div>
                <span className="pred-odds-clock">
                    {isLive && m.elapsed ? `${m.elapsed}'` : m.status_short}
                </span>
            </div>

            <div className="pred-odds-bar">
                <div className="pred-odds-seg home" style={{ width: `${(hw * 100).toFixed(1)}%` }}>
                    {hw > 0.12 ? `${(hw * 100).toFixed(0)}%` : ""}
                </div>
                <div className="pred-odds-seg draw" style={{ width: `${(d * 100).toFixed(1)}%` }}>
                    {d > 0.12 ? `${(d * 100).toFixed(0)}%` : ""}
                </div>
                <div className="pred-odds-seg away" style={{ width: `${(aw * 100).toFixed(1)}%` }}>
                    {aw > 0.12 ? `${(aw * 100).toFixed(0)}%` : ""}
                </div>
            </div>

            <div className="pred-odds-legend">
                <span style={{ color: "var(--home)" }}>
                    {m.home_name.split(" ").pop()} win {(hw * 100).toFixed(0)}%
                </span>
                <span>Draw {(d * 100).toFixed(0)}%</span>
                <span style={{ color: "var(--away)" }}>
                    {m.away_name.split(" ").pop()} win {(aw * 100).toFixed(0)}%
                </span>
            </div>
        </div>
    )
}