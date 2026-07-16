"use client"
// components/match/LiveProbCard.tsx
// Live win probability. Now backed by GET /matches/{id}/live-prob — the
// same in-play model (ml/in_play.py) the counterfactual agent conditions
// on — polled every 30s. Falls back to the local calcProbs() heuristic only
// if that fetch fails, so the card never goes blank on a transient error.
//
// Fix (audit finding): this component previously computed its OWN
// probability from score + momentum + elapsed time — a fourth, independent
// probability model alongside the backend's Elo/Betfair prior, in-play
// model, and counterfactual conditioning, with no guarantee any of them
// agreed. calcProbs() is kept only as an offline/error fallback so the UI
// degrades gracefully instead of going blank.

import { useEffect, useState } from "react"
import { useMomentumStream } from "@/hooks/useMomentumStream"
import type { MatchState } from "@/types/match"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const POLL_INTERVAL = 30_000

interface Props { state: MatchState; fixtureId: string }

interface LiveProbResponse {
    fixture_id: number
    elapsed: number
    status_short: string
    home_win: number
    draw: number
    away_win: number
    pre_match_source: string
}

// Fallback-only heuristic — used if /live-prob is unreachable.
function calcProbs(
    homeGoals: number, awayGoals: number,
    homeMom: number, elapsed: number, status: string
): { home: number; draw: number; away: number } {
    const FT = ["FT", "AET", "PEN"].includes(status)
    if (FT) {
        if (homeGoals > awayGoals) return { home: 1, draw: 0, away: 0 }
        if (awayGoals > homeGoals) return { home: 0, draw: 0, away: 1 }
        return { home: 0, draw: 1, away: 0 }
    }

    const NS = status === "NS"
    if (NS) return { home: 0.45, draw: 0.27, away: 0.28 }

    const maxMin = 90
    const timeElapsed = Math.min(1, (elapsed ?? 45) / maxMin)
    const certainty = timeElapsed ** 0.7

    const diff = homeGoals - awayGoals
    const momAdv = (homeMom - 0.5) * 0.3

    let pH = 0.45 + momAdv
    let pD = 0.27
    let pA = 0.28 - momAdv

    if (diff > 0) {
        pH += certainty * (0.35 * Math.min(diff, 2))
        pD -= certainty * 0.12
        pA -= certainty * (0.23 * Math.min(diff, 2))
    } else if (diff < 0) {
        pA += certainty * (0.35 * Math.min(-diff, 2))
        pD -= certainty * 0.12
        pH -= certainty * (0.23 * Math.min(-diff, 2))
    }

    const total = pH + pD + pA
    return {
        home: Math.max(0.02, pH / total),
        draw: Math.max(0.02, pD / total),
        away: Math.max(0.02, pA / total),
    }
}

export function LiveProbCard({ state, fixtureId }: Props) {
    const { momentum } = useMomentumStream(fixtureId)
    const [live, setLive] = useState<LiveProbResponse | null>(null)
    const [usingFallback, setUsingFallback] = useState(false)

    useEffect(() => {
        let cancelled = false

        async function load() {
            try {
                const r = await fetch(`${API}/matches/${fixtureId}/live-prob`)
                if (!r.ok) throw new Error(`${r.status}`)
                const data: LiveProbResponse = await r.json()
                if (!cancelled) {
                    setLive(data)
                    setUsingFallback(false)
                }
            } catch {
                if (!cancelled) setUsingFallback(true)
            }
        }

        load()
        const t = setInterval(load, POLL_INTERVAL)
        return () => { cancelled = true; clearInterval(t) }
    }, [fixtureId])

    const homeMom = momentum?.home.momentum_score ?? 0.5
    const elapsed = state.elapsed ?? 0

    const probs = live && !usingFallback
        ? { home: live.home_win, draw: live.draw, away: live.away_win }
        : calcProbs(state.home_score, state.away_score, homeMom, elapsed, state.status_short)

    const isLive = ["1H", "HT", "2H", "ET", "P", "LIVE"].includes(state.status_short)
    const isFT = ["FT", "AET", "PEN"].includes(state.status_short)

    const hn = state.home_name.length > 10
        ? state.home_name.split(" ").pop()!
        : state.home_name
    const an = state.away_name.length > 10
        ? state.away_name.split(" ").pop()!
        : state.away_name

    const homeW = Math.round(probs.home * 100)
    const drawW = Math.round(probs.draw * 100)
    const awayW = 100 - homeW - drawW

    const modelNote = usingFallback || !live
        ? `Score + momentum heuristic${momentum ? " · EWMA" : ""} (offline fallback)`
        : `In-play model · ${live.pre_match_source} prior`

    return (
        <div>
            <div className="stats-section-header">
                Win Probability
                <span style={{
                    marginLeft: "auto", fontSize: ".52rem", color: "var(--text-3)",
                    textTransform: "none", letterSpacing: 0,
                }}>
                    {isFT ? "Final" : isLive ? `${elapsed}'` : "Pre-match"}
                </span>
            </div>

            <div style={{ padding: "3px 14px 5px" }}>


                <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 4, marginBottom: 3, alignItems: "baseline" }}>
                    <span style={{ fontSize: ".68rem", fontWeight: 700, color: "var(--home)" }}>
                        {hn} <span style={{ fontSize: ".76rem" }}>{homeW}%</span>
                    </span>
                    <span style={{ fontSize: ".56rem", color: "var(--text-3)", textAlign: "center" }}>{drawW}%</span>
                    <span style={{ fontSize: ".68rem", fontWeight: 700, color: "var(--away)", textAlign: "right" }}>
                        <span style={{ fontSize: ".76rem" }}>{awayW}%</span> {an}
                    </span>
                </div>


                <div style={{ height: 4, borderRadius: 2, overflow: "hidden", display: "flex", gap: 1 }}>
                    <div style={{ width: `${homeW}%`, background: "var(--home)" }} />
                    <div style={{ width: `${drawW}%`, background: "var(--text-3)" }} />
                    <div style={{ width: `${awayW}%`, background: "var(--away)" }} />
                </div>

                {/* Model note — kept, but as small as possible; drops entirely
                    on narrow renders rather than getting clipped mid-word. */}
                <div style={{
                    marginTop: 3, fontFamily: "var(--font-mono)", fontSize: ".5rem",
                    color: "var(--text-3)", whiteSpace: "nowrap", overflow: "hidden",
                    textOverflow: "ellipsis",
                }}>
                    {modelNote}
                </div>

            </div>
        </div>
    )
}
