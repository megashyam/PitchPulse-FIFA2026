"use client"
// components/match/BracketProbChart.tsx
// Top 8 win distribution (hover reveals probabilities) plus market odds for
// this fixture from GET /matches/{fixture_id}/prediction.

import { useEffect, useMemo, useState } from "react"
import { usePredictStream } from "@/hooks/usePredictStream"
import { Flag } from "@/components/Flag"
import { STAGES, STAGE_LABELS } from "@/types/predict"
import type { Stage } from "@/types/predict"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

const COMPACT_LABELS: Record<Stage, string> = {
    r32: "R32", r16: "R16", qf: "QF", sf: "SF", final: "FINAL", champion: "CHAMPION",
}
const STAGE_COLORS: Record<Stage, string> = {
    r32: "#4f86f7", r16: "#4f86f7", qf: "#9b6cf7",
    sf: "#f59e0b", final: "#f59e0b", champion: "#10d9a0",
}

interface MatchOdds {
    fixture_id: number
    home_name: string
    away_name: string
    match_odds: {
        home_win: { p: number; ci_lo: number; ci_hi: number }
        draw: { p: number; ci_lo: number; ci_hi: number }
        away_win: { p: number; ci_lo: number; ci_hi: number }
    }
    source: "market_odds" | "elo"
}

interface Props { defaultTeams?: string[]; fixtureId: string }

export function BracketProbChart({ defaultTeams = [], fixtureId }: Props) {
    const { prediction, status, isLoading, error, triggerSim } = usePredictStream()
    const [selected, setSelected] = useState<string | null>(null)
    const [hovered, setHovered] = useState<string | null>(null)
    const [matchOdds, setMatchOdds] = useState<MatchOdds | null>(null)
    const [oddsError, setOddsError] = useState(false)

    const isRunning = status?.status === "running" || isLoading
    const hasData = !!prediction

    // Top 8 teams by champion probability — this IS the Win Distribution chart
    const top8 = useMemo(() =>
        hasData ? [...prediction!.teams].sort((a, b) => b.champion.p - a.champion.p).slice(0, 8) : []
        , [prediction])

    const maxP = top8[0]?.champion.p ?? 1

    useEffect(() => {
        if (!hasData || selected) return
        const preferred = defaultTeams.find(n => top8.some(t => t.name === n))
        setSelected(preferred ?? top8[0]?.name ?? null)
    }, [hasData, top8])

    // Fetch odds for THIS match only
    useEffect(() => {
        if (!fixtureId) return
        fetch(`${API}/matches/${fixtureId}/prediction`)
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(setMatchOdds)
            .catch(() => setOddsError(true))
    }, [fixtureId])

    const activeName = hovered ?? selected
    const activeTeam = top8.find(t => t.name === activeName) ?? null

    if (!hasData && !isRunning) return (
        <div style={{ padding: "24px 16px", textAlign: "center" }}>
            <div style={{ fontSize: "1.4rem", marginBottom: 8, opacity: .5 }}>🎯</div>
            <div style={{ fontSize: ".8rem", fontWeight: 600, color: "var(--text-1)", marginBottom: 6 }}>
                No simulation results yet
            </div>
            <div style={{ fontSize: ".72rem", color: "var(--text-3)", marginBottom: 12, lineHeight: 1.5 }}>
                Run the Monte Carlo simulation to see bracket probabilities.
            </div>
            <button onClick={() => triggerSim(50_000)} className="nar-force-btn">Run 50k Sim</button>
            {error && <div style={{ marginTop: 8, fontSize: ".7rem", color: "var(--c-red)" }}>{error}</div>}
        </div>
    )

    if (isRunning && !hasData) return (
        <div style={{ padding: "24px 16px", textAlign: "center" }}>
            <span className="live-dot" style={{ width: 10, height: 10 }} />
            <div style={{ fontSize: ".8rem", fontWeight: 600, color: "var(--text-1)", margin: "10px 0 4px" }}>
                Simulating 50,000 tournaments…
            </div>
            <div style={{ fontSize: ".72rem", color: "var(--text-3)" }}>~8 seconds on CPU</div>
        </div>
    )

    return (
        <div>

            {/* ── Win distribution — top 8, hover reveals bracket path ── */}
            <div style={{ padding: "12px 14px 4px" }}>
                <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    marginBottom: 10,
                }}>
                    <span style={{
                        fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none",
                        letterSpacing: "normal", color: "var(--text-3)",
                    }}>
                        Win Distribution — Top 8
                    </span>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".54rem", color: "var(--text-3)" }}>
                        MC output · 95% CI
                    </span>
                </div>

                <div style={{ display: "flex", alignItems: "flex-end", gap: 6, height: 130 }}>
                    {top8.map((team, i) => {
                        const p = team.champion.p
                        const barH = Math.max(6, (p / maxP) * 110)
                        const isSel = team.name === selected
                        const isHov = team.name === hovered
                        const col = i === 0 ? "var(--accent)"
                            : i < 3 ? "var(--home)"
                                : "#9b6cf7"
                        return (
                            <div
                                key={team.name}
                                onClick={() => setSelected(team.name)}
                                onMouseEnter={() => setHovered(team.name)}
                                onMouseLeave={() => setHovered(null)}
                                style={{
                                    flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
                                    gap: 4, cursor: "pointer", position: "relative",
                                }}
                            >
                                {/* Hover tooltip — champion % */}
                                {isHov && (
                                    <div style={{
                                        position: "absolute", bottom: barH + 22, left: "50%", transform: "translateX(-50%)",
                                        background: "var(--bg-4)", border: "1px solid var(--border-bright)",
                                        borderRadius: 6, padding: "3px 8px", whiteSpace: "nowrap",
                                        fontFamily: "var(--font-mono)", fontSize: ".68rem", fontWeight: 700,
                                        color: "var(--text-1)", zIndex: 5, boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
                                    }}>
                                        {(p * 100).toFixed(1)}%
                                    </div>
                                )}
                                <div style={{
                                    width: "100%", height: barH, borderRadius: "4px 4px 0 0",
                                    background: col,
                                    outline: isSel ? "2px solid var(--text-1)" : "none",
                                    outlineOffset: 1,
                                    opacity: isHov || isSel ? 1 : 0.85,
                                    transition: "opacity .15s, height .3s ease",
                                }} />
                                <Flag team={team.name} size="sm" />
                                <span style={{ fontFamily: "var(--font-mono)", fontSize: ".6rem", color: "var(--text-2)", fontWeight: 600 }}>
                                    {team.name.slice(0, 3).toUpperCase()}
                                </span>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Selected/hovered team's full bracket path */}
            {activeTeam && (
                <div style={{ padding: "14px 14px 10px", borderTop: "1px solid var(--border)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
                        <Flag team={activeTeam.name} size="sm" />
                        <span style={{ fontSize: ".78rem", fontWeight: 700, color: "var(--text-1)" }}>{activeTeam.name}</span>
                    </div>
                    {STAGES.map(stage => {
                        const pct = Math.round(activeTeam[stage].p * 100)
                        return (
                            <div key={stage} style={{
                                display: "grid", gridTemplateColumns: "62px 1fr 40px",
                                alignItems: "center", gap: 10, marginBottom: 8,
                            }}>
                                <span style={{ fontFamily: "var(--font-mono)", fontSize: ".72rem", color: "var(--text-2)" }}>
                                    {COMPACT_LABELS[stage]}
                                </span>
                                <div style={{ height: 6, background: "var(--bg-4)", borderRadius: 3, overflow: "hidden" }}>
                                    <div style={{
                                        width: `${pct}%`, height: "100%", borderRadius: 3,
                                        background: STAGE_COLORS[stage], transition: "width .4s ease",
                                    }} />
                                </div>
                                <span style={{ fontFamily: "var(--font-mono)", fontSize: ".76rem", fontWeight: 700, color: "var(--text-1)", textAlign: "right" }}>
                                    {pct}%
                                </span>
                            </div>
                        )
                    })}
                </div>
            )}

            {/* ── Per-match market odds — THIS fixture only ── */}
            <div style={{ padding: "12px 14px 14px", borderTop: "1px solid var(--border-bright)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                    <span style={{ width: 3, height: 10, background: "var(--away)", borderRadius: 2, flexShrink: 0 }} />
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)" }}>
                        Market Odds — Per Match
                    </span>
                </div>

                {oddsError ? (
                    <div style={{ fontSize: ".72rem", color: "var(--text-3)", fontStyle: "italic" }}>
                        Odds not available for this fixture
                    </div>
                ) : !matchOdds ? (
                    <div style={{ fontSize: ".72rem", color: "var(--text-3)" }}>Loading…</div>
                ) : (
                    <>
                        <div style={{
                            display: "inline-flex", alignItems: "center", gap: 6,
                            padding: "3px 9px", borderRadius: 12, marginBottom: 8,
                            background: matchOdds.source === "market_odds" ? "rgba(240,84,84,.1)" : "var(--glass-bg-inner)",
                            border: `1px solid ${matchOdds.source === "market_odds" ? "rgba(240,84,84,.3)" : "var(--glass-border-inner)"}`,
                        }}>
                            <span style={{
                                fontFamily: "var(--font-mono)", fontSize: ".6rem", fontWeight: 700,
                                color: matchOdds.source === "market_odds" ? "#f05454" : "var(--text-3)",
                            }}>
                                {matchOdds.source === "market_odds" ? "THE ODDS API · MARKET ODDS" : "ELO PRIOR"}
                            </span>
                        </div>
                        <div style={{ fontSize: ".7rem", color: "var(--text-3)", marginBottom: 10 }}>
                            Per-match win probabilities (input to MC sim)
                        </div>

                        <div style={{ background: "var(--glass-bg-inner)", border: "1px solid var(--glass-border-inner)", borderRadius: "var(--r-md)", padding: 16 }}>
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                                <span style={{ fontSize: ".8rem", fontWeight: 700 }}>
                                    <span style={{ color: "var(--home)" }}>{matchOdds.home_name}</span>
                                    {" vs "}
                                    <span style={{ color: "var(--away)" }}>{matchOdds.away_name}</span>
                                </span>
                            </div>
                            <div style={{ display: "flex", height: 26, borderRadius: 5, overflow: "hidden" }}>
                                <div style={{
                                    width: `${(matchOdds.match_odds.home_win.p * 100).toFixed(0)}%`,
                                    background: "var(--home)", display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: ".68rem", fontWeight: 700, color: "#fff",
                                }}>
                                    {(matchOdds.match_odds.home_win.p * 100).toFixed(0)}%
                                </div>
                                <div style={{
                                    width: `${(matchOdds.match_odds.draw.p * 100).toFixed(0)}%`,
                                    background: "var(--bg-4)", display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: ".68rem", fontWeight: 700, color: "var(--text-2)",
                                }}>
                                    {(matchOdds.match_odds.draw.p * 100).toFixed(0)}%
                                </div>
                                <div style={{
                                    width: `${(matchOdds.match_odds.away_win.p * 100).toFixed(0)}%`,
                                    background: "var(--away)", display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: ".68rem", fontWeight: 700, color: "#fff",
                                }}>
                                    {(matchOdds.match_odds.away_win.p * 100).toFixed(0)}%
                                </div>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: ".64rem" }}>
                                <span style={{ color: "var(--home)" }}>{matchOdds.home_name} win {(matchOdds.match_odds.home_win.p * 100).toFixed(0)}%</span>
                                <span style={{ color: "var(--text-3)" }}>Draw {(matchOdds.match_odds.draw.p * 100).toFixed(0)}%</span>
                                <span style={{ color: "var(--away)" }}>{matchOdds.away_name} win {(matchOdds.match_odds.away_win.p * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    </>
                )}
            </div>

        </div>
    )
}