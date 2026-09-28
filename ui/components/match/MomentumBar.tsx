"use client"
// components/match/MomentumBar.tsx — hero percentage numbers + 3-cell grid

import { useMomentumStream } from "@/hooks/useMomentumStream"
import type { MomentumSnapshot } from "@/hooks/useMomentumStream"
import { useState, useEffect } from "react"

export function MomentumBar({ fixtureId }: { fixtureId: string }) {
    const { momentum } = useMomentumStream(fixtureId)
    if (!momentum) return null
    return <Display m={momentum} />
}

function Display({ m }: { m: MomentumSnapshot }) {
    const homePct = Math.round(m.home.momentum_score * 100)
    const awayPct = 100 - homePct
    const homeGoalPct = (m.home.goal_prob_5min * 100).toFixed(1)
    const awayGoalPct = (m.away.goal_prob_5min * 100).toFixed(1)

    const [open, setOpen] = useState(true)
    useEffect(() => { const s = localStorage.getItem("card:momentum"); if (s) setOpen(s === "open") }, [])
    const toggle = () => { const n = !open; setOpen(n); localStorage.setItem("card:momentum", n ? "open" : "closed") }

    return (
        <div className="momentum-wrap">

            {/* Header */}
            <div className="momentum-header" style={{ cursor: "pointer" }} onClick={toggle}>
                <div>
                    <div className="momentum-title">Momentum</div>
                    <div className="momentum-title-sub">EWMA model — updates every 30s</div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className="momentum-live-badge">
                        <span className="live-dot" style={{ width: 5, height: 5 }} />
                        live
                    </span>
                    <button className="card-shell-chevron" onClick={e => { e.stopPropagation(); toggle() }}>
                        {open ? "▲" : "▼"}
                    </button>
                </div>
            </div>

            {open && (
                <div style={{ animation: "card-expand .18s ease" }}>

                    {/* Hero numbers */}
                    <div className="momentum-hero">
                        <div>
                            <div className="momentum-hero-pct home">
                                {homePct}%
                            </div>
                            <div className="momentum-hero-sub home">{homeGoalPct}% chance/5 min</div>
                        </div>

                        <div className="momentum-hero-center">
                            <div className="momentum-vs-label">vs</div>
                            <div className="momentum-hero-bar">
                                <div className="momentum-hero-bar-home" style={{ width: `${homePct}%` }} />
                                <div className="momentum-hero-bar-away" style={{ width: `${awayPct}%` }} />
                            </div>
                            <div className="momentum-vs-label">momentum</div>
                        </div>

                        <div style={{ textAlign: "right" }}>
                            <div className="momentum-hero-pct away">
                                {awayPct}%
                            </div>
                            <div className="momentum-hero-sub away">{awayGoalPct}% chance/5 min</div>
                        </div>
                    </div>

                    {/* 3-cell breakdown */}
                    <div className="momentum-grid">

                        <div className="momentum-grid-cell">
                            <span className="momentum-grid-label">Shots (15')</span>
                            <div className="momentum-grid-values">
                                <span className="momentum-grid-val home">{m.home.shots_15min}</span>
                                <span className="momentum-grid-divider">/</span>
                                <span className="momentum-grid-val away">{m.away.shots_15min}</span>
                            </div>
                        </div>

                        <div className="momentum-grid-cell">
                            <span className="momentum-grid-label">xG (15')</span>
                            <div className="momentum-grid-values">
                                <span className="momentum-grid-val home">{m.home.xg_15min.toFixed(2)}</span>
                                <span className="momentum-grid-divider">/</span>
                                <span className="momentum-grid-val away">{m.away.xg_15min.toFixed(2)}</span>
                            </div>
                        </div>

                        <div className="momentum-grid-cell">
                            <span className="momentum-grid-label">xG (match)</span>
                            <div className="momentum-grid-values">
                                <span className="momentum-grid-val home">{m.home.xg_total.toFixed(2)}</span>
                                <span className="momentum-grid-divider">/</span>
                                <span className="momentum-grid-val away">{m.away.xg_total.toFixed(2)}</span>
                            </div>
                        </div>

                    </div>

                </div>
            )}

        </div>
    )
}