"use client"
// components/match/MatchAITabs.tsx — no grid, each component full width

import { useState, useEffect } from "react"
import type { MatchState } from "@/types/match"
import { TacticalCard } from "@/components/match/TacticalCard"
import { LineupCard } from "@/components/match/LineupCard"
import { CounterfactualPanel } from "@/components/match/CounterfactualPanel"
import { PreMatchBriefingCard } from "@/components/match/PreMatchBriefingCard"

interface Props { state: MatchState; fixtureId: string }

const TABS = [
    { id: "tactical", icon: "📐", label: "Tactical" },
    { id: "counterfactual", icon: "🔀", label: "What If?" },
    { id: "briefing", icon: "📋", label: "Briefing" },
]

export function MatchAITabs({ state, fixtureId }: Props) {
    const [active, setActive] = useState("tactical")

    useEffect(() => {
        const s = localStorage.getItem("match-ai-tab")
        if (s) setActive(s)
    }, [])

    const switchTab = (id: string) => {
        setActive(id)
        localStorage.setItem("match-ai-tab", id)
    }

    return (
        <div className="ai-tabs">
            <div className="ai-tabs-nav">
                {TABS.map(t => (
                    <button key={t.id}
                        className={`ai-tab-btn${active === t.id ? " active" : ""}`}
                        onClick={() => switchTab(t.id)}>
                        <span className="ai-tab-icon">{t.icon}</span>
                        {t.label}
                    </button>
                ))}
            </div>

            <div className="ai-tabs-body">

                {active === "tactical" && (
                    <>
                        {/* Tactical fingerprint — full width */}
                        <TacticalCard state={state} fixtureId={fixtureId} />

                        {/* Divider + label */}
                        <div style={{
                            borderTop: "1px solid var(--border-bright)",
                            padding: "10px 14px 8px",
                            borderBottom: "1px solid var(--border)",
                            display: "flex", alignItems: "center", gap: 6,
                        }}>
                            <div style={{
                                width: 3, height: 10, background: "var(--home)",
                                borderRadius: 2, flexShrink: 0
                            }} />
                            <span style={{
                                fontFamily: "var(--font-mono)", fontSize: ".6rem",
                                textTransform: "none", letterSpacing: "normal",
                                color: "var(--text-3)"
                            }}>
                                Formations
                            </span>
                        </div>

                        {/* Lineup — full width, untouched */}
                        <LineupCard
                            fixtureId={fixtureId}
                            homeTeam={state.home_name}
                            awayTeam={state.away_name}
                        />
                    </>
                )}

                {active === "counterfactual" && (
                    <CounterfactualPanel fixtureId={fixtureId} />
                )}

                {active === "briefing" && (
                    <PreMatchBriefingCard fixtureId={fixtureId} statusShort={state.status_short} />
                )}

            </div>
        </div>
    )
}