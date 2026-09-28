"use client"
// components/match/RightColumnTabs.tsx

import { useState, useEffect } from "react"
import { IntelFeed } from "@/components/match/IntelFeed"
import { LiveProbCard } from "@/components/match/LiveProbCard"
import { XGCard } from "@/components/match/XGCard"
import { TeamFormCard } from "@/components/match/TeamFormCard"
import { MomentumBar } from "@/components/match/MomentumBar"
import { StadiumCard } from "@/components/match/StadiumCard"
import { RefereeCard } from "@/components/match/RefereeCard"
import { BracketProbChart } from "@/components/match/BracketProbChart"
import { NarrativeCarousel } from "@/components/match/NarrativeCarousel"
import type { MatchState } from "@/types/match"

interface Props { state: MatchState; fixtureId: string }

const TABS = [
    { id: "live", icon: "📡", label: "Live" },
    { id: "predictor", icon: "🏆", label: "Predictor" },
    { id: "narrative", icon: "🔥", label: "Narrative" },
]

export function RightColumnTabs({ state, fixtureId }: Props) {
    const [active, setActive] = useState("live")

    useEffect(() => {
        const s = localStorage.getItem("col3-tab")
        if (s) setActive(s)
    }, [])

    const switchTab = (id: string) => {
        setActive(id)
        localStorage.setItem("col3-tab", id)
    }

    return (
        <div className="ai-tabs">
            <div className="ai-tabs-nav">
                {TABS.map(t => (
                    <button
                        key={t.id}
                        className={`ai-tab-btn${active === t.id ? " active" : ""}`}
                        onClick={() => switchTab(t.id)}
                    >
                        <span className="ai-tab-icon">{t.icon}</span>
                        {t.label}
                    </button>
                ))}
            </div>

            <div className="ai-tabs-body">

                {active === "live" && (
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                        <IntelFeed fixtureId={fixtureId} statusShort={state.status_short} />
                        <LiveProbCard state={state} fixtureId={fixtureId} />
                        <XGCard state={state} />
                        <TeamFormCard
                            fixtureId={fixtureId}
                            homeTeam={state.home_name}
                            awayTeam={state.away_name}
                        />
                        <MomentumBar fixtureId={fixtureId} />
                        <StadiumCard venue={state.venue} round={state.round} />
                        {state.referee?.trim() && <RefereeCard state={state} />}
                    </div>
                )}

                {active === "predictor" && (
                    <BracketProbChart defaultTeams={[state.home_name, state.away_name]} fixtureId={fixtureId} />
                )}

                {active === "narrative" && (
                    <NarrativeCarousel homeTeam={state.home_name} awayTeam={state.away_name} />
                )}

            </div>
        </div>
    )
}