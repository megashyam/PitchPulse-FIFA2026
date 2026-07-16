"use client"


import { useState, useEffect } from "react"
import type { MatchState } from "@/types/match"
import { TacticalCard } from "@/components/match/TacticalCard"
import { LineupCard } from "@/components/match/LineupCard"
import { CounterfactualPanel } from "@/components/match/CounterfactualPanel"
import { PreMatchBriefingCard } from "@/components/match/PreMatchBriefingCard"
import { IntelFeed } from "@/components/match/IntelFeed"
import { TeamFormCard } from "@/components/match/TeamFormCard"
import { MomentumBar } from "@/components/match/MomentumBar"
import { StadiumCard } from "@/components/match/StadiumCard"
import { RefereeCard } from "@/components/match/RefereeCard"
import { BracketProbChart } from "@/components/match/BracketProbChart"
import { NarrativeCarousel } from "@/components/match/NarrativeCarousel"

interface Props { state: MatchState; fixtureId: string }

const TABS = [
    { id: "tactical", label: "Tactical" },
    { id: "counterfactual", label: "What If?" },
    { id: "briefing", label: "Briefing" },
    { id: "live", label: "Live" },
    { id: "predictor", label: "Predictor" },
    { id: "narrative", label: "Narrative" },
]

export function UnifiedTabs({ state, fixtureId }: Props) {
    const [active, setActive] = useState("tactical")

    useEffect(() => {
        const s = localStorage.getItem("match-unified-tab")
        if (s) setActive(s)
    }, [])

    const switchTab = (id: string) => {
        setActive(id)
        localStorage.setItem("match-unified-tab", id)
    }

    return (
        <div className="mp-tabs">
            <div className="mp-tabs-nav">
                {TABS.map(t => (
                    <button key={t.id}
                        className={`ai-tab-btn${active === t.id ? " active" : ""}`}
                        onClick={() => switchTab(t.id)}>
                        {t.label}
                    </button>
                ))}
            </div>

            <div className="mp-tabs-body">

                {active === "tactical" && (
                    <>
                        <TacticalCard state={state} fixtureId={fixtureId} />
                        <div className="mp-section-label">
                            <span className="mp-section-label-bar" />
                            <span>Formations</span>
                        </div>
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
                    <PreMatchBriefingCard fixtureId={fixtureId} />
                )}

                {active === "live" && (
                    <div className="mp-tabs-stack">

                        <IntelFeed fixtureId={fixtureId} statusShort={state.status_short} />
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
