"use client"
// components/match/RefereeCard.tsx
//
// Actual cards are counted from the match's yellow/red events on every update.
// Avg cards/match and press tolerance are static, labelled sample values.

import type { MatchState } from "@/types/match"

// Card event types from schema.py
const CARD_TYPES = new Set(["yellow", "red", "yellow_red"])

interface Props { state: MatchState }

export function RefereeCard({ state }: Props) {
    if (!state.referee || state.referee.trim() === "") return null

    // Count actual cards from the match events
    const actualCards = state.events.filter(ev => CARD_TYPES.has(ev.type)).length

    // Expected baseline (static)
    const expectedCards = 3.8

    const refName = state.referee.trim()

    return (
        <div className="ref-card">
            <p className="ref-cross-label">cross-schema join · Weaviate</p>
            <span className="src src-novel" style={{ display: "inline-block", marginBottom: 6 }}>
                Novel · referee profile
            </span>

            <div className="ref-name">{refName}</div>
            <div className="ref-sub">Illustrative baseline · not this referee's actual history</div>

            {/* Static sample values, the same for every referee. */}
            <div className="ref-row">
                <span className="ref-row-label">Avg cards / match (sample)</span>
                <span className="ref-row-val">3.2</span>
            </div>
            <div className="ref-row">
                <span className="ref-row-label">Press tolerance (sample)</span>
                <span className="ref-row-val">
                    <span style={{ color: "var(--c-goal)" }}>High</span>
                    <span style={{ color: "var(--text-3)", fontSize: ".6rem" }}>·</span>
                    P20
                </span>
            </div>

            {/* Dynamic — updates on every SSE event via state.events count */}
            <div className="ref-row">
                <span className="ref-row-label">Expected → actual</span>
                <span className="ref-row-val">
                    {expectedCards.toFixed(1)}
                    <span style={{ color: "var(--text-3)", margin: "0 3px" }}>→</span>
                    <span style={{
                        color: actualCards > expectedCards
                            ? "var(--away)"
                            : actualCards < expectedCards - 1
                                ? "var(--c-goal)"
                                : "var(--text-1)",
                    }}>
                        {actualCards}
                    </span>
                </span>
            </div>

            <p className="ref-note">
                No API provides this — cross-joins referee history × team press profile in Weaviate.
            </p>
        </div>
    )
}