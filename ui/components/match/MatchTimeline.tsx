"use client"
// components/match/MatchTimeline.tsx
// Horizontal dot timeline showing key events at their minute markers.

import type { MatchState } from "@/types/match"

const DOT_COLOR: Record<string, string> = {
    goal: "var(--c-goal)",
    own_goal: "var(--c-own-goal)",
    penalty_goal: "var(--c-goal)",
    yellow: "var(--c-yellow)",
    red: "var(--c-red)",
    yellow_red: "var(--c-red)",
    substitution: "var(--c-sub)",
    var: "var(--c-var)",
}

interface Props { state: MatchState }

export function MatchTimeline({ state }: Props) {
    const maxMin = Math.max(90, state.elapsed ?? 90)
    const keyEvs = state.events.filter(ev =>
        ["goal", "own_goal", "penalty_goal", "yellow", "red", "yellow_red", "substitution"].includes(ev.type)
    )

    return (
        <div className="timeline-wrap">
            <h3 className="panel-title">Timeline</h3>
            <div className="timeline">
                {/* HT line */}
                <div className="timeline-ht" style={{ left: `${(45 / maxMin) * 100}%` }}>
                    <span className="ht-label">HT</span>
                </div>

                {/* progress line */}
                <div className="timeline-track">
                    <div
                        className="timeline-progress"
                        style={{ width: `${Math.min(100, ((state.elapsed ?? 0) / maxMin) * 100)}%` }}
                    />
                </div>

                {/* event dots */}
                {keyEvs.map((ev, i) => {
                    const pct = Math.min(99, (ev.elapsed / maxMin) * 100)
                    const color = DOT_COLOR[ev.type] ?? "var(--c-muted)"
                    const isHome = ev.team_name === state.home_name
                    const min = ev.extra ? `${ev.elapsed}+${ev.extra}'` : `${ev.elapsed}'`

                    return (
                        <div
                            key={i}
                            className={`tl-dot ${isHome ? "home" : "away"}`}
                            style={{ left: `${pct}%`, "--dot-color": color } as React.CSSProperties}
                            title={`${min} — ${ev.player_name ?? ev.team_name}`}
                        >
                            <div className="dot" />
                            <span className="dot-min">{min}</span>
                        </div>
                    )
                })}

                {/* minute markers */}
                <div className="tl-markers">
                    {[0, 15, 30, 45, 60, 75, 90].map(m => (
                        <span
                            key={m}
                            className="tl-marker"
                            style={{ left: `${(m / maxMin) * 100}%` }}
                        >
                            {m}'
                        </span>
                    ))}
                </div>
            </div>
        </div>
    )
}