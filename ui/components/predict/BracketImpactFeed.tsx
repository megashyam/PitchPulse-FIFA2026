"use client"
// components/predict/BracketImpactFeed.tsx
// Same stale result guard as CounterfactualPanel.
// Fetches current match elapsed to filter out results from previous cycles.

import { useEffect, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface CfEntry {
    minute: number
    event_type: string
    event_team: string
    path_shift_pct: number
    top_changes: { team: string; before: number; after: number; delta: number }[]
    narrative: string
    n_sims: number
    updated_at: string
}

const EV_LABELS: Record<string, string> = {
    goal: "goal", own_goal: "own goal", penalty_goal: "penalty",
    red: "red card", yellow_red: "second yellow",
}

function useBracketImpact() {
    const [entries, setEntries] = useState<CfEntry[]>([])
    const [matchName, setMatchName] = useState("")
    const [matchElapsed, setMatchElapsed] = useState(0)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        let mounted = true
        async function load() {
            try {
                const r1 = await fetch(`${API}/matches`)
                if (!r1.ok) return
                const { fixtures: ids } = await r1.json()
                if (!ids?.length) return
                const fid = ids[0]

                const state = await fetch(`${API}/matches/${fid}`).then(r => r.json())
                if (mounted) {
                    setMatchName(`${state.home_name} vs ${state.away_name}`)
                    setMatchElapsed(state.elapsed ?? 0)
                }

                const r3 = await fetch(`${API}/matches/${fid}/counterfactual/feed`)
                if (r3.ok) {
                    const data = await r3.json()
                    if (mounted) setEntries(data.entries ?? [])
                }
            } catch { }
            if (mounted) setLoading(false)
        }
        load()
        const t = setInterval(load, 30_000)
        return () => { mounted = false; clearInterval(t) }
    }, [])

    return { entries, matchName, matchElapsed, loading }
}

export function BracketImpactFeed() {
    const { entries, matchName, matchElapsed, loading } = useBracketImpact()

    // Filter out stale results from previous replay cycles
    const freshEntries = entries.filter(e => e.minute <= matchElapsed + 5)

    return (
        <div className="pred-bracket">
            <div className="pred-bracket-hdr">
                <div className="pred-bracket-hdr-left">
                    <span className="src src-novel">Novel · counterfactual narrator</span>
                    {matchName && <span className="pred-bracket-match">{matchName}</span>}
                </div>
                <span className="pred-bracket-meta">MC divergence score → LLM</span>
            </div>

            <div className="pred-bracket-section">Bracket impact — key events</div>

            {loading && <p className="pred-bracket-empty">Loading…</p>}

            {!loading && freshEntries.length === 0 && (
                <p className="pred-bracket-empty">
                    {matchElapsed < 5
                        ? "Waiting for the first key event…"
                        : "No bracket impact data yet. Run "}
                    {matchElapsed >= 5 && (
                        <code style={{ color: "var(--amber)", fontSize: ".7rem" }}>
                            /matches/&#123;id&#125;/counterfactual/trigger
                        </code>
                    )}
                </p>
            )}

            {freshEntries.length > 0 && (
                <>
                    <div className="pred-bracket-events">
                        {freshEntries.map((e, i) => <EventRow key={i} e={e} />)}
                    </div>
                    {freshEntries[0]?.narrative && (
                        <div className="pred-bracket-narrative-wrap">
                            <span className="pred-bracket-narrative-meta">
                                post-event narrative
                            </span>
                            <p className="pred-bracket-narrative">"{freshEntries[0].narrative}"</p>
                        </div>
                    )}
                    <div className="pred-bracket-infra">
                        <span>Redis cache · 24h TTL</span>
                        <span>FastAPI REST endpoint</span>
                        <span>Sim runtime ~8s · numpy</span>
                        <span>Trigger: post-event MC re-run</span>
                    </div>
                </>
            )}
        </div>
    )
}

function EventRow({ e }: { e: CfEntry }) {
    const evLabel = EV_LABELS[e.event_type] ?? e.event_type
    const shift = `+${(e.path_shift_pct * 100).toFixed(0)}%`
    const top = e.top_changes[0]
    const isHigh = e.path_shift_pct >= 0.5

    let summary = `${(e.path_shift_pct * 100).toFixed(0)}% of sim trajectories shifted.`
    if (top) {
        summary += ` ${top.team} ${top.delta > 0 ? "▲" : "▼"}: `
            + `${(top.before * 100).toFixed(0)}% → ${(top.after * 100).toFixed(0)}%`
    }

    return (
        <div className={`pred-bracket-event${isHigh ? " high" : ""}`}>
            <div className="pred-bracket-event-hdr">
                <span className="pred-bracket-event-name">
                    {e.event_team} {evLabel} · {e.minute}'
                </span>
                <span className="pred-bracket-shift"
                    style={{ color: isHigh ? "var(--amber)" : "var(--text-2)" }}>
                    {shift}
                </span>
            </div>
            <p className="pred-bracket-event-summary">{summary}</p>
        </div>
    )
}