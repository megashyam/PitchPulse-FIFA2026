"use client"
// components/match/IntelFeed.tsx — match intelligence activity feed
//
// Driven by the hook's `phase`: a spinner only while waiting on live
// analysis, and a distinct message for matches that aren't live.

import { useIntelStream } from "@/hooks/useIntelStream"
import type { IntelEntry } from "@/hooks/useIntelStream"
import { viaLabel } from "@/lib/via"
import { useState, useEffect } from "react"

const TYPE_META: Record<string, { icon: string; label: string }> = {
    event_reaction: { icon: "⚽", label: "Event" },
    xg_divergence: { icon: "📊", label: "xG" },
    tactical: { icon: "🧠", label: "Tactical" },
}

export function IntelFeed({ fixtureId, statusShort }: { fixtureId: string; statusShort?: string }) {
    const { entries, phase } = useIntelStream(fixtureId, statusShort)
    const [open, setOpen] = useState(true)
    useEffect(() => { const s = localStorage.getItem("card:intel"); if (s) setOpen(s === "open") }, [])
    const toggle = () => { const n = !open; setOpen(n); localStorage.setItem("card:intel", n ? "open" : "closed") }

    return (
        <div className="intel-wrap-v2">
            <div className="intel-header-v2">
                <div className="intel-header-main" onClick={toggle} style={{ cursor: "pointer" }}>
                    <span className="intel-header-title">Live Intelligence</span>
                    <span className="intel-header-sub">AI analysis — goals, tactics, xG shifts</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className="intel-header-badge">
                        <span className="live-dot" style={{ width: 5, height: 5 }} />
                        Mistral · 5 min
                    </span>
                    <button className="card-shell-chevron" onClick={toggle}>{open ? "▲" : "▼"}</button>
                </div>
            </div>

            {open && (
                <div className="intel-entries-v2" style={{ animation: "card-expand .18s ease" }}>
                    {phase === "loading" ? (
                        <IntelLoadingState text="Loading analysis…" />
                    ) : phase === "streaming" ? (
                        entries.map((e, i) => <IntelCard key={`${e.minute}-${i}`} entry={e} />)
                    ) : phase === "idle_notlive" ? (
                        <p className="intel-waiting-v2">
                            Kickoff hasn't happened yet — analysis begins once the match goes live.
                        </p>
                    ) : (
                        <p className="intel-waiting-v2">
                            No AI narration for this match yet — appears at minute 5 or after a goal.
                        </p>
                    )}
                </div>
            )}
        </div>
    )
}

function IntelLoadingState({ text }: { text: string }) {
    return (
        <div className="intel-loading-row">
            <span className="intel-loading-spinner" aria-hidden="true" />
            <span className="intel-waiting-v2" style={{ margin: 0 }}>{text}</span>
            <style jsx>{`
                .intel-loading-row {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 4px 0;
                }
                .intel-loading-spinner {
                    flex-shrink: 0;
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    border: 2px solid var(--border);
                    border-top-color: var(--c-ai, #a78bfa);
                    animation: intel-spin 0.8s linear infinite;
                }
                @keyframes intel-spin {
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    )
}

function IntelCard({ entry }: { entry: IntelEntry }) {
    const meta = TYPE_META[entry.narration_type] ?? TYPE_META.tactical
    const ts = new Date(entry.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

    return (
        <div className="intel-entry-v3">
            <div className="intel-icon-circle" data-type={entry.narration_type}>
                {meta.icon}
            </div>
            <div className="intel-entry-body">
                <div className="intel-entry-row1">
                    <span className="intel-entry-minute">{entry.minute}'</span>
                    <span className="intel-entry-type-badge" data-type={entry.narration_type}>
                        {meta.label}
                    </span>
                </div>
                <p className="intel-entry-text">{entry.narrative}</p>
                <div className="intel-entry-footer">
                    <span className="intel-entry-via">
                        {viaLabel(entry.via)}
                        {entry.rag_docs_used > 0 && ` · ${entry.rag_docs_used} sources`}
                    </span>
                    <span className="intel-entry-timestamp">{ts}</span>
                </div>
            </div>
        </div>
    )
}