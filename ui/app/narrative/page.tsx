"use client"


import { useEffect, useState } from "react"
import { useNarrativeStream } from "@/hooks/useNarrativeStream"
import type { NarrativeSpike } from "@/hooks/useNarrativeStream"
import { CommentBubbles } from "@/components/narrative/CommentBubbles"
import { triggerHeaders } from "@/lib/api"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

const SRC: Record<string, { color: string; bg: string; icon: string; label: string; max: number; unit: string }> = {
    mastodon: { color: "#6364FF", bg: "rgba(99,100,255,.12)", icon: "🐘", label: "Mastodon", max: 50, unit: "posts/min" },
    bluesky: { color: "#4f86f7", bg: "rgba(79,134,247,.12)", icon: "🦋", label: "Bluesky", max: 40, unit: "mentions/min" },
    trends: { color: "#f59e0b", bg: "rgba(245,158,11,.12)", icon: "📈", label: "Trends", max: 100, unit: "index 0–100" },
    wikipedia: { color: "#10d9a0", bg: "rgba(16,217,160,.12)", icon: "📖", label: "Wikipedia", max: 10, unit: "edits/min" },
}
const SOURCES = ["mastodon", "bluesky", "trends", "wikipedia"] as const
const DEFAULT_CFG = SRC.mastodon

function cfgFor(src: string) {
    return SRC[src] ?? DEFAULT_CFG
}

const LIVE_STATUSES = new Set(["1H", "HT", "2H", "ET", "P"])

function fixtureBadge(spike: NarrativeSpike): { label: string; color: string; bg: string } | null {
    if (!spike.match_status) return null
    if (LIVE_STATUSES.has(spike.match_status)) {
        return { label: "LIVE", color: "var(--amber)", bg: "rgba(245,158,11,.14)" }
    }
    if (spike.match_status === "NS") {
        return { label: "UPCOMING", color: "var(--accent)", bg: "var(--accent-dim)" }
    }
    return null
}

async function forceTick(): Promise<string> {
    try {
        const r = await fetch(`${API}/narrative/trigger`, { headers: triggerHeaders() })
        const d = await r.json()
        if (d.status === "warming_up") return d.message ?? "Still warming up…"
        if (d.status === "no_spikes") return "No anomalies this tick"
        return `${d.spikes_detected} spike${d.spikes_detected !== 1 ? "s" : ""} detected and stored`
    } catch {
        return "Trigger failed — is FastAPI running?"
    }
}

export default function NarrativePage() {
    const { spikes, spikeCount, isWarming, isWaiting } = useNarrativeStream()
    const [selectedId, setSelectedId] = useState<string | null>(null)
    const [triggering, setTriggering] = useState(false)
    const [trigMsg, setTrigMsg] = useState<string | null>(null)

    const [arcCache, setArcCache] = useState<Record<string, string>>({})
    const [arcLoadingTopic, setArcLoadingTopic] = useState<string | null>(null)

    const selected = spikes.find(s => s.spike_id === selectedId) ?? spikes[0] ?? null
    const latestSrc = selected?.sources ?? { mastodon: 0, bluesky: 0, trends: 0, wikipedia: 0 }
    const resolvedArc = selected ? (selected.arc || arcCache[selected.topic] || null) : null

    useEffect(() => {
        if (!selected || resolvedArc || arcLoadingTopic === selected.topic) return
        let cancelled = false
        setArcLoadingTopic(selected.topic)

        fetch(`${API}/narrative/topic/${encodeURIComponent(selected.topic)}/arc`)
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (cancelled || !data?.arc) return
                setArcCache(prev => ({ ...prev, [data.topic]: data.arc }))
            })
            .catch(() => { })
            .finally(() => { if (!cancelled) setArcLoadingTopic(null) })

        return () => { cancelled = true }
    }, [selected?.topic, resolvedArc])

    async function handleForce() {
        setTriggering(true); setTrigMsg(null)
        const msg = await forceTick()
        setTrigMsg(msg); setTriggering(false)
    }

    const selectSpike = (id: string) => setSelectedId(id)
    const isSparse = spikes.length > 0 && spikes.length <= 3

    return (
        <div className="ip-page">


            <div className="ip-header">
                <div className="ip-header-left">
                    <div className="ip-eyebrow">IsolationForest · contamination=0.05 · 60s cadence · sorted by live fixture</div>
                    <div className="ip-title">Narrative Hub</div>
                    <div className="ip-subtitle">
                        Multi-source signal spike detection — Mastodon · Bluesky · Trends · Wikipedia
                    </div>
                </div>
                <div className="ip-actions">
                    {spikeCount > 0 && (
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: ".68rem", color: "var(--accent)", padding: "5px 10px", background: "var(--accent-dim)", border: "1px solid var(--accent-glow)", borderRadius: "var(--r-sm)" }}>
                            {spikeCount} spike{spikeCount !== 1 ? "s" : ""} detected
                        </span>
                    )}
                    {trigMsg && (
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: ".64rem", color: "var(--text-2)", maxWidth: 260 }}>{trigMsg}</span>
                    )}
                    <button onClick={handleForce} disabled={triggering} className="nar-force-btn">
                        {triggering ? "Detecting…" : "Force tick"}
                    </button>
                </div>
            </div>


            {selected && (
                <div className="narrative-arc-top">
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: ".6rem", textTransform: "uppercase", letterSpacing: ".12em", color: "var(--c-ai)", display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                        <span style={{ width: 3, height: 10, background: "var(--c-ai)", borderRadius: 2, display: "inline-block" }} />
                        Mistral 7B · Arc Synthesis · Weaviate RAG
                    </div>

                    <div className="spike-arc-panel">
                        <div className="spike-arc-panel-header">
                            <div className="spike-arc-dot" />
                            <span className="spike-arc-label">Narrative Arc — {selected.topic}</span>
                            {(() => {
                                const badge = fixtureBadge(selected)
                                return badge ? (
                                    <span style={{
                                        marginLeft: "auto", fontFamily: "var(--font-mono)", fontSize: ".56rem",
                                        letterSpacing: ".08em", color: badge.color, background: badge.bg,
                                        padding: "2px 8px", borderRadius: 10,
                                    }}>
                                        {badge.label}
                                    </span>
                                ) : null
                            })()}
                        </div>
                        {resolvedArc ? (
                            <p className="spike-arc-body">"{resolvedArc}"</p>
                        ) : (
                            <div className="spike-arc-loading">
                                <span className="arc-spinner" aria-hidden="true" />
                                <div>
                                    <div className="arc-loading-title">Mistral 7B is analysing this spike…</div>
                                    <div className="arc-loading-sub">Pulling historical precedent from Weaviate, then generating a take — usually 5-15 seconds</div>
                                </div>
                                <style jsx>{`
                                    .spike-arc-loading {
                                        display: flex;
                                        align-items: center;
                                        gap: 12px;
                                        padding: 4px 0;
                                    }
                                    .arc-spinner {
                                        flex-shrink: 0;
                                        display: inline-block;
                                        width: 20px;
                                        height: 20px;
                                        border-radius: 50%;
                                        border: 2.5px solid var(--border);
                                        border-top-color: var(--c-ai, #a78bfa);
                                        animation: arc-spin 0.8s linear infinite;
                                    }
                                    .arc-loading-title {
                                        font-size: .78rem;
                                        font-weight: 600;
                                        color: var(--text-1);
                                    }
                                    .arc-loading-sub {
                                        font-size: .68rem;
                                        color: var(--text-3);
                                        margin-top: 2px;
                                    }
                                    @keyframes arc-spin {
                                        to { transform: rotate(360deg); }
                                    }
                                `}</style>
                            </div>
                        )}
                    </div>


                    <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 8 }}>
                        <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "uppercase", letterSpacing: ".1em", color: "var(--text-3)" }}>
                            Driving sources
                        </div>
                        {(selected.source_names ?? []).map(src => {
                            const cfg = cfgFor(src)
                            const val = (selected.sources as Record<string, number>)[src] ?? 0
                            const pct = Math.min(100, (val / cfg.max) * 100)
                            return (
                                <div key={src} style={{ display: "grid", gridTemplateColumns: "26px 1fr 44px", alignItems: "center", gap: 8 }}>
                                    <span>{cfg.icon}</span>
                                    <div style={{ height: 5, background: "var(--bg-4)", borderRadius: 3, overflow: "hidden" }}>
                                        <div style={{ height: "100%", width: `${pct.toFixed(1)}%`, background: cfg.color, borderRadius: 3 }} />
                                    </div>
                                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".66rem", color: cfg.color, textAlign: "right" }}>
                                        {val.toFixed(1)}
                                    </span>
                                </div>
                            )
                        })}
                    </div>


                    <div style={{ marginTop: 14, padding: "12px", background: "var(--bg-3)", borderRadius: "var(--r-md)", border: "1px solid var(--border)" }}>
                        <div style={{ fontFamily: "var(--font-mono)", fontSize: ".56rem", textTransform: "uppercase", letterSpacing: ".1em", color: "var(--text-3)", marginBottom: 8 }}>
                            Detection info
                        </div>
                        {[
                            ["Severity", `${(selected.severity * 100).toFixed(0)}% above rolling baseline`],
                            ["Tick", `#${selected.tick}`],
                            ["Sources", selected.source_names?.join(", ") || "—"],
                            ["Detected", new Date(selected.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })],
                        ].map(([k, v]) => (
                            <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid var(--border)", fontSize: ".7rem" }}>
                                <span style={{ color: "var(--text-3)" }}>{k}</span>
                                <span style={{ color: "var(--text-1)", fontFamily: "var(--font-mono)", fontSize: ".66rem" }}>{v}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}


            <div className="source-strip source-strip-compact">
                {SOURCES.map(src => {
                    const cfg = cfgFor(src)
                    const val = latestSrc[src] ?? 0
                    const pct = Math.min(100, (val / cfg.max) * 100)
                    const isDriving = selected?.source_names?.includes(src)
                    return (
                        <div key={src} className={`source-cell${isDriving ? " driving" : ""}`}>
                            <div className="source-cell-top">
                                <span className="source-cell-icon">{cfg.icon}</span>
                                <span className="source-cell-name">{cfg.label}</span>
                                {isDriving && <span style={{ marginLeft: "auto", fontFamily: "var(--font-mono)", fontSize: ".5rem", color: cfg.color, background: cfg.bg, padding: "1px 5px", borderRadius: 3 }}>DRIVING</span>}
                            </div>
                            <div className="source-cell-value" style={{ color: isDriving ? cfg.color : "var(--text-1)" }}>
                                {val.toFixed(1)}
                            </div>
                            <div className="source-cell-unit">{cfg.unit}</div>
                            <div className="source-cell-bar">
                                <div
                                    className={`source-cell-fill${isDriving ? " driving-fill" : ""}`}
                                    style={{ width: `${pct.toFixed(1)}%`, background: cfg.color }}
                                />
                            </div>
                        </div>
                    )
                })}
            </div>


            {(isWarming || (isWaiting && spikes.length === 0)) && (
                <div className="nar-empty" style={{ minHeight: 260 }}>
                    <div className="nar-empty-icon">📡</div>
                    <div className="nar-empty-title">
                        {isWarming ? "Building 72h baseline — warming up" : "No spikes detected yet"}
                    </div>
                    <div className="nar-empty-sub">
                        {isWarming
                            ? "IsolationForest needs ~30 ticks to establish a rolling baseline. Click Force tick to accelerate warm-up."
                            : "The detector runs every 60 seconds across Mastodon, Bluesky, Trends, and Wikipedia. Use Force tick to run immediately."}
                    </div>
                    <button onClick={handleForce} disabled={triggering} className="nar-force-btn">
                        {triggering ? "Detecting…" : "Force tick now"}
                    </button>
                </div>
            )}

            {spikes.length > 0 && (<>


                <div className="spike-grid-section">
                    <div className="spike-grid-header">
                        <div className="spike-grid-title">
                            {isSparse ? "Detected spikes" : "All Spikes"} — sorted by live fixture, then severity — click to inspect
                        </div>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", color: "var(--text-3)" }}>
                            {spikes.length} total
                        </span>
                    </div>

                    <div className={`spike-grid${isSparse ? " spike-grid-sparse" : ""}`}>
                        {spikes.map(spike => {
                            const srcName = spike.source_names?.[0] ?? "mastodon"
                            const cfg = cfgFor(srcName)
                            const isSel = selectedId === spike.spike_id || (!selectedId && spike === spikes[0])
                            const ts = new Date(spike.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                            const sevPct = (spike.severity * 100).toFixed(0)
                            const badge = fixtureBadge(spike)

                            return (
                                <div
                                    key={spike.spike_id}
                                    className={`spike-grid-card${isSel ? " selected" : ""}`}
                                    onClick={() => selectSpike(spike.spike_id)}
                                    style={{ borderTopColor: isSel ? cfg.color : "transparent" }}
                                >
                                    <div className="sg-card-top">
                                        <div className="sg-card-icon" style={{ background: cfg.bg, border: `1px solid ${cfg.color}40` }}>
                                            {cfg.icon}
                                        </div>
                                        <div className="sg-card-topic">
                                            {spike.topic}
                                            {badge && (
                                                <span style={{
                                                    marginLeft: 6, fontFamily: "var(--font-mono)", fontSize: ".5rem",
                                                    letterSpacing: ".06em", color: badge.color, background: badge.bg,
                                                    padding: "1px 6px", borderRadius: 8, verticalAlign: "middle",
                                                }}>
                                                    {badge.label}
                                                </span>
                                            )}
                                        </div>
                                        <div className="sg-card-sev" style={{ color: spike.severity > 0.6 ? "var(--amber)" : cfg.color }}>
                                            {sevPct}%
                                        </div>
                                    </div>
                                    <div className="sg-card-sevbar">
                                        <div className="sg-card-sevbar-fill" style={{ width: `${sevPct}%`, background: cfg.color }} />
                                    </div>
                                    <p className="sg-card-summary">{spike.summary}</p>
                                    <div className="sg-card-footer">
                                        <span className="sg-card-sources">
                                            {(spike.source_names ?? [srcName]).map(s => cfgFor(s).icon).join(" ")}
                                        </span>
                                        <span className="sg-card-time">{ts}</span>
                                    </div>
                                </div>
                            )
                        })}

                        {spikes.length === 1 && (
                            <div className="spike-grid-card ghost">
                                <div className="sg-ghost-icon">＋</div>
                                <div className="sg-ghost-text">More spikes will appear here as they're detected</div>
                            </div>
                        )}
                    </div>
                </div>


                {selected && (
                    <div style={{ padding: "0 0 4px" }}>
                        <div style={{
                            padding: "14px 18px 0", fontFamily: "var(--font-mono)",
                            fontSize: ".6rem", textTransform: "uppercase",
                            letterSpacing: ".1em", color: "var(--text-3)"
                        }}>
                            Live comments — {selected.topic}
                        </div>
                        <CommentBubbles topic={selected.topic} />
                    </div>
                )}


                {selected && (
                    <div style={{ padding: "16px 18px", borderTop: "1px solid var(--border-bright)" }}>
                        <div className="spike-detail-topic-row">
                            <div className="spike-detail-topic">{selected.topic}</div>
                            <div className="spike-detail-badge">
                                <div className="spike-detail-sev">{(selected.severity * 100).toFixed(0)}%</div>
                                <div className="spike-detail-sev-lbl">above baseline</div>
                            </div>
                        </div>
                        <p className="spike-detail-summary">{selected.summary}</p>
                        <div className="spike-signal-grid">
                            {SOURCES.map(src => {
                                const val = selected.sources[src] ?? 0
                                const cfg = cfgFor(src)
                                return (
                                    <div key={src} className="spike-signal-cell">
                                        <span className="spike-signal-name">{cfg.icon} {cfg.label}</span>
                                        <span className="spike-signal-val" style={{ color: cfg.color }}>{val.toFixed(1)}</span>
                                        <div style={{ height: 3, background: "var(--bg-5)", borderRadius: 2, overflow: "hidden", marginTop: 2 }}>
                                            <div style={{ height: "100%", width: `${Math.min(100, (val / cfg.max) * 100).toFixed(1)}%`, background: cfg.color, borderRadius: 2 }} />
                                        </div>
                                        <span style={{ fontFamily: "var(--font-mono)", fontSize: ".52rem", color: "var(--text-3)" }}>{cfg.unit}</span>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                )}


                <div className="predict-method" style={{ margin: "16px 18px 0" }}>
                    <strong>Methodology —</strong> IsolationForest(contamination=0.05, n_estimators=100) on a 72-hour rolling window (4320 ticks × 4 features). Score threshold −0.10. Results sorted so topics tied to a currently-live fixture rank first, upcoming-kickoff topics next, then severity within each tier. Spikes trigger RAG over NarrativeArcs → Mistral 7B synthesis → arc stored back into Weaviate. 5-min cooldown per topic.
                </div>

            </>)}
        </div>
    )
}