"use client"
// components/match/NarrativeCarousel.tsx
// Detail panel first, 2-column spike grid below it.
// Clicking any spike (in the grid or elsewhere) opens SpikeCommentModal
// with floating, non-overlapping real comment bubbles for that topic.

import { useState } from "react"
import { useNarrativeStream } from "@/hooks/useNarrativeStream"
import { Flag } from "@/components/Flag"
import { CommentBubbles } from "@/components/narrative/CommentBubbles"

const SRC_ICONS: Record<string, string> = {
    mastodon: "🐘", bluesky: "🦋", trends: "📈", wikipedia: "📖",
}
const SRC_LABELS: Record<string, string> = {
    mastodon: "Mastodon", bluesky: "Bluesky", trends: "Trends", wikipedia: "Wikipedia",
}
const SRC_UNITS: Record<string, string> = {
    mastodon: "posts/hr", bluesky: "posts/hr", trends: "× last hr", wikipedia: "edits/hr",
}
const SRC_MAX: Record<string, number> = {
    mastodon: 60, bluesky: 1000, trends: 5, wikipedia: 20,
}
const SRC_COLOR: Record<string, string> = {
    mastodon: "#6364FF", bluesky: "#4f86f7", trends: "#f59e0b", wikipedia: "#10d9a0",
}
const NAME_ALIAS: Record<string, string> = {
    "United States": "USA", "United States of America": "USA",
    "Korea Republic": "South Korea", "IR Iran": "Iran",
}
const canonTeam = (n?: string) => (n ? NAME_ALIAS[n] ?? n : "")
const SOURCES = ["mastodon", "bluesky", "trends", "wikipedia"] as const

function isTeamTopic(topic: string) {
    return !["WC2026", "WorldCup2026"].includes(topic)
}

function seededPoints(seed: string, n = 8): number[] {
    let h = 0
    for (let i = 0; i < seed.length; i++) { h = (h << 5) - h + seed.charCodeAt(i); h |= 0 }
    const pts: number[] = []
    let v = 0.3 + (Math.abs(h % 100) / 100) * 0.2
    for (let i = 0; i < n; i++) {
        h = (h * 1103515245 + 12345) & 0x7fffffff
        v = Math.max(0.1, Math.min(0.95, v + ((h % 100) / 100 - 0.42) * 0.25))
        pts.push(v)
    }
    return pts
}

// Deterministic pseudo-random walk, NOT real historical tick data — there's
// no per-minute time series backing this signal, only the current severity/
// source snapshot. <title> makes that discoverable on hover instead of the
// sparkline silently reading as a real trend line next to genuine numbers.
function Sparkline({ seed, color }: { seed: string; color: string }) {
    const pts = seededPoints(seed)
    const w = 100, h = 26
    const step = w / (pts.length - 1)
    const path = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${(i * step).toFixed(1)} ${(h - p * h).toFixed(1)}`).join(" ")
    const areaPath = `${path} L ${w} ${h} L 0 ${h} Z`
    return (
        <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: h, display: "block" }}>
            <title>Simulated trend for display only — not historical data</title>
            <path d={areaPath} fill={color} opacity={0.12} />
            <path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
        </svg>
    )
}

function severityColor(sev: number): string {
    if (sev >= 0.7) return "#f05454"
    if (sev >= 0.4) return "#f59e0b"
    return "#10d9a0"
}

export function NarrativeCarousel({ homeTeam, awayTeam }: { homeTeam?: string; awayTeam?: string }) {
    const { spikes, isWarming, isWaiting } = useNarrativeStream()
    const [selectedId, setSelectedId] = useState<string | null>(null)
    const [commentsOpen, setCommentsOpen] = useState(false)
    const [selSource, setSelSource] = useState<string | null>(null)

    // Tournament-wide "trending now": the most-discussed stories across all
    // teams, sorted by how far above baseline each spike sits. The current
    // match's teams are flagged with a marker.
    const matchTeams = new Set(
        [canonTeam(homeTeam), canonTeam(awayTeam)].filter(Boolean)
    )
    const trending = [...spikes].sort((a, b) => b.severity - a.severity)
    const spikeCount = trending.length

    const selected = trending.find(s => s.spike_id === selectedId) ?? trending[0] ?? null

    // Default the source detail to the top driving source (never "stuck on mastodon").
    const drivingDefault = selected
        ? (selected.source_names?.[0]
            ?? SOURCES.reduce((a, b) =>
                (selected!.sources[b] ?? 0) / SRC_MAX[b] > (selected!.sources[a] ?? 0) / SRC_MAX[a] ? b : a,
                SOURCES[0]))
        : "mastodon"
    const activeSource =
        selSource && selected && (selected.sources as Record<string, number>)[selSource] != null
            ? selSource
            : drivingDefault

    if (isWarming || (isWaiting && spikes.length === 0)) return (
        <div style={{ padding: "24px 16px", textAlign: "center" }}>
            <div style={{ fontSize: "1.4rem", marginBottom: 8, opacity: .5 }}>📡</div>
            <div style={{ fontSize: ".8rem", fontWeight: 600, color: "var(--text-1)", marginBottom: 4 }}>
                {isWarming ? "Building baseline…" : "No spikes yet"}
            </div>
            <div style={{ fontSize: ".72rem", color: "var(--text-3)" }}>
                Live narrative spikes will appear here as they're detected
            </div>
        </div>
    )

    if (trending.length === 0) return (
        <div style={{ padding: "24px 16px", textAlign: "center" }}>
            <div style={{ fontSize: "1.4rem", marginBottom: 8, opacity: .5 }}>🔥</div>
            <div style={{ fontSize: ".8rem", fontWeight: 600, color: "var(--text-1)", marginBottom: 4 }}>
                No trending stories yet
            </div>
            <div style={{ fontSize: ".72rem", color: "var(--text-3)" }}>
                Trending narrative spikes across the tournament will appear here as they're detected
            </div>
        </div>
    )

    return (
        <div>

            {/* ── Detail panel — the selected/most-recent spike, stacked FIRST ── */}
            {selected && (
                <div style={{ padding: "16px 14px", borderBottom: "1px solid var(--border-bright)" }}>

                    {/* Narrative Arc, above everything else in this panel */}
                    <div style={{ marginBottom: 16 }}>
                        <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--c-ai)", display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                            <span style={{ width: 3, height: 10, background: "var(--c-ai)", borderRadius: 2, flexShrink: 0 }} />
                            LLM arc synthesis · Weaviate RAG
                        </div>
                        <div style={{ background: "var(--glass-bg-inner)", border: "1px solid var(--glass-border-inner)", borderRadius: "var(--r-md)", padding: "16px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--c-ai)", flexShrink: 0 }} />
                                <span style={{ fontFamily: "var(--font-mono)", fontSize: ".56rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-2)" }}>
                                    Narrative Arc
                                </span>
                            </div>
                            {selected.arc ? (
                                <p style={{ fontSize: ".78rem", color: "var(--text-1)", fontStyle: "italic", lineHeight: 1.55, margin: 0 }}>
                                    "{selected.arc}"
                                </p>
                            ) : (
                                <p style={{ fontSize: ".76rem", color: "var(--text-2)", lineHeight: 1.5, margin: 0 }}>
                                    {selected.summary}
                                </p>
                            )}
                        </div>
                    </div>

                    <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 8 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            {isTeamTopic(selected.topic)
                                ? <Flag team={selected.topic} size="md" />
                                : <span style={{ fontSize: "1.6rem" }}>🌍</span>}
                            <span style={{ fontFamily: "var(--font-display)", fontSize: "1.5rem", letterSpacing: "normal", color: "var(--text-1)" }}>
                                {selected.topic.toUpperCase()}
                            </span>
                            {matchTeams.has(canonTeam(selected.topic)) && (
                                <span style={{ fontFamily: "var(--font-mono)", fontSize: ".52rem", color: "var(--accent)", border: "1px solid var(--accent-glow)", borderRadius: 4, padding: "1px 4px", alignSelf: "center" }}>THIS MATCH</span>
                            )}
                        </div>
                        <div style={{ textAlign: "right" }}>
                            <div style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", color: severityColor(selected.severity), lineHeight: 1 }}>
                                {Math.round(selected.severity * 100)}%
                            </div>
                            <div style={{ fontFamily: "var(--font-mono)", fontSize: ".56rem", color: "var(--text-3)", textTransform: "none" }}>
                                above baseline
                            </div>
                        </div>
                    </div>

                    <p style={{ fontSize: ".76rem", color: "var(--text-2)", marginBottom: 10 }}>{selected.summary}</p>

                    {/* Compact single-row source stats; driving sources get a ▲ marker. */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6, marginBottom: 10 }}>
                        {SOURCES.map(src => {
                            const val = selected.sources[src] ?? 0
                            const driving = selected.source_names?.includes(src)
                            const active = activeSource === src
                            const color = SRC_COLOR[src]
                            return (
                                <button key={src} onClick={() => setSelSource(src)} style={{
                                    background: active ? `${color}22` : driving ? `${color}12` : "var(--glass-bg-inner)",
                                    border: `1px solid ${active ? color : driving ? color + "40" : "var(--glass-border-inner)"}`,
                                    borderRadius: 6, padding: "5px 6px", textAlign: "center", position: "relative", cursor: "pointer",
                                }}>
                                    {driving && <span style={{ position: "absolute", top: 2, right: 3, fontSize: ".5rem", color }}>▲</span>}
                                    <div style={{ fontSize: ".7rem" }}>{SRC_ICONS[src]}</div>
                                    <div style={{ fontFamily: "var(--font-mono)", fontSize: ".72rem", fontWeight: 700, color }}>
                                        {val.toFixed(1)}
                                    </div>
                                </button>
                            )
                        })}
                    </div>

                    {/* Per-source detail — updates when a source button above is clicked.
                        The number is real (severity/source data from the backend); the
                        sparkline beside it is a seeded pseudo-random walk, not real
                        historical tick data — labelled so it doesn't read as one. */}
                    <div style={{ marginBottom: 10 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: ".56rem", color: "var(--text-3)", marginBottom: 4 }}>
                            <span>{SRC_LABELS[activeSource]}</span>
                            <span>{((selected.sources as Record<string, number>)[activeSource] ?? 0).toFixed(1)} {SRC_UNITS[activeSource]}</span>
                        </div>
                        <Sparkline seed={`${selected.spike_id}:${activeSource}`} color={SRC_COLOR[activeSource]} />
                        <div style={{ fontFamily: "var(--font-mono)", fontSize: ".52rem", color: "var(--text-3)", opacity: .7, marginTop: 2 }}>
                            simulated trend, not historical data
                        </div>
                    </div>

                    <button
                        onClick={() => setCommentsOpen(o => !o)}
                        style={{
                            width: "100%", padding: "8px", borderRadius: "var(--r-md)",
                            background: "var(--accent-dim)", border: "1px solid var(--accent-glow)",
                            color: "var(--accent)", fontSize: ".74rem", fontWeight: 700, cursor: "pointer",
                        }}
                    >
                        💬 {commentsOpen ? "Hide live comments" : "View live comments"}
                    </button>

                    {/* Inline sliding carousel, directly beneath this button. */}
                    {commentsOpen && <CommentBubbles topic={selected.topic} />}

                    <div style={{ fontFamily: "var(--font-mono)", fontSize: ".56rem", color: "var(--text-3)", marginTop: 8 }}>
                        ID: {selected.spike_id} · Tick {selected.tick}
                    </div>
                </div>
            )}

            {/* ── 2-column grid of all spikes ── */}
            <div style={{ padding: "12px 14px 14px" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)" }}>
                        Trending now · {spikeCount} stor{spikeCount !== 1 ? "ies" : "y"}
                    </span>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    {trending.map(spike => {
                        const color = severityColor(spike.severity)
                        const isSel = spike.spike_id === selected?.spike_id
                        return (
                            <div
                                key={spike.spike_id}
                                onClick={() => setSelectedId(spike.spike_id)}
                                onDoubleClick={() => { setSelectedId(spike.spike_id); setCommentsOpen(true) }}
                                style={{
                                    background: "var(--glass-bg-inner)", border: `1px solid ${isSel ? "var(--accent)" : "var(--glass-border-inner)"}`,
                                    borderRadius: "var(--r-md)", padding: "12px", cursor: "pointer",
                                    transition: "all .15s",
                                }}
                            >
                                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                                    {isTeamTopic(spike.topic)
                                        ? <Flag team={spike.topic} size="sm" />
                                        : <span style={{ fontSize: ".9rem" }}>🌍</span>}
                                    <span style={{ fontSize: ".72rem", fontWeight: 700, color: "var(--text-1)", flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                                        {spike.topic}
                                    </span>
                                    {matchTeams.has(canonTeam(spike.topic)) && (
                                        <span style={{ fontFamily: "var(--font-mono)", fontSize: ".46rem", color: "var(--accent)", border: "1px solid var(--accent-glow)", borderRadius: 3, padding: "0 2px", whiteSpace: "nowrap" }}>MATCH</span>
                                    )}
                                    <span style={{ fontSize: ".76rem", fontWeight: 700, color }}>
                                        {Math.round(spike.severity * 100)}%
                                    </span>
                                </div>
                                <Sparkline seed={spike.spike_id} color={color} />
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 4 }}>
                                    <span style={{ fontSize: ".7rem" }}>
                                        {(spike.source_names ?? []).map(s => SRC_ICONS[s] ?? "").join(" ")}
                                    </span>
                                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".54rem", color: "var(--text-2)" }}>
                                        {new Date(spike.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                                    </span>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>

        </div>
    )
}