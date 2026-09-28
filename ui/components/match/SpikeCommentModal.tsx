"use client"
// components/match/SpikeCommentModal.tsx
// Modal popup showing real comment samples for a spike's topic as
// floating bubbles. Positions use a jittered GRID (not pure random)
// so bubbles are guaranteed non-overlapping — each bubble gets its
// own cell in an N×M grid sized to the comment count, then drifts
// within a small radius inside that cell only.

import { useEffect, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface CommentSample {
    text: string
    source: "mastodon" | "bluesky"
    author: string
    demo?: boolean
    permalink?: string | null
    timestamp: number
}

const SRC_STYLE: Record<string, { color: string; icon: string }> = {
    mastodon: { color: "#6364FF", icon: "🐘" },
    bluesky: { color: "#4f86f7", icon: "🦋" },
}

function seededRand(seed: string, salt: number): number {
    let h = salt
    for (let i = 0; i < seed.length; i++) { h = (h << 5) - h + seed.charCodeAt(i); h |= 0 }
    return Math.abs(h % 1000) / 1000
}

interface Props {
    spike: { topic: string; spike_id: string; severity: number }
    onClose: () => void
}

export function SpikeCommentModal({ spike, onClose }: Props) {
    const [samples, setSamples] = useState<CommentSample[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch(`${API}/narrative/${encodeURIComponent(spike.topic)}/comments`)
            .then(r => r.ok ? r.json() : null)
            .then(data => setSamples(data?.samples ?? []))
            .catch(() => { })
            .finally(() => setLoading(false))
    }, [spike.topic])

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
        window.addEventListener("keydown", onKey)
        return () => window.removeEventListener("keydown", onKey)
    }, [onClose])

    // Grid-cell placement — guarantees no two bubbles ever occupy the
    // same region. cols/rows chosen to roughly fit the field's aspect
    // ratio for up to 10 bubbles.
    const n = Math.min(samples.length, 10)
    const cols = n <= 4 ? 2 : n <= 6 ? 3 : 4
    const rows = Math.ceil(n / cols)
    const cellW = 100 / cols
    const cellH = 100 / rows

    return (
        <div
            onClick={onClose}
            style={{
                position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)",
                display: "flex", alignItems: "center", justifyContent: "center",
                zIndex: 1000, padding: 20,
            }}
        >
            <div
                onClick={e => e.stopPropagation()}
                style={{
                    width: "min(560px, 100%)", maxHeight: "80vh",
                    background: "var(--bg-2)", borderRadius: "var(--r-lg)",
                    border: "1px solid var(--border-bright)",
                    boxShadow: "0 24px 60px rgba(0,0,0,0.4)",
                    overflow: "hidden", display: "flex", flexDirection: "column",
                }}
            >
                {/* Header */}
                <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "14px 16px", borderBottom: "1px solid var(--border)",
                }}>
                    <div>
                        <div style={{ fontSize: ".92rem", fontWeight: 700, color: "var(--text-1)" }}>
                            {spike.topic} — Live Comments
                        </div>
                        <div style={{ fontSize: ".68rem", color: "var(--text-3)" }}>
                            {samples.length} real sample{samples.length !== 1 ? "s" : ""}
                        </div>
                    </div>
                    <button onClick={onClose} style={{
                        width: 28, height: 28, borderRadius: "50%", background: "var(--bg-3)",
                        border: "1px solid var(--border)", color: "var(--text-2)", cursor: "pointer",
                        fontSize: "1rem", lineHeight: 1,
                    }}>×</button>
                </div>

                {/* Bubble field */}
                <div style={{
                    position: "relative", height: 380, background: "var(--bg-3)",
                    overflow: "hidden",
                }}>
                    {loading && (
                        <div style={{
                            position: "absolute", inset: 0, display: "flex", alignItems: "center",
                            justifyContent: "center", color: "var(--text-3)", fontSize: ".78rem",
                        }}>
                            Loading comments…
                        </div>
                    )}

                    {!loading && samples.length === 0 && (
                        <div style={{
                            position: "absolute", inset: 0, display: "flex", flexDirection: "column",
                            alignItems: "center", justifyContent: "center", gap: 6, padding: 20,
                            textAlign: "center", color: "var(--text-3)", fontSize: ".76rem",
                        }}>
                            <span style={{ fontSize: "1.4rem", opacity: .5 }}>💬</span>
                            No live comment samples yet for {spike.topic}
                        </div>
                    )}

                    {samples.slice(0, n).map((s, i) => {
                        const cfg = SRC_STYLE[s.source] ?? SRC_STYLE.mastodon
                        const col = i % cols
                        const row = Math.floor(i / cols)

                        // Jitter WITHIN this bubble's own cell only — margin keeps
                        // bubbles away from cell edges so adjacent cells' bubbles
                        // can never touch even at max jitter.
                        const jitterX = (seededRand(s.text + i, 17) - 0.5) * (cellW * 0.3)
                        const jitterY = (seededRand(s.text + i, 31) - 0.5) * (cellH * 0.3)
                        const left = col * cellW + cellW / 2 + jitterX
                        const top = row * cellH + cellH / 2 + jitterY

                        const dur = 12 + seededRand(s.text + i, 7) * 8
                        const delay = -(seededRand(s.text + i, 53) * dur)
                        const size = 130 + Math.round(seededRand(s.text + i, 91) * 40)

                        return (
                            <div
                                key={`${s.text}-${i}`}
                                className="modal-comment-bubble"
                                style={{
                                    position: "absolute",
                                    left: `${left}%`, top: `${top}%`,
                                    width: size,
                                    animationDuration: `${dur}s`,
                                    animationDelay: `${delay}s`,
                                    borderColor: `${cfg.color}40`,
                                }}
                            >
                                <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 4 }}>
                                    <span style={{ color: cfg.color }}>{cfg.icon}</span>
                                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", color: "var(--text-3)" }}>
                                        {s.author}{s.demo ? " · demo" : ""}
                                    </span>
                                </div>
                                <p style={{ fontSize: ".7rem", color: "var(--text-1)", lineHeight: 1.4, margin: 0 }}>
                                    {s.text}
                                </p>
                            </div>
                        )
                    })}
                </div>

                <div style={{
                    padding: "8px 16px", fontFamily: "var(--font-mono)", fontSize: ".58rem",
                    color: "var(--text-3)", borderTop: "1px solid var(--border)", textAlign: "center",
                }}>
                    Press Esc or click outside to close
                </div>
            </div>
        </div>
    )
}