"use client"
// components/match/NarrativeHoverBubbles.tsx
// 5 fixed circular bubbles positioned beside a hovered spike card.
// The BUBBLES don't move — only the comment text inside each cycles
// through the fetched samples on a staggered interval, so it reads as
// "shuffling comments" rather than "floating shapes."

import { useEffect, useState, useRef } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const BUBBLE_COUNT = 5
const CYCLE_MS = 2600

interface CommentSample {
    text: string
    source: "mastodon" | "bluesky"
    author: string
    timestamp: number
}

const SRC_ICONS: Record<string, string> = { mastodon: "🐘", bluesky: "🦋" }

// 5 fixed positions arranged in an arc beside the card — evenly spaced
// so none can ever overlap. Angles cover a ~200° arc to the right of
// the anchor point (flips left if not enough screen room, handled by caller).
const ANGLES_DEG = [-70, -35, 0, 35, 70]
const RADIUS = 78

interface Props {
    topic: string
    anchorRect: DOMRect
    flipLeft: boolean
}

export function NarrativeHoverBubbles({ topic, anchorRect, flipLeft }: Props) {
    const [samples, setSamples] = useState<CommentSample[]>([])
    const [cycleIdx, setCycleIdx] = useState<number[]>(Array(BUBBLE_COUNT).fill(0))
    const timers = useRef<ReturnType<typeof setInterval>[]>([])

    useEffect(() => {
        fetch(`${API}/narrative/${encodeURIComponent(topic)}/comments`)
            .then(r => r.ok ? r.json() : null)
            .then(data => setSamples(data?.samples ?? []))
            .catch(() => { })
    }, [topic])

    // Staggered independent cycling per bubble slot
    useEffect(() => {
        if (samples.length === 0) return
        timers.current.forEach(clearInterval)
        timers.current = []

        for (let i = 0; i < BUBBLE_COUNT; i++) {
            const delay = i * (CYCLE_MS / BUBBLE_COUNT)
            const t = setTimeout(() => {
                const interval = setInterval(() => {
                    setCycleIdx(prev => {
                        const next = [...prev]
                        next[i] = (next[i] + 1) % Math.max(1, samples.length)
                        return next
                    })
                }, CYCLE_MS)
                timers.current.push(interval)
            }, delay)
            timers.current.push(t as any)
        }
        return () => timers.current.forEach(clearInterval)
    }, [samples])

    if (samples.length === 0) return null

    const anchorCx = flipLeft ? anchorRect.left : anchorRect.right
    const anchorCy = anchorRect.top + anchorRect.height / 2

    return (
        <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 900 }}>
            {ANGLES_DEG.map((deg, i) => {
                const rad = (deg * Math.PI) / 180
                const dx = (flipLeft ? -1 : 1) * RADIUS * Math.cos(rad)
                const dy = RADIUS * Math.sin(rad)
                const cx = anchorCx + dx
                const cy = anchorCy + dy
                const sample = samples[cycleIdx[i] % samples.length]
                const cfg = SRC_ICONS[sample.source] ?? "💬"

                return (
                    <div
                        key={i}
                        style={{
                            position: "absolute",
                            left: cx, top: cy,
                            transform: "translate(-50%, -50%)",
                            width: 66, height: 66,
                            borderRadius: "50%",
                            background: "var(--bg-2)",
                            border: "1px solid var(--border-bright)",
                            boxShadow: "0 6px 16px rgba(0,0,0,0.25)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            padding: 6,
                            overflow: "hidden",
                        }}
                    >
                        <div
                            key={cycleIdx[i]}
                            style={{
                                fontSize: ".52rem", lineHeight: 1.15, color: "var(--text-1)",
                                textAlign: "center", display: "-webkit-box",
                                WebkitLineClamp: 4, WebkitBoxOrient: "vertical",
                                overflow: "hidden", animation: "bubble-fade-in .4s ease",
                            }}
                        >
                            <span style={{ fontSize: ".7rem" }}>{cfg}</span><br />
                            {sample.text.slice(0, 60)}
                        </div>
                    </div>
                )
            })}
        </div>
    )
}