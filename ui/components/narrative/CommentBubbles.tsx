"use client"
// components/narrative/CommentBubbles.tsx
//
// Netflix-style auto-scrolling row of live comment cards. The whole row
// marquees horizontally on its own, pausing on hover/touch so the user can
// read. Cards are duplicated once so the scroll loops seamlessly.
// Data: GET /narrative/{topic}/comments, polled every 15s.

import { useEffect, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface CommentSample {
    text: string
    source: "mastodon" | "bluesky"
    author: string
    permalink?: string | null
    timestamp: number
    demo?: boolean
}

const SRC_STYLE: Record<string, { color: string; icon: string; label: string }> = {
    mastodon: { color: "#6364FF", icon: "🐘", label: "Mastodon" },
    bluesky: { color: "#4f86f7", icon: "🦋", label: "Bluesky" },
}

interface Props { topic: string | null }

export function CommentBubbles({ topic }: Props) {
    const [samples, setSamples] = useState<CommentSample[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        if (!topic) return
        let mounted = true
        setLoading(true)

        const load = () => {
            fetch(`${API}/narrative/${encodeURIComponent(topic)}/comments`)
                .then(r => r.ok ? r.json() : null)
                .then(data => {
                    if (mounted && data?.samples) setSamples(data.samples)
                })
                .catch(() => { })
                .finally(() => { if (mounted) setLoading(false) })
        }

        load()
        const t = setInterval(load, 15_000)
        return () => { mounted = false; clearInterval(t) }
    }, [topic])

    if (!topic) return null

    if (!loading && samples.length === 0) {
        return (
            <div className="comment-row-empty">
                <span style={{ opacity: .5 }}>💬</span>
                <span>No live comment samples yet for {topic}</span>
                <style jsx>{`
                    .comment-row-empty {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 14px;
                        font-size: .74rem;
                        color: var(--text-3);
                        border-top: 1px solid var(--border);
                    }
                `}</style>
            </div>
        )
    }

    if (loading && samples.length === 0) {
        return (
            <div className="comment-row-loading">
                <span className="comment-row-spinner" aria-hidden="true" />
                <span>Loading live comments…</span>
                <style jsx>{`
                    .comment-row-loading {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 14px;
                        font-size: .74rem;
                        color: var(--text-3);
                        border-top: 1px solid var(--border);
                    }
                    .comment-row-spinner {
                        display: inline-block;
                        width: 14px;
                        height: 14px;
                        border-radius: 50%;
                        border: 2px solid var(--border);
                        border-top-color: var(--accent, #38bdf8);
                        animation: crow-spin 0.8s linear infinite;
                    }
                    @keyframes crow-spin { to { transform: rotate(360deg); } }
                `}</style>
            </div>
        )
    }

    // Duplicate the list so the marquee can loop seamlessly (translateX
    // -50% lands exactly at the start of the second copy).
    const loopItems = [...samples, ...samples]
    const durationS = Math.max(18, samples.length * 5)

    return (
        <div className="comment-row-wrap">
            <div
                className="comment-row-track"
                style={{ animationDuration: `${durationS}s` }}
            >
                {loopItems.map((s, i) => {
                    const cfg = SRC_STYLE[s.source] ?? SRC_STYLE.mastodon
                    return (
                        <div className="comment-row-card" key={`${s.text}-${i}`}>
                            <div className="comment-row-head">
                                <span style={{ color: cfg.color }}>{cfg.icon}</span>
                                <span className="comment-row-author">{s.author || "anon"}</span>
                                {s.demo && <span className="comment-row-src" style={{ color: "var(--text-3)" }}>demo</span>}
                                <span className="comment-row-src" style={{ color: cfg.color }}>{cfg.label}</span>
                            </div>
                            <p className="comment-row-text">{s.text}</p>
                        </div>
                    )
                })}
            </div>

            <style jsx>{`
                /* .comment-row-track is absolutely positioned inside this
                   fixed-height wrapper so the marquee can't widen any flex or
                   grid ancestor. */
                .comment-row-wrap {
                    position: relative;
                    width: 100%;
                    max-width: 100%;
                    min-width: 0;
                    overflow: hidden;
                    height: 150px;
                    border-top: 1px solid var(--border);
                    -webkit-mask-image: linear-gradient(to right, transparent, #000 6%, #000 94%, transparent);
                    mask-image: linear-gradient(to right, transparent, #000 6%, #000 94%, transparent);
                }
                .comment-row-track {
                    position: absolute;
                    top: 12px;
                    left: 0;
                    display: flex;
                    gap: 10px;
                    width: max-content;
                    animation-name: crow-marquee;
                    animation-timing-function: linear;
                    animation-iteration-count: infinite;
                }
                .comment-row-wrap:hover .comment-row-track {
                    animation-play-state: paused;
                }
                @keyframes crow-marquee {
                    from { transform: translateX(0); }
                    to   { transform: translateX(-50%); }
                }
                .comment-row-card {
                    flex: 0 0 auto;
                    width: 190px;
                    height: 100%;
                    background: var(--glass-bg-inner);
                    border: 1px solid var(--glass-border-inner);
                    border-radius: var(--r-md, 10px);
                    padding: 12px;
                }
                .comment-row-head {
                    display: flex;
                    align-items: center;
                    gap: px;
                    margin-bottom: 4px;
                }
                .comment-row-author {
                    font-size: .64rem;
                    font-weight: 600;
                    color: var(--text-2);
                    flex: 1;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                .comment-row-src {
                    font-family: var(--font-mono);
                    font-size: .5rem;
                    text-transform: none;
                    letter-spacing: normal;
                }
                .comment-row-text {
                    font-size: .7rem;
                    line-height: 1.5;
                    color: var(--text-1);
                    margin: 0;
                    display: -webkit-box;
                    -webkit-line-clamp: 5;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }
            `}</style>
        </div>
    )
}