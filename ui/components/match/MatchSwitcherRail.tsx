"use client"
// components/match/MatchSwitcherRail.tsx
// Left icon rail for jumping between matches without going back to the
// list — live matches first, then every finished match (most recent
// first). Previously restricted "finished" to only today's kickoffs,
// which left the rail nearly empty this late in the tournament (almost
// everything finished on earlier days). The list scrolls internally
// (.mp-rail-list has its own max-height), so there's no need to cap or
// day-restrict what's shown.

import { useEffect, useState } from "react"
import Link from "next/link"
import { Flag } from "@/components/Flag"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const LIVE = new Set(["1H", "2H", "ET", "P", "HT"])
const FINISHED = new Set(["FT", "AET", "PEN"])

interface RailFixture {
    id: string
    home_name: string
    away_name: string
    status_short: string
    kickoff_time: string | null
}

function toRailFixture(d: any): RailFixture | null {
    if (!d) return null
    return {
        id: String(d.fixture_id),
        home_name: d.home_name,
        away_name: d.away_name,
        status_short: d.status_short,
        kickoff_time: d.kickoff_time,
    }
}

function kickoffMs(f: RailFixture): number {
    if (!f.kickoff_time) return 0
    const t = new Date(f.kickoff_time).getTime()
    return Number.isNaN(t) ? 0 : t
}

interface Props { activeId: string }

export function MatchSwitcherRail({ activeId }: Props) {
    const [fixtures, setFixtures] = useState<RailFixture[]>([])

    useEffect(() => {
        let mounted = true
        async function load() {
            try {
                const r = await fetch(`${API}/matches/summary`)
                const { fixtures: raw } = await r.json()
                const parsed = (raw || []).map(toRailFixture).filter((f: RailFixture | null): f is RailFixture => f !== null)
                if (mounted) setFixtures(parsed)
            } catch { }
        }
        load()
        const t = setInterval(load, 30000)
        return () => { mounted = false; clearInterval(t) }
    }, [])

    const live = fixtures.filter(f => LIVE.has(f.status_short))
    const finished = fixtures
        .filter(f => FINISHED.has(f.status_short))
        .sort((a, b) => kickoffMs(b) - kickoffMs(a))
    const shown = [...live, ...finished]

    return (
        <nav className="mp-rail">
            <Link href="/" className="mp-rail-home" title="All matches">⚽</Link>
            <div className="mp-rail-list">
                {shown.map(f => (
                    <Link
                        key={f.id}
                        href={`/match/${f.id}`}
                        className={`mp-rail-item${f.id === activeId ? " active" : ""}`}
                        title={`${f.home_name} vs ${f.away_name}`}
                    >
                        <Flag team={f.home_name} size="sm" />
                        {LIVE.has(f.status_short) && <span className="mp-rail-live-dot" />}
                    </Link>
                ))}
            </div>
            <Link href="/" className="mp-rail-more" title="More matches">⌄</Link>
        </nav>
    )
}
