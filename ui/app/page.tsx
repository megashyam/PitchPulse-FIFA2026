"use client"
// app/page.tsx — grid of match cards, loaded with one GET /matches/summary call

import { useEffect, useState } from "react"
import Link from "next/link"
import { Flag } from "@/components/Flag"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const LIVE = new Set(["1H", "2H", "ET", "P", "HT", "LIVE"])

interface Fixture {
  id: string; home_name: string; away_name: string
  home_score: number; away_score: number
  status_short: string; elapsed: number | null
  round: string; kickoff_time: string | null; updated_at: string
}

function parseDate(f: Fixture): Date {
  try { return new Date(f.kickoff_time || f.updated_at) } catch { return new Date() }
}

// US Eastern, DST-aware.
const TZ = "America/New_York"
const dayKey = (d: Date) =>
  new Intl.DateTimeFormat("en-CA", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit" }).format(d)
function dateLabel(d: Date): string {
  const diff = (Date.parse(dayKey(d)) - Date.parse(dayKey(new Date()))) / 86400000
  if (diff === 0) return "Today"
  if (diff === 1) return "Tomorrow"
  if (diff === -1) return "Yesterday"
  return d.toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric", timeZone: TZ })
}

function toFixture(d: any): Fixture | null {
  if (!d) return null
  return {
    id: String(d.fixture_id), home_name: d.home_name, away_name: d.away_name,
    home_score: d.home_score ?? 0, away_score: d.away_score ?? 0,
    status_short: d.status_short, elapsed: d.elapsed,
    round: d.round || "", kickoff_time: d.kickoff_time, updated_at: d.updated_at,
  }
}

export default function HomePage() {
  const [fixtures, setFixtures] = useState<Fixture[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<"all" | "live" | "upcoming">("all")

  useEffect(() => {
    async function load() {
      try {
        // Single bulk call for every fixture.
        const r = await fetch(`${API}/matches/summary`)
        const { fixtures: raw } = await r.json()
        const details: (Fixture | null)[] = (raw || []).map(toFixture)
        setFixtures(details.filter((f): f is Fixture => f !== null))
      } catch { } finally { setLoading(false) }
    }
    load()
    const t = setInterval(load, 30000)
    return () => clearInterval(t)
  }, [])

  const filtered = fixtures.filter(f => {
    if (filter === "live") return LIVE.has(f.status_short)
    if (filter === "upcoming") return f.status_short === "NS"
    return true
  })

  // group by date
  const groups = new Map<string, { label: string; items: Fixture[] }>()
  const sorted = [...filtered].sort((a, b) => parseDate(b).getTime() - parseDate(a).getTime())
  for (const f of sorted) {
    const d = parseDate(f)
    const k = dayKey(d)
    if (!groups.has(k)) groups.set(k, { label: dateLabel(d), items: [] })
    groups.get(k)!.items.push(f)
  }

  const liveCount = fixtures.filter(f => LIVE.has(f.status_short)).length

  return (
    <div className="v4-home">

      {/* Hero */}
      <div className="v4-home-hero">
        <div className="v4-hero-eyebrow">FIFA World Cup 2026 · USA · Canada · Mexico</div>
        <div className="v4-hero-title">Match Center</div>
        <div className="v4-hero-sub">
          {liveCount > 0 ? `${liveCount} match${liveCount > 1 ? "es" : ""} live now · ` : ""}
          AI-powered analysis, predictions &amp; insights
        </div>
      </div>

      {/* Filter tabs */}
      <div className="v4-home-tabs">
        <button className={`v4-home-tab${filter === "all" ? " active" : ""}`} onClick={() => setFilter("all")}>All Matches</button>
        <button className={`v4-home-tab${filter === "live" ? " active" : ""}`} onClick={() => setFilter("live")}>
          Live {liveCount > 0 && `(${liveCount})`}
        </button>
        <button className={`v4-home-tab${filter === "upcoming" ? " active" : ""}`} onClick={() => setFilter("upcoming")}>Upcoming</button>
      </div>

      {loading && (
        <div className="v4-skeleton">
          {[1, 2, 3].map(i => <div key={i} className="v4-skel" style={{ height: 140 }} />)}
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--text-3)" }}>
          No {filter !== "all" ? filter : ""} matches to show
        </div>
      )}

      {[...groups.entries()].map(([key, group]) => (
        <div key={key} className="v4-date-group">
          <div className="v4-date-label">{group.label}</div>
          <div className="v4-match-grid">
            {group.items.map(f => <MatchCard key={f.id} f={f} />)}
          </div>
        </div>
      ))}

    </div>
  )
}

function MatchCard({ f }: { f: Fixture }) {
  const isLive = LIVE.has(f.status_short)
  const isFT = ["FT", "AET", "PEN"].includes(f.status_short)
  const isNS = f.status_short === "NS"
  const showScore = !isNS
  const homeWin = isFT && f.home_score > f.away_score
  const awayWin = isFT && f.away_score > f.home_score
  const group = f.round.match(/Group ([A-L])/)?.[1]
  const kickoff = f.kickoff_time
    ? new Date(f.kickoff_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: TZ })
    : "TBD"

  return (
    <Link href={`/match/${f.id}`} className="v4-match-card">
      <div className="v4-mc-top">
        <span className="v4-mc-comp">World Cup 2026</span>
        {isLive ? (
          <span className="v4-mc-status-live">
            <span className="live-dot" /> {f.elapsed}'
          </span>
        ) : group ? (
          <span className="v4-mc-group">Group {group}</span>
        ) : (
          <span className="v4-mc-comp">{isFT ? "FT" : ""}</span>
        )}
      </div>

      {showScore ? (
        <div className="v4-mc-teams">
          <div className={`v4-mc-team${homeWin ? " winner" : awayWin ? " loser" : ""}`}>
            <Flag team={f.home_name} size="md" />
            <span className="v4-mc-team-name">{f.home_name}</span>
            <span className="v4-mc-team-score">{f.home_score}</span>
          </div>
          <div className={`v4-mc-team${awayWin ? " winner" : homeWin ? " loser" : ""}`}>
            <Flag team={f.away_name} size="md" />
            <span className="v4-mc-team-name">{f.away_name}</span>
            <span className="v4-mc-team-score">{f.away_score}</span>
          </div>
        </div>
      ) : (
        <>
          <div className="v4-mc-teams">
            <div className="v4-mc-team">
              <Flag team={f.home_name} size="md" />
              <span className="v4-mc-team-name">{f.home_name}</span>
            </div>
            <div className="v4-mc-team">
              <Flag team={f.away_name} size="md" />
              <span className="v4-mc-team-name">{f.away_name}</span>
            </div>
          </div>
          <div className="v4-mc-divider" />
          <div className="v4-mc-kickoff">
            <span className="v4-mc-kickoff-time">{kickoff}</span>
            <span className="v4-mc-kickoff-label">Kick-off</span>
          </div>
        </>
      )}

      <div className="v4-mc-footer">
        <span className="v4-mc-venue">{isLive ? "● Live now" : isFT ? "Full time" : group ? `Group ${group}` : "Scheduled"}</span>
        <span className="v4-mc-cta">View →</span>
      </div>
    </Link>
  )
}
