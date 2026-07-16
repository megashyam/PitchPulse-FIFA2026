"use client"



import { useEffect, useState } from "react"
import { useMatchStream } from "@/hooks/useMatchStream"
import { ScoreHeader } from "@/components/match/ScoreHeader"
import { StatsPanel } from "@/components/match/StatsPanel"
import { LiveProbCard } from "@/components/match/LiveProbCard"
import { EventsFeed } from "@/components/match/EventsFeed"
import { MatchTimeline } from "@/components/match/MatchTimeline"
import { GroupTable } from "@/components/match/GroupTable"
import { MatchSwitcherRail } from "@/components/match/MatchSwitcherRail"
import { UnifiedTabs } from "@/components/match/UnifiedTabs"
import type { MatchState } from "@/types/match"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface Props { params: { id: string } }

export default function MatchPage({ params }: Props) {
  const { id } = params
  const [initial, setInitial] = useState<MatchState | null>(null)
  const [initError, setInitError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API}/matches/${id}`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() })
      .then(setInitial)
      .catch(e => setInitError(e.message))
  }, [id])

  const { state: live, isWaiting, error: streamError } = useMatchStream(id)
  const state = live ?? initial

  if (initError && !state) return (
    <div className="page-error">
      <p>Could not load match <code>{id}</code></p>
      <p className="error-detail">{initError}</p>
      <a href="/" className="back-link">← All matches</a>
    </div>
  )

  if (!state) return (
    <div className="page-loading">
      <div className="spinner" />
      <p>Loading match…</p>
    </div>
  )

  return (
    <div className="mp-shell">

      <MatchSwitcherRail activeId={id} />

      <div className="mp-main">

        <div className="mp-header">
          {(streamError || isWaiting) && (
            <div className="match-topbar">
              {streamError && <span className="stream-error">⚠ {streamError}</span>}
              {isWaiting && <span className="stream-waiting">Connecting…</span>}
            </div>
          )}
          <ScoreHeader state={state} updatedAt={state.updated_at} />
        </div>


        <div className="mp-body">

          <div className="mp-side">
            <StatsPanel home={state.home_stats} away={state.away_stats} />
            <LiveProbCard state={state} fixtureId={id} />
            <EventsFeed state={state} />
            <MatchTimeline state={state} />
            <GroupTable fixtureId={id} highlightTeams={[state.home_name, state.away_name]} />
          </div>

          <div className="mp-content">
            <UnifiedTabs state={state} fixtureId={id} />
          </div>

        </div>
      </div>
    </div>
  )
}
