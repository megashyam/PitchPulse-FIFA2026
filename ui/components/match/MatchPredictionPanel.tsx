"use client"
// components/match/MatchPredictionPanel.tsx
// Per-match W/D/L prediction with CI bands + tournament implications.
// Sits below MomentumBar on the match dashboard.

import { useMatchPrediction } from "@/hooks/useMatchPrediction"
import type { MatchOdds, TeamTournament } from "@/hooks/useMatchPrediction"

interface Props {
    fixtureId: string
}

export function MatchPredictionPanel({ fixtureId }: Props) {
    const { prediction, loading } = useMatchPrediction(fixtureId)

    if (loading || !prediction) return null

    const { match_odds: odds, home_name, away_name, source } = prediction

    return (
        <div className="pred-panel">
            <div className="pred-header">
                <h3 className="panel-title">Match Prediction</h3>

            </div>

            {/* ── W/D/L stacked bar ── */}
            <div className="pred-bar">
                <div
                    className="pred-seg home"
                    style={{ width: `${(odds.home_win.p * 100).toFixed(2)}%` }}
                >
                    {odds.home_win.p > 0.10 ? `${(odds.home_win.p * 100).toFixed(0)}%` : ""}
                </div>
                <div
                    className="pred-seg draw"
                    style={{ width: `${(odds.draw.p * 100).toFixed(2)}%` }}
                >
                    {odds.draw.p > 0.10 ? `${(odds.draw.p * 100).toFixed(0)}%` : ""}
                </div>
                <div
                    className="pred-seg away"
                    style={{ width: `${(odds.away_win.p * 100).toFixed(2)}%` }}
                >
                    {odds.away_win.p > 0.10 ? `${(odds.away_win.p * 100).toFixed(0)}%` : ""}
                </div>
            </div>

            {/* ── Probability rows with CI ── */}
            <div className="pred-rows">
                <ProbRow
                    label={`${home_name} win`}
                    odds={odds.home_win}
                    color="var(--home)"
                />
                <ProbRow
                    label="Draw"
                    odds={odds.draw}
                    color="var(--text-3)"
                />
                <ProbRow
                    label={`${away_name} win`}
                    odds={odds.away_win}
                    color="var(--away)"
                />
            </div>

            {/* ── Tournament implications ── */}
            {(prediction.home_tournament || prediction.away_tournament) && (
                <div className="pred-tourney">
                    <div className="pred-tourney-header">Tournament path implications</div>
                    <div className="pred-tourney-grid">
                        {prediction.home_tournament && (
                            <TourneyCard team={prediction.home_tournament} side="home" />
                        )}
                        {prediction.away_tournament && (
                            <TourneyCard team={prediction.away_tournament} side="away" />
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}

// ── Probability bar with CI band ──────────────────────────────────────────

function ProbRow({ label, odds, color }: { label: string; odds: MatchOdds; color: string }) {
    const pct = (odds.p * 100).toFixed(1)
    const ciLo = (odds.ci_lo * 100).toFixed(1)
    const ciHi = (odds.ci_hi * 100).toFixed(1)
    const fillW = `${(odds.p * 100).toFixed(2)}%`
    const ciLeft = `${(odds.ci_lo * 100).toFixed(2)}%`
    const ciW = `${((odds.ci_hi - odds.ci_lo) * 100).toFixed(2)}%`
    // A model prior is a point estimate — only draw a band when one exists.
    const hasCI = odds.ci_hi - odds.ci_lo > 1e-6

    return (
        <div className="pred-prob-row">
            <span className="pred-prob-label">{label}</span>
            <div className="pred-prob-track">
                {/* CI band */}
                {hasCI && <div className="pred-ci-band" style={{ left: ciLeft, width: ciW }} />}
                {/* Fill */}
                <div className="pred-prob-fill" style={{ width: fillW, background: color }} />
            </div>
            <span className="pred-prob-val" style={{ color }}>
                {pct}%
                {hasCI && <span className="pred-ci-label">{ciLo}–{ciHi}</span>}
            </span>
        </div>
    )
}

// ── Tournament path card ──────────────────────────────────────────────────

const STAGES = ["r32", "r16", "qf", "sf", "final", "champion"] as const
const LABELS: Record<string, string> = {
    r32: "R32", r16: "R16", qf: "QF", sf: "SF", final: "F", champion: "★"
}

function TourneyCard({ team, side }: { team: TeamTournament; side: "home" | "away" }) {
    const color = side === "home" ? "var(--home)" : "var(--away)"

    return (
        <div className="pred-tourney-card">
            <div className="pred-tourney-name" style={{ color }}>{team.name}</div>
            {STAGES.map(stage => {
                const p = (team as any)[stage]?.p ?? 0
                const pct = (p * 100).toFixed(1)
                const w = `${(p * 100).toFixed(2)}%`
                return (
                    <div key={stage} className="pred-tourney-row">
                        <span className="pred-tourney-stage">{LABELS[stage]}</span>
                        <div className="pred-tourney-bar">
                            <div className="pred-tourney-fill" style={{ width: w, background: color }} />
                        </div>
                        <span className="pred-tourney-val">{pct}%</span>
                    </div>
                )
            })}
        </div>
    )
}