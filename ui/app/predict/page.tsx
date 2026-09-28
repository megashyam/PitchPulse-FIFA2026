"use client"
// app/predict/page.tsx — fully interactive outcome predictor

import { useMemo, useState } from "react"
import { usePredictStream } from "@/hooks/usePredictStream"
import { PerMatchOddsPanel } from "@/components/predict/PerMatchOddsPanel"
import { BracketImpactFeed } from "@/components/predict/BracketImpactFeed"
import type { TeamPrediction, Stage } from "@/types/predict"

const STAGES: Stage[] = ["r32", "r16", "qf", "sf", "final", "champion"]
const STAGE_LABELS: Record<Stage, string> = { r32: "R32", r16: "R16", qf: "QF", sf: "SF", final: "Final", champion: "Champion" }
const STAGE_COLORS: Record<Stage, string> = {
    r32: "#4f86f7", r16: "#4f86f7", qf: "#9b6cf7", sf: "#f59e0b", final: "#f59e0b", champion: "#10d9a0",
}

const GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

export default function PredictPage() {
    const { prediction, status, isLoading, error, triggerSim, refresh } = usePredictStream()
    const isRunning = status?.status === "running" || isLoading
    const hasData = !!prediction

    const [activeGroup, setActiveGroup] = useState("A")
    const [sortStage, setSortStage] = useState<Stage>("champion")
    const [expandedTeam, setExpandedTeam] = useState<string | null>(null)
    const [selectedTop8, setSelectedTop8] = useState<string | null>(null)

    // Top 4 for KPI strip
    const top4 = useMemo(() =>
        hasData ? [...prediction!.teams].sort((a, b) => b.champion.p - a.champion.p).slice(0, 4) : []
        , [prediction])

    // Teams in the active group sorted by champion prob
    const groupTeams = useMemo(() =>
        hasData
            ? prediction!.teams.filter(t => t.group === activeGroup).sort((a, b) => b.champion.p - a.champion.p)
            : []
        , [prediction, activeGroup])

    // Full table sorted
    const sorted = useMemo(() =>
        hasData ? [...prediction!.teams].sort((a, b) => b[sortStage].p - a[sortStage].p) : []
        , [prediction, sortStage])

    // Top 8 for chart
    const top8 = useMemo(() =>
        hasData ? [...prediction!.teams].sort((a, b) => b.champion.p - a.champion.p).slice(0, 8) : []
        , [prediction])
    const maxP = top8[0]?.champion.p ?? 1

    const toggleTeam = (name: string) => setExpandedTeam(prev => prev === name ? null : name)

    return (
        <div className="ip-page">

            {/* ── Header ── */}
            <div className="ip-header">
                <div className="ip-header-left">
                    <div className="ip-eyebrow">Monte Carlo · 48-team · numpy vectorized</div>
                    <div className="ip-title">Outcome Predictor</div>
                    <div className="ip-subtitle">
                        {hasData
                            ? `${prediction!.n_sims.toLocaleString()} runs · ${prediction!.elapsed_s}s · ${new Date(prediction!.run_at).toLocaleTimeString()}`
                            : "Run the simulation to generate win probabilities for all 48 teams"}
                    </div>
                </div>
                <div className="ip-actions">
                    {isRunning && <span style={{ fontFamily: "var(--font-mono)", fontSize: ".68rem", color: "var(--amber)", display: "flex", alignItems: "center", gap: 6 }}><span className="live-dot" />Simulating…</span>}
                    {error && <span style={{ fontSize: ".72rem", color: "var(--c-red)" }}>{error}</span>}
                    <button onClick={refresh} disabled={isRunning} className="btn-refresh" title="Refresh">↻</button>
                    <button onClick={() => triggerSim(50_000)} disabled={isRunning} className="btn-sim">
                        {isRunning ? "Running…" : hasData ? "Re-run 50k" : "Run 50k Sim"}
                    </button>
                </div>
            </div>

            {/* ── Empty state ── */}
            {!hasData && !isRunning && (
                <div className="nar-empty" style={{ minHeight: 320 }}>
                    <div className="nar-empty-icon">🎯</div>
                    <div className="nar-empty-title">No simulation results yet</div>
                    <div className="nar-empty-sub">Generate win probabilities, group advancement chances, and bracket implications for all 48 WC 2026 teams.</div>
                    <button onClick={() => triggerSim(50_000)} className="nar-force-btn">Run 50,000 Simulations</button>
                </div>
            )}

            {isRunning && !hasData && (
                <div className="nar-empty" style={{ minHeight: 200 }}>
                    <div className="live-dot" style={{ width: 10, height: 10 }} />
                    <div className="nar-empty-title">Simulating 50,000 tournaments…</div>
                    <div className="nar-empty-sub">Takes ~8 seconds on CPU. Results will appear automatically.</div>
                </div>
            )}

            {hasData && (<>

                {/* ── KPI strip — top 4 teams ── */}
                <div className="kpi-strip">
                    {top4.map((team, i) => {
                        const colors = ["var(--accent)", "var(--home)", "var(--c-ai)", "var(--amber)"]
                        const icons = ["🥇", "🥈", "🥉", "4️⃣"]
                        const bg = ["rgba(16,217,160,.12)", "rgba(79,134,247,.12)", "rgba(155,108,247,.12)", "rgba(245,158,11,.1)"]
                        return (
                            <div key={team.name} className="kpi-cell">
                                <div className="kpi-cell-top">
                                    <div className="kpi-cell-icon" style={{ background: bg[i] }}>{icons[i]}</div>
                                    <span className="kpi-cell-rank">#{i + 1} · Champion</span>
                                </div>
                                <div className="kpi-cell-value" style={{ color: colors[i] }}>
                                    {(team.champion.p * 100).toFixed(1)}%
                                </div>
                                <div className="kpi-cell-name">{team.name}</div>
                                <div className="kpi-cell-sub">Elo {team.elo.toFixed(0)} · Group {team.group}</div>
                                {/* Hover expand — full tournament path */}
                                <div className="kpi-cell-expand">
                                    {STAGES.map(s => (
                                        <div key={s} className="kpi-expand-row">
                                            <span className="kpi-expand-stage">{STAGE_LABELS[s]}</span>
                                            <div className="kpi-expand-bar-wrap">
                                                <div className="kpi-expand-bar" style={{ width: `${(team[s].p * 100).toFixed(1)}%`, background: STAGE_COLORS[s] }} />
                                            </div>
                                            <span className="kpi-expand-pct">{(team[s].p * 100).toFixed(0)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )
                    })}
                </div>

                {/* ── Main grid ── */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1.6fr", gap: 0, borderBottom: "1px solid var(--border-bright)" }}>

                    {/* Left: group selector + team bars */}
                    <div style={{ borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column" }}>

                        <div style={{ padding: "16px 18px 0", borderBottom: "1px solid var(--border)" }}>
                            <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                                <span style={{ width: 3, height: 10, background: "var(--accent)", borderRadius: 2, display: "inline-block" }} />
                                Group Stage
                            </div>
                            <div className="group-selector">
                                {GROUPS.map(g => (
                                    <button key={g} className={`group-pill${activeGroup === g ? " active" : ""}`} onClick={() => setActiveGroup(g)}>
                                        {g}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)", padding: "12px 18px 6px", borderBottom: "1px solid var(--border)" }}>
                            Group {activeGroup} — advancement &amp; WC win
                        </div>

                        <div className="team-prob-list" style={{ flex: 1 }}>
                            {groupTeams.map((team, i) => (
                                <>
                                    <div
                                        key={team.name}
                                        className={`team-prob-row${expandedTeam === team.name ? " expanded" : ""}`}
                                        onClick={() => toggleTeam(team.name)}
                                    >
                                        <span className="team-prob-rank">{i + 1}</span>
                                        <div className="team-prob-name-wrap">
                                            <span className="team-prob-name">{team.name}</span>
                                            <div className="team-prob-bar-wrap">
                                                <div className="team-prob-bar" style={{ width: `${(team.r32.p * 100).toFixed(1)}%`, background: "var(--home)" }} />
                                            </div>
                                        </div>
                                        <span className="team-prob-pct" style={{ color: "var(--home)" }}>{(team.r32.p * 100).toFixed(0)}%<br />
                                            <span style={{ fontSize: ".56rem", color: "var(--text-3)", fontWeight: 400 }}>advance</span>
                                        </span>
                                        <span className="team-prob-pct" style={{ color: "var(--accent)" }}>
                                            {(team.champion.p * 100).toFixed(1)}%<br />
                                            <span style={{ fontSize: ".56rem", color: "var(--text-3)", fontWeight: 400 }}>WC win</span>
                                        </span>
                                    </div>
                                    {expandedTeam === team.name && (
                                        <div key={`${team.name}-expand`} className="team-prob-expand">
                                            <div className="team-path-grid">
                                                {STAGES.map(s => (
                                                    <div key={s} className="team-path-cell">
                                                        <span className="team-path-stage">{STAGE_LABELS[s]}</span>
                                                        <div className="team-path-bar-track">
                                                            <div className="team-path-bar-fill" style={{ width: `${(team[s].p * 100).toFixed(1)}%`, background: STAGE_COLORS[s] }} />
                                                        </div>
                                                        <span className="team-path-pct" style={{ color: STAGE_COLORS[s] }}>{(team[s].p * 100).toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </>
                            ))}
                        </div>

                    </div>

                    {/* Right: top 8 chart + match odds */}
                    <div style={{ display: "flex", flexDirection: "column" }}>

                        {/* Top 8 chart */}
                        <div style={{ padding: "16px 18px", borderBottom: "1px solid var(--border)" }}>
                            <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)", marginBottom: 12, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                    <span style={{ width: 3, height: 10, background: "var(--c-ai)", borderRadius: 2, display: "inline-block" }} />
                                    Win distribution — top 8
                                </span>
                                <span>MC output · 95% CI shown</span>
                            </div>
                            <div className="top8-chart-interactive">
                                {top8.map((team, i) => {
                                    const p = team.champion.p
                                    const barH = Math.max(4, (p / maxP) * 150)
                                    const col = i === 0 ? "var(--accent)"
                                        : i < 3 ? "var(--home)"
                                            : `rgba(155,108,247,${0.9 - i * 0.07})`
                                    return (
                                        <div key={team.name} className={`top8-col${selectedTop8 === team.name ? " selected" : ""}`}
                                            onClick={() => setSelectedTop8(t => t === team.name ? null : team.name)}>
                                            <div className="top8-bar-wrap">
                                                <div className="top8-bar" style={{ height: barH, background: col, color: col }} />
                                            </div>
                                            <span className="top8-pct">{(p * 100).toFixed(1)}%</span>
                                            <span className="top8-abbr">{team.name.slice(0, 3).toUpperCase()}</span>
                                        </div>
                                    )
                                })}
                            </div>
                            {selectedTop8 && (() => {
                                const t = top8.find(x => x.name === selectedTop8)!
                                return (
                                    <div style={{ marginTop: 12, padding: "10px 12px", background: "var(--bg-3)", borderRadius: "var(--r-md)", animation: "card-expand .15s ease" }}>
                                        <div style={{ fontWeight: 600, fontSize: ".78rem", marginBottom: 8 }}>{t.name} — full tournament path</div>
                                        <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 6 }}>
                                            {STAGES.map(s => (
                                                <div key={s} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                                                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".52rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)" }}>{STAGE_LABELS[s]}</span>
                                                    <div style={{ height: 3, background: "var(--bg-4)", borderRadius: 2, overflow: "hidden" }}>
                                                        <div style={{ height: "100%", width: `${(t[s].p * 100).toFixed(1)}%`, background: STAGE_COLORS[s], borderRadius: 2 }} />
                                                    </div>
                                                    <span style={{ fontFamily: "var(--font-mono)", fontSize: ".62rem", fontWeight: 500, color: STAGE_COLORS[s] }}>{(t[s].p * 100).toFixed(1)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )
                            })()}
                        </div>

                        {/* Per-match odds */}
                        <div style={{ flex: 1, padding: "14px 18px", overflow: "auto" }}>
                            <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
                                <span style={{ width: 3, height: 10, background: "var(--away)", borderRadius: 2, display: "inline-block" }} />
                                Market odds — per match
                            </div>
                            <PerMatchOddsPanel />
                        </div>

                    </div>
                </div>

                {/* ── Tournament table ── */}
                <div style={{ padding: "0 0 0" }}>
                    <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)", display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ width: 3, height: 10, background: "var(--amber)", borderRadius: 2, display: "inline-block" }} />
                            All 48 Teams — click rows to expand · click headers to sort
                        </div>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", color: "var(--text-3)" }}>
                            sorted by {STAGE_LABELS[sortStage]}
                        </span>
                    </div>
                    <div className="tt-wrap">
                        <table className="tt">
                            <thead>
                                <tr>
                                    <th className="tt-rank">#</th>
                                    <th style={{ textAlign: "left" }}>Team</th>
                                    <th>Grp</th>
                                    <th className="num">Elo</th>
                                    {STAGES.map(s => (
                                        <th key={s} className={`num${sortStage === s ? " sort-col" : ""}`} onClick={() => setSortStage(s)}>
                                            {STAGE_LABELS[s]}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {sorted.map((team, i) => (
                                    <>
                                        <tr key={team.name} className={expandedTeam === team.name ? "tt-expanded" : ""} onClick={() => toggleTeam(team.name)}>
                                            <td className="tt-rank">{i + 1}</td>
                                            <td className="tt-name">{team.name}</td>
                                            <td className="tt-group"><span className="group-badge">{team.group}</span></td>
                                            <td className="tt-elo tt-num">{team.elo.toFixed(0)}</td>
                                            {STAGES.map(s => {
                                                const p = team[s].p
                                                const hi = s === sortStage
                                                return (
                                                    <td key={s} className="tt-num">
                                                        <div className="tt-pbar">
                                                            <div className="tt-pbar-track">
                                                                <div className="tt-pbar-fill" style={{ width: `${(p * 100).toFixed(1)}%`, background: hi ? STAGE_COLORS[s] : "var(--border-light)" }} />
                                                            </div>
                                                            <span className={`tt-pbar-val${hi ? " hi" : ""}`}>{(p * 100).toFixed(1)}%</span>
                                                        </div>
                                                    </td>
                                                )
                                            })}
                                        </tr>
                                        {expandedTeam === team.name && (
                                            <tr key={`${team.name}-ex`} className="tt-expand-row">
                                                <td colSpan={4 + STAGES.length}>
                                                    <div className="tt-expand-inner">
                                                        {STAGES.map(s => (
                                                            <div key={s} className="tt-expand-cell">
                                                                <span className="tt-expand-stage">{STAGE_LABELS[s]}</span>
                                                                <div className="tt-expand-track">
                                                                    <div className="tt-expand-fill" style={{ width: `${(team[s].p * 100).toFixed(1)}%`, background: STAGE_COLORS[s] }} />
                                                                </div>
                                                                <span className="tt-expand-pct" style={{ color: STAGE_COLORS[s] }}>{(team[s].p * 100).toFixed(1)}%</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Bracket impact */}
                <div style={{ padding: "16px 18px", borderTop: "1px solid var(--border-bright)" }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", textTransform: "none", letterSpacing: "normal", color: "var(--text-3)", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ width: 3, height: 10, background: "var(--c-ai)", borderRadius: 2, display: "inline-block" }} />
                        Bracket impact — counterfactual analysis
                    </div>
                    <BracketImpactFeed />
                </div>

                <div className="predict-method" style={{ margin: "0 18px 0" }}>
                    <strong>Methodology —</strong> {prediction!.n_sims.toLocaleString()} independent full-tournament MC runs. Match probabilities from Elo ratings (draw decay), overridden by Shin-corrected market odds where available. Third-place advancement: best 8 of 12 groups. 95% CI: ±1.96√(p̂(1−p̂)/N).
                </div>

            </>)}

        </div>
    )
}