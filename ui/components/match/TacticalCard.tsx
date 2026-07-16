"use client"

import type { MatchState } from "@/types/match"
import { useTactical, type TeamFingerprint } from "@/hooks/useTactical"

function FingerprintNarrative({ homeName, awayName, fp }: { homeName: string; awayName: string; fp: TeamFingerprint }) {
    const m = fp.match
    return (
        <div className="tactical-narrative">
            <div className="tactical-narrative-head">
                <span className="tactical-narrative-title">
                    {homeName} vs {awayName} · FIFA WC {m.season}
                </span>
            </div>
            <p className="tactical-narrative-quote">"{m.content}"</p>
            <div className="tactical-narrative-zones">
                <div>
                    <span className="tactical-narrative-zone-label">Overall</span>
                    <span className="tactical-narrative-zone-val">{m.ppda.toFixed(1)}</span>
                </div>
                <div>
                    <span className="tactical-narrative-zone-label">Mid 3rd</span>
                    <span className="tactical-narrative-zone-val">{m.ppda_mid_third.toFixed(1)}</span>
                </div>
                <div>
                    <span className="tactical-narrative-zone-label">Att 3rd</span>
                    <span className="tactical-narrative-zone-val">{m.ppda_att_third.toFixed(1)}</span>
                </div>
                <div>
                    <span className="tactical-narrative-zone-label">Press intensity</span>
                    <span className="tactical-narrative-zone-val">{m.press_intensity.toFixed(2)}</span>
                </div>
            </div>

            <style jsx>{`
                .tactical-narrative {
                    padding: 12px 18px 14px;
                    border-top: 1px solid var(--border);
                }
                .tactical-narrative-head {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-family: var(--font-mono);
                    font-size: .68rem;
                    margin-bottom: 6px;
                }
                .tactical-narrative-title {
                    font-weight: 700;
                    color: var(--text-1);
                }
                .tactical-narrative-quote {
                    font-size: .78rem;
                    line-height: 1.55;
                    color: var(--text-2);
                    font-style: italic;
                    white-space: pre-line;
                    margin: 0 0 10px;
                }
                .tactical-narrative-zones {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 8px;
                }
                .tactical-narrative-zones > div {
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                }
                .tactical-narrative-zone-label {
                    font-family: var(--font-mono);
                    font-size: .52rem;
                    text-transform: uppercase;
                    letter-spacing: .06em;
                    color: var(--text-3);
                }
                .tactical-narrative-zone-val {
                    font-family: var(--font-mono);
                    font-size: .8rem;
                    font-weight: 600;
                    color: var(--text-1);
                }
            `}</style>
        </div>
    )
}

export function TacticalCard({ state, fixtureId }: { state: MatchState; fixtureId: string }) {
    const { tactical } = useTactical(fixtureId)
    const h = state.home_stats; const a = state.away_stats
    const hasPoss = h.possession > 0 || a.possession > 0
    const totalViz = (h.possession + a.possession) || 100
    const homeZonePct = (h.possession / totalViz) * 100
    const homeDominant = homeZonePct >= 50
    const homeAbbr = state.home_name.slice(0, 3).toUpperCase()
    const awayAbbr = state.away_name.slice(0, 3).toUpperCase()
    const fp = tactical?.home ?? null
    const fpAway = tactical?.away ?? null
    const ppdaDisplay = fp ? fp.match.ppda.toFixed(1)
        : h.shots_total > 0 ? (a.passes_total / h.shots_total).toFixed(1) : "—"

    const hasFingerprint = !!fp && fp.match.match_pct != null
    const hasAwayFingerprint = !!fpAway && fpAway.match.match_pct != null

    return (
        <div>

            <div className="tactical-2col">
                <div className="tactical-col-left">


                    <div className="tactical-pitch-v2">
                        <div className="tactical-pitch-circle" />
                        <div className="tactical-pitch-box-home" />
                        <div className="tactical-pitch-box-away" />


                        <div
                            className="tactical-zone-overlay home"
                            style={{
                                width: hasPoss ? `${homeZonePct.toFixed(1)}%` : "50%",
                                background: homeDominant
                                    ? "rgba(79,134,247,.28)"
                                    : "rgba(79,134,247,.12)",
                            }}
                        >
                            <span className="tactical-zone-label">{homeAbbr}</span>
                        </div>


                        <div
                            className="tactical-zone-overlay away"
                            style={{
                                width: hasPoss ? `${(100 - homeZonePct).toFixed(1)}%` : "50%",
                                background: !homeDominant
                                    ? "rgba(240,84,84,.28)"
                                    : "rgba(240,84,84,.12)",
                            }}
                        >
                            <span className="tactical-zone-label">{awayAbbr}</span>
                        </div>
                    </div>

                    {/* 2-column stat grid — compressed (smaller padding/font)
                        to free up vertical room. Scoped styles so this
                        doesn't touch the global tactical-stat-* classes used
                        elsewhere. */}
                    <div className="tactical-stats-grid tactical-stats-grid-compact">

                        <div className="tactical-stat-cell">
                            <span className="tactical-stat-cell-label">Possession</span>
                            <span className="tactical-stat-cell-value" style={{ color: "var(--home)" }}>
                                {hasPoss ? `${h.possession.toFixed(0)}%` : "—"}
                            </span>
                        </div>

                        <div className="tactical-stat-cell">
                            <span className="tactical-stat-cell-label">Away Poss.</span>
                            <span className="tactical-stat-cell-value" style={{ color: "var(--away)" }}>
                                {hasPoss ? `${a.possession.toFixed(0)}%` : "—"}
                            </span>
                        </div>

                        <div className="tactical-stat-cell">
                            <span className="tactical-stat-cell-label">Pass Acc. (H)</span>
                            <span className="tactical-stat-cell-value">
                                {h.pass_accuracy > 0 ? `${h.pass_accuracy.toFixed(0)}%` : "—"}
                            </span>
                        </div>

                        <div className="tactical-stat-cell">
                            <span className="tactical-stat-cell-label">Pass Acc. (A)</span>
                            <span className="tactical-stat-cell-value">
                                {a.pass_accuracy > 0 ? `${a.pass_accuracy.toFixed(0)}%` : "—"}
                            </span>
                        </div>

                        <div className="tactical-stat-cell">
                            <span className="tactical-stat-cell-label">PPDA</span>
                            <span className="tactical-stat-cell-value">{ppdaDisplay}</span>
                        </div>

                        <div className="tactical-stat-cell">
                            <span className="tactical-stat-cell-label">Source</span>
                            <span className="tactical-stat-cell-value" style={{ fontSize: ".72rem", color: "var(--c-ai)" }}>
                                {fp ? "Weaviate" : "Live proxy"}
                            </span>
                        </div>

                    </div>

                    {/* Fingerprint match summary — lives with the raw stats
                        on the left; only the prose narratives (below) are
                        on the right. */}
                    <div className="tactical-fingerprint-block">
                        <div>
                            <div className="tactical-fp-label">Closest tactical profile</div>
                            {!fp && (
                                <div style={{ fontSize: ".7rem", color: "var(--text-3)", marginTop: 3 }}>
                                    Run tactical indexer to enable cosine matching
                                </div>
                            )}
                        </div>
                        {hasFingerprint ? (
                            <div className="tactical-fp-match">
                                <div style={{ textAlign: "right" }}>
                                    <div className="tactical-fp-team">{homeAbbr} ≈ {fp!.match.team}</div>
                                    <div style={{ fontFamily: "var(--font-mono)", fontSize: ".58rem", color: "var(--text-3)" }}>
                                        {fp!.match.competition} {fp!.match.season}
                                    </div>
                                </div>
                                <div className="tactical-fp-pct">{fp!.match.match_pct}%</div>
                            </div>
                        ) : (
                            <div style={{ fontFamily: "var(--font-mono)", fontSize: ".68rem", color: "var(--text-3)" }}>
                                {homeDominant ? "High press" : "Low block"}
                                <span style={{ marginLeft: 6, color: "var(--text-3)", fontSize: ".6rem" }}>proxy</span>
                            </div>
                        )}
                    </div>
                </div>

                <div className="tactical-col-right">

                    {/* Narrative — the real, already-computed descriptor text
                        behind each fingerprint match (built from actual
                        StatsBomb PPDA/xG/pressure counts at index time, not
                        LLM-generated), plus the per-zone PPDA values the
                        backend already returns but the card never showed. */}
                    {hasFingerprint && (
                        <FingerprintNarrative homeName={state.home_name} awayName={state.away_name} fp={fp!} />
                    )}
                    {hasAwayFingerprint && (
                        <FingerprintNarrative homeName={state.home_name} awayName={state.away_name} fp={fpAway!} />
                    )}
                </div>
            </div>

            <style jsx>{`
                .tactical-2col {
                    display: grid;
                    grid-template-columns: 1fr 1.3fr;
                    align-items: start;
                }
                .tactical-col-left {
                    border-right: 1px solid var(--border);
                    min-width: 0;
                }
                .tactical-col-right {
                    min-width: 0;
                }
                @media (max-width: 720px) {
                    .tactical-2col {
                        grid-template-columns: 1fr;
                    }
                    .tactical-col-left {
                        border-right: none;
                        border-bottom: 1px solid var(--border);
                    }
                }
                .tactical-stats-grid-compact {
                    padding: 8px 10px !important;
                    gap: 4px 10px !important;
                }
                .tactical-stats-grid-compact :global(.tactical-stat-cell) {
                    padding: 4px 0 !important;
                }
                .tactical-stats-grid-compact :global(.tactical-stat-cell-label) {
                    font-size: .58rem !important;
                    margin-bottom: 1px !important;
                }
                .tactical-stats-grid-compact :global(.tactical-stat-cell-value) {
                    font-size: .82rem !important;
                }
            `}</style>

        </div>
    )
}