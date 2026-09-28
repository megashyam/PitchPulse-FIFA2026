"use client"
// components/match/LineupCard.tsx
// Compact vertical formation widget for the narrow left stats column.
// One team at a time (tab toggle) — a horizontal pitch doesn't fit this
// width, so players are grouped into positional lines (attackers at top,
// GK at the bottom) and centered in wrapping rows instead.

import { useEffect, useState } from "react"
import { Flag } from "@/components/Flag"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface Player { number: number; name: string; position: string; line?: "G" | "D" | "M" | "F"; grid?: string; photo?: string | null }
interface TeamLineup { team: string; formation: string; startingXI: Player[]; coach?: string | null }
interface LineupData { home: TeamLineup; away: TeamLineup; source: "espn" | "api-sports" | "zafronix_squad" | "unavailable"; projected?: boolean }

const ROLES: Record<string, string[]> = {
    "4-3-3": ["GK", "LB", "CB", "CB", "RB", "LCM", "CM", "RCM", "LW", "ST", "RW"],
    "4-4-2": ["GK", "LB", "CB", "CB", "RB", "LM", "LCM", "RCM", "RM", "ST", "ST"],
    "4-2-3-1": ["GK", "LB", "CB", "CB", "RB", "CDM", "CDM", "LM", "CAM", "RM", "ST"],
    "3-5-2": ["GK", "CB", "CB", "CB", "LWB", "CM", "CM", "CM", "RWB", "ST", "ST"],
    "5-3-2": ["GK", "LWB", "CB", "CB", "CB", "RWB", "LCM", "CM", "RCM", "ST", "ST"],
    "4-1-4-1": ["GK", "LB", "CB", "CB", "RB", "CDM", "LM", "LCM", "RCM", "RM", "ST"],
    "3-4-3": ["GK", "CB", "CB", "CB", "LM", "LCM", "RCM", "RM", "LW", "ST", "RW"],
    "3-4-2-1": ["GK", "CB", "CB", "CB", "LM", "LCM", "RCM", "RM", "SS", "SS", "ST"],
}

const DEF_ROLES = new Set(["CB", "LB", "RB", "LWB", "RWB"])
const MID_ROLES = new Set(["CDM", "CM", "LCM", "RCM", "LM", "RM", "CAM"])

type Line = "ATT" | "MID" | "DEF" | "GK"
const LINE_ORDER: Line[] = ["ATT", "MID", "DEF", "GK"]
const LINE_LABEL: Record<Line, string> = { ATT: "Attack", MID: "Midfield", DEF: "Defence", GK: "Goalkeeper" }

function lineOf(role: string): Line {
    if (role === "GK") return "GK"
    if (DEF_ROLES.has(role)) return "DEF"
    if (MID_ROLES.has(role)) return "MID"
    return "ATT"
}

function shortName(n: string): string {
    if (!n) return ""
    const p = n.trim().split(" ")
    return p.length === 1 ? n : p[p.length - 1]
}

const LINE_OF_CODE: Record<string, Line> = { G: "GK", D: "DEF", M: "MID", F: "ATT" }

function groupByLine(lineup: TeamLineup): Record<Line, { player?: Player; role: string; idx: number }[]> {
    // Confirmed ESPN lineups tag each player's line; use it directly so any
    // formation (4-1-3-2, 3-4-2-1, ...) renders from real positions.
    if (lineup.startingXI.length && lineup.startingXI.every(p => p.line)) {
        const groups: Record<Line, { player?: Player; role: string; idx: number }[]> = { ATT: [], MID: [], DEF: [], GK: [] }
        lineup.startingXI.forEach((p, i) => groups[LINE_OF_CODE[p.line!]].push({ player: p, role: p.position, idx: i }))
        return groups
    }
    const roles = ROLES[lineup.formation] ?? ROLES["4-3-3"]
    const groups: Record<Line, { player?: Player; role: string; idx: number }[]> = { ATT: [], MID: [], DEF: [], GK: [] }
    roles.forEach((role, i) => {
        groups[lineOf(role)].push({ player: lineup.startingXI[i], role, idx: i })
    })
    return groups
}

function PlayerChip({ entry, accent, idKey }: {
    entry: { player?: Player; role: string; idx: number }
    accent: string
    idKey: string
}) {
    const [imgFailed, setImgFailed] = useState(false)
    const { player, role, idx } = entry
    const hasPhoto = !!player?.photo && !imgFailed
    const num = player ? player.number : idx + 1
    const label = player ? shortName(player.name) : role

    return (
        <div className="lineup-chip" key={idKey}>
            <div className="lineup-avatar" style={{ borderColor: accent }}>
                {hasPhoto ? (
                    <img
                        src={player!.photo!}
                        alt={player!.name}
                        className="lineup-avatar-img"
                        onError={() => setImgFailed(true)}
                    />
                ) : (
                    <span className="lineup-avatar-num" style={{ color: accent }}>{num}</span>
                )}
                {hasPhoto && (
                    <span className="lineup-avatar-badge" style={{ background: accent }}>{num}</span>
                )}
            </div>
            <span className="lineup-chip-name">{label}</span>
        </div>
    )
}

interface Props { fixtureId: string; homeTeam: string; awayTeam: string }

export function LineupCard({ fixtureId, homeTeam, awayTeam }: Props) {
    const [data, setData] = useState<LineupData | null>(null)
    const [loading, setLoading] = useState(true)
    const [activeTeam, setActiveTeam] = useState<"home" | "away">("home")

    useEffect(() => {
        if (!fixtureId) return
        const load = () =>
            fetch(`${API}/matches/${fixtureId}/lineups`)
                .then(r => r.ok ? r.json() : null)
                .then(d => { if (d) setData(d) })
                .catch(() => { })
                .finally(() => setLoading(false))
        load()
        const t = setInterval(load, 60_000)
        return () => clearInterval(t)
    }, [fixtureId])

    const home: TeamLineup = data?.home ?? { team: homeTeam, formation: "", startingXI: [] }
    const away: TeamLineup = data?.away ?? { team: awayTeam, formation: "", startingXI: [] }
    const isConfirmed = data?.source === "espn" || data?.source === "api-sports"
    const isProjected = data?.source === "zafronix_squad"
    const homeAbbr = homeTeam.slice(0, 3).toUpperCase()
    const awayAbbr = awayTeam.slice(0, 3).toUpperCase()

    const active = activeTeam === "home" ? home : away
    const activeName = activeTeam === "home" ? homeTeam : awayTeam
    const accent = activeTeam === "home" ? "var(--home)" : "var(--away)"
    const groups = groupByLine(active)
    const totalPlayers = home.startingXI.length + away.startingXI.length

    const sourceLabel = isConfirmed ? "confirmed" : isProjected ? "projected XI" : loading ? "loading…" : null

    return (
        <div className="lineup-sidebar">
            <div className="lineup-tabs">
                <button
                    className={`lineup-tab-btn${activeTeam === "home" ? " active" : ""}`}
                    onClick={() => setActiveTeam("home")}
                    style={activeTeam === "home" ? { color: "var(--home)", borderColor: "var(--home)" } : undefined}
                >
                    <Flag team={homeTeam} size="sm" />
                    {homeAbbr}
                </button>
                <button
                    className={`lineup-tab-btn${activeTeam === "away" ? " active" : ""}`}
                    onClick={() => setActiveTeam("away")}
                    style={activeTeam === "away" ? { color: "var(--away)", borderColor: "var(--away)" } : undefined}
                >
                    <Flag team={awayTeam} size="sm" />
                    {awayAbbr}
                </button>
            </div>

            <div className="lineup-meta">
                <span className="lineup-meta-formation">{active.formation}</span>
                {sourceLabel && <span className="lineup-meta-source">{sourceLabel}</span>}
            </div>

            <div className="lineup-stack">
                {LINE_ORDER.map(line => (
                    groups[line].length > 0 && (
                        <div className="lineup-line" key={line}>
                            <span className="lineup-line-label">{LINE_LABEL[line]}</span>
                            <div className="lineup-line-row">
                                {groups[line].map(entry => (
                                    <PlayerChip
                                        key={entry.idx}
                                        entry={entry}
                                        accent={accent}
                                        idKey={`${activeTeam}-${entry.idx}`}
                                    />
                                ))}
                            </div>
                        </div>
                    )
                ))}
            </div>

            {active.coach && (
                <div className="lineup-coach">
                    <span className="lineup-coach-label">Coach</span>
                    <span className="lineup-coach-name">{active.coach}</span>
                </div>
            )}

            <div className="lineup-footer">
                {isConfirmed
                    ? `Confirmed XI via ${data?.source === "espn" ? "ESPN" : "API-Sports"} · ${totalPlayers} players`
                    : isProjected
                        ? `Projected from the 2026 squad — not the confirmed XI`
                        : "Lineups not available"}
            </div>

            <style jsx global>{`
                .lineup-sidebar {
                    background: var(--glass-bg-inner);
                    border: 1px solid var(--glass-border-inner);
                    border-radius: var(--r-md);
                    backdrop-filter: var(--glass-blur);
                    -webkit-backdrop-filter: var(--glass-blur);
                    margin: 8px;
                    overflow: hidden;
                    flex-shrink: 0;
                }
                .lineup-tabs {
                    display: flex;
                    border-bottom: 1px solid var(--glass-border-inner);
                }
                .lineup-tab-btn {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    padding: 10px 6px;
                    background: transparent;
                    border: none;
                    border-bottom: 2px solid transparent;
                    font-family: var(--font-mono);
                    font-size: .68rem;
                    font-weight: 700;
                    color: var(--text-3);
                    cursor: pointer;
                    transition: color .12s, border-color .12s;
                }
                .lineup-tab-btn.active {
                    background: var(--glass-bg-inner);
                }
                .lineup-meta {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    padding: 8px 12px;
                    border-bottom: 1px solid var(--glass-border-inner);
                }
                .lineup-meta-formation {
                    font-family: var(--font-mono);
                    font-size: .68rem;
                    font-weight: 700;
                    color: var(--text-1);
                }
                .lineup-meta-source {
                    font-family: var(--font-mono);
                    font-size: .56rem;
                    color: var(--text-2);
                }
                .lineup-stack {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    padding: 14px 12px;
                    background: #1b4d2e;
                }
                .lineup-line {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                .lineup-line-label {
                    font-family: var(--font-mono);
                    font-size: .5rem;
                    text-transform: none;
                    letter-spacing: normal;
                    color: rgba(255, 255, 255, .55);
                    text-align: center;
                }
                .lineup-line-row {
                    display: flex;
                    flex-direction: row;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 10px;
                }
                .lineup-chip {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 3px;
                    width: 46px;
                }
                .lineup-avatar {
                    position: relative;
                    width: 40px;
                    height: 40px;
                    flex-shrink: 0;
                    border-radius: 50%;
                    border: 1.5px solid;
                    background: var(--bg-3);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                }
                .lineup-avatar-img {
                    width: 40px;
                    height: 40px;
                    object-fit: cover;
                    border-radius: 50%;
                    display: block;
                }
                .lineup-avatar-num {
                    font-family: var(--font-mono);
                    font-size: .74rem;
                    font-weight: 700;
                }
                .lineup-avatar-badge {
                    position: absolute;
                    bottom: -2px;
                    right: -2px;
                    min-width: 14px;
                    height: 14px;
                    padding: 0 2px;
                    border-radius: 7px;
                    border: 1px solid var(--bg-1, #0b0f19);
                    font-family: var(--font-mono);
                    font-size: .48rem;
                    font-weight: 700;
                    color: #fff;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .lineup-chip-name {
                    width: 100%;
                    font-size: .56rem;
                    color: rgba(255, 255, 255, .85);
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                .lineup-coach {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                    padding: 8px 12px;
                    border-top: 1px solid var(--glass-border-inner);
                    font-size: .68rem;
                }
                .lineup-coach-label {
                    font-family: var(--font-mono);
                    font-size: .52rem;
                    text-transform: none;
                    letter-spacing: normal;
                    color: var(--text-3);
                }
                .lineup-coach-name {
                    color: var(--text-2);
                }
                .lineup-footer {
                    padding: 8px 12px;
                    border-top: 1px solid var(--glass-border-inner);
                    font-family: var(--font-mono);
                    font-size: .54rem;
                    color: var(--text-2);
                    text-align: center;
                }
            `}</style>
        </div>
    )
}
