"use client"
// hero banner

import { useEffect } from "react"
import { Flag } from "@/components/Flag"
import { teamColor } from "@/lib/flag"

const STATUS_LABEL: Record<string, string> = {
    NS: "NOT STARTED", "1H": "1ST HALF", HT: "HALF TIME",
    "2H": "2ND HALF", ET: "EXTRA TIME", BT: "BREAK",
    P: "PENALTIES", FT: "FULL TIME", AET: "AET", PEN: "PENALTIES",
}
const LIVE = new Set(["1H", "2H", "ET", "P"])
// US Eastern, DST-aware.
const TZ = "America/New_York"

// page background: the side ahead on the scoreboard gets the dominant share of the
// gradient in its national color; level scores (0-0, genuine draw) split evenly.
// home is always left, away always right (matching the names/flags); only the
// transition point moves.
//
// a dark scrim sits on top, fading a constant color's (black) own alpha. blending black
// toward the opaque var(--bg-page) interpolates RGB through a muddy gray and nearly hid
// the score digits in light theme. an alpha-only fade composites cleanly, gives the top
// of the page more contrast than lower down, and does not follow the theme toggle.
//
// no fade to var(--bg-page): body uses background-attachment: fixed (see globals.css),
// so this paints over the full viewport at any scroll position.
//
// set as a CSS variable on <html> and painted by body (see globals.css) instead of as a
// card background, so the color fills the whole page as one surface.
function pageHeroLayer(homeName: string, awayName: string, homeScore: number, awayScore: number): string {
    const home = teamColor(homeName)
    const away = teamColor(awayName)

    const colorSplit = homeScore === awayScore
        ? `${home} 0%, ${home} 48%, ${away} 52%, ${away} 100%`
        : homeScore > awayScore
            ? `${home} 0%, ${home} 62%, ${away} 100%`
            : `${home} 0%, ${away} 38%, ${away} 100%`

    return [
        "linear-gradient(180deg, rgba(0, 0, 0, .38) 0%, rgba(0, 0, 0, .22) 30%, rgba(0, 0, 0, .12) 60%, rgba(0, 0, 0, .12) 100%)",
        `linear-gradient(120deg, ${colorSplit})`,
    ].join(", ")
}

// sets the page-wide hero background via a CSS custom property on <html>, so the whole
// viewport carries the match colors; reset on unmount to clear the tint on pages with
// no match
function usePageHeroBackground(homeName: string, awayName: string, homeScore: number, awayScore: number) {
    useEffect(() => {
        const root = document.documentElement
        root.style.setProperty("--page-hero-layer", pageHeroLayer(homeName, awayName, homeScore, awayScore))
        return () => {
            root.style.removeProperty("--page-hero-layer")
        }
    }, [homeName, awayName, homeScore, awayScore])
}

export function ScoreHeader({ state, updatedAt }: { state: any; updatedAt?: string }) {
    usePageHeroBackground(state.home_name, state.away_name, state.home_score, state.away_score)
    const isLive = LIVE.has(state.status_short)
    const label = STATUS_LABEL[state.status_short] ?? state.status_short
    const elapsed = state.elapsed ?? 0
    const maxMin = ["2H", "FT", "AET"].includes(state.status_short) ? 90 : 45
    const progress = state.status_short === "FT" ? 100
        : isLive ? Math.min(100, (elapsed / maxMin) * 100) : 0

    return (
        <div className="score-header-v3 score-header-v3-team-colored">
            {updatedAt && (
                <span style={{
                    position: "absolute", top: 6, right: 10,
                    fontFamily: "var(--font-mono)", fontSize: ".54rem",
                    color: "rgba(255,255,255,.6)",
                }}>
                    Updated {new Date(updatedAt).toLocaleTimeString("en-US", { timeZone: TZ })}
                </span>
            )}
            <div className="score-header-v3-inner">

                <div className="score-team-v3 home">
                    <div className="score-team-crest-row">
                        <Flag team={state.home_name} size="lg" />
                        <span className="score-team-name-v3">{state.home_name}</span>
                    </div>
                    {state.venue && <span className="score-team-sub">{state.venue}</span>}
                </div>

                <div className="score-center-v3">
                    <div className="score-digits-v3">
                        <span className="score-digit-v3">{state.home_score}</span>
                        <span className="score-dash-v3">–</span>
                        <span className="score-digit-v3">{state.away_score}</span>
                    </div>
                    <div className="score-status-v3">
                        {isLive && <span className="live-dot" />}
                        <span className="score-status-badge">{label}</span>
                        {isLive && elapsed > 0 && (
                            <span className="score-elapsed-v3">{elapsed}'</span>
                        )}
                    </div>
                </div>

                <div className="score-team-v3 away">
                    <div className="score-team-crest-row">
                        <span className="score-team-name-v3">{state.away_name}</span>
                        <Flag team={state.away_name} size="lg" />
                    </div>
                    {state.round && <span className="score-team-sub">{state.round}</span>}
                </div>

            </div>

            <div className="score-header-v3-timeline">
                <div className="score-header-v3-timeline-fill" style={{ width: `${progress.toFixed(1)}%` }} />
            </div>
        </div>
    )
}