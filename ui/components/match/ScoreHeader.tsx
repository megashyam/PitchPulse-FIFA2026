"use client"
// components/match/ScoreHeader.tsx — v3 hero banner

import { useEffect } from "react"
import type { MatchState } from "@/types/match"
import { Flag } from "@/components/Flag"
import { teamColor } from "@/lib/flag"

const STATUS_LABEL: Record<string, string> = {
    NS: "NOT STARTED", "1H": "1ST HALF", HT: "HALF TIME",
    "2H": "2ND HALF", ET: "EXTRA TIME", BT: "BREAK",
    P: "PENALTIES", FT: "FULL TIME", AET: "AET", PEN: "PENALTIES",
}
const LIVE = new Set(["1H", "2H", "ET", "P"])
// US Eastern — correctly follows DST (audit H6 fix; was a fixed "Etc/GMT+5"
// offset, which is only correct during EST/winter months).
const TZ = "America/New_York"

// Page background: whichever side is ahead on the scoreboard gets the
// dominant share of the gradient in their national color; level scores
// (0-0, or a genuine draw) get an even split instead of picking a side.
// home is always anchored left, away always anchored right (matching where
// their names/flags render) — only the transition POINT moves to reflect
// who's ahead, never which side each color occupies.
//
// A dark scrim sits on top of the color split, fading a CONSTANT color's
// (black) own alpha down rather than interpolating between two different
// colors — mixing e.g. translucent black toward an opaque var(--bg-page)
// directly (a single gradient stop going from rgba(0,0,0,.6) to an opaque
// light color) interpolates RGB channels too, which passes through a muddy
// gray band along the way and nearly swallowed the score digits in light
// theme. An alpha-only fade composites cleanly instead, and gives the top
// of the page (behind the header text) more contrast than lower down —
// doesn't follow the light/dark theme toggle, same as a photo-hero would
// stay dark-tinted in either theme.
//
// No fade to var(--bg-page): body uses background-attachment:fixed (see
// globals.css), so this paints fresh over the full viewport regardless of
// scroll position — the color needs to reach every edge of the screen, not
// fade out partway down before the actual page content ends.
//
// Set as a CSS variable on <html> and painted by `body` (see globals.css)
// rather than as this component's own background, so the color fills the
// full page width/height as one continuous surface instead of being boxed
// into a card — the header's text just sits on top of it.
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

// Sets the page-wide hero background via a CSS custom property on <html>,
// so the whole viewport — not just a bounded header card — carries the
// match's colors. Reset on unmount so leaving the match page doesn't leave
// a stray tint on pages with no match context.
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