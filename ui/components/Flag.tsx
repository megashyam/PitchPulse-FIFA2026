"use client"
// components/Flag.tsx — image flag with letter fallback
import { flagUrl } from "@/lib/flag"

interface Props { team: string; size?: "sm" | "md" | "lg" | "xl" }
const PX = { sm: 24, md: 36, lg: 56, xl: 72 }

export function Flag({ team, size = "md" }: Props) {
    const url = flagUrl(team, size === "xl" || size === "lg" ? "80" : "40")
    const px = PX[size]
    if (!url) {
        return (
            <div className={`v4-flag-placeholder v4-flag-${size}`}
                style={{ width: px, height: px, fontSize: px * 0.34 }}>
                {team.slice(0, 2).toUpperCase()}
            </div>
        )
    }
    return <img src={url} alt={team} className={`v4-flag v4-flag-${size}`} loading="lazy" />
}