"use client"


import Link from "next/link"
import { usePathname } from "next/navigation"
import { useEffect, useState } from "react"
import { ThemeToggle } from "@/components/ThemeToggle"

export function NavBar() {
    const pathname = usePathname()
    const [time, setTime] = useState("")
    const onMatchPage = pathname.startsWith("/match")

    useEffect(() => {
        const tick = () => setTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }))
        tick()
        const t = setInterval(tick, 1000)
        return () => clearInterval(t)
    }, [])

    return (
        <div className="wc-navbar">
            <div className="wc-navbar-left">
                {onMatchPage && (
                    <Link href="/" className="back-link">← Matches</Link>
                )}
            </div>

            <Link href="/" className="wc-navbar-logo">
                <img src="/IMG_6227.webp" alt="" className="wc-navbar-logo-img" />
                FIFA World Cup 2026
            </Link>

            <div className="wc-navbar-right">
                <ThemeToggle />
                <span className="wc-navbar-clock">{time}</span>
            </div>
        </div>
    )
}