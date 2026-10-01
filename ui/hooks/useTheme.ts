"use client"
// hooks/useTheme.ts
// Light/dark theme toggle — persists to localStorage, applies
// data-theme="light"|"dark" to <html>, defaults to system preference
// on first visit, falls back to dark if unavailable.

import { useEffect, useState, useCallback } from "react"

type Theme = "light" | "dark"

const STORAGE_KEY = "wc2026-theme"

function getInitialTheme(): Theme {
    if (typeof window === "undefined") return "dark"
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === "light" || stored === "dark") return stored
    const prefersLight = window.matchMedia?.("(prefers-color-scheme: light)").matches
    return prefersLight ? "light" : "dark"
}

export function useTheme() {
    const [theme, setThemeState] = useState<Theme>("dark")
    const [mounted, setMounted] = useState(false)

    useEffect(() => {
        const initial = getInitialTheme()
        setThemeState(initial)
        document.documentElement.setAttribute("data-theme", initial)
        setMounted(true)
    }, [])

    const setTheme = useCallback((t: Theme) => {
        setThemeState(t)
        document.documentElement.setAttribute("data-theme", t)
        localStorage.setItem(STORAGE_KEY, t)
    }, [])

    const toggleTheme = useCallback(() => {
        setTheme(theme === "dark" ? "light" : "dark")
    }, [theme, setTheme])

    return { theme, setTheme, toggleTheme, mounted }
}
