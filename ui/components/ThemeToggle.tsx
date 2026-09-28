"use client"
// components/ThemeToggle.tsx
// Sliding pill switch — drop into NavBar (or anywhere) to toggle theme.

import { useTheme } from "@/hooks/useTheme"

export function ThemeToggle() {
    const { theme, toggleTheme, mounted } = useTheme()

    // Avoid a flash of wrong state before the client has determined the
    // real theme on first mount
    if (!mounted) return <div style={{ width: 46, height: 24 }} />

    const isDark = theme === "dark"

    return (
        <button
            onClick={toggleTheme}
            aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
            title={`Switch to ${isDark ? "light" : "dark"} mode`}
            role="switch"
            aria-checked={isDark}
            style={{
                width: 46, height: 24, borderRadius: 999, padding: 3,
                display: "flex", alignItems: "center",
                justifyContent: isDark ? "flex-end" : "flex-start",
                background: isDark ? "var(--bg-4)" : "var(--accent-dim)",
                border: `1px solid ${isDark ? "var(--border-bright)" : "var(--accent-glow)"}`,
                cursor: "pointer", transition: "background .2s ease, border-color .2s ease",
                flexShrink: 0,
            }}
        >
            <span
                style={{
                    width: 18, height: 18, borderRadius: "50%",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    background: isDark ? "var(--text-1)" : "var(--accent)",
                    color: isDark ? "var(--bg-2)" : "#04211a",
                    fontSize: ".62rem", lineHeight: 1,
                    transition: "transform .2s ease, background .2s ease",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
                }}
            >
                {isDark ? "🌙" : "☀️"}
            </span>
        </button>
    )
}
