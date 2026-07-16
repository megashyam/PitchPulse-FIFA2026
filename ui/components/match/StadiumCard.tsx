"use client"



import { useEffect, useState } from "react"

interface StadiumInfo {
    fullName: string
    city: string
    country: string
    capacity: number
    surface: "Grass" | "FieldTurf" | "Hybrid Grass"
    lat: number
    lon: number
}

const STADIUMS: Record<string, StadiumInfo> = {
    "MetLife Stadium": { fullName: "MetLife Stadium", city: "East Rutherford, NJ", country: "USA", capacity: 82500, surface: "Grass", lat: 40.8135, lon: -74.0745 },
    "AT&T Stadium": { fullName: "AT&T Stadium", city: "Arlington, TX", country: "USA", capacity: 80000, surface: "FieldTurf", lat: 32.7473, lon: -97.0945 },
    "SoFi Stadium": { fullName: "SoFi Stadium", city: "Inglewood, CA", country: "USA", capacity: 70240, surface: "Grass", lat: 33.9535, lon: -118.3392 },
    "Levi's Stadium": { fullName: "Levi's Stadium", city: "Santa Clara, CA", country: "USA", capacity: 68500, surface: "Grass", lat: 37.4033, lon: -121.9694 },
    "Allegiant Stadium": { fullName: "Allegiant Stadium", city: "Las Vegas, NV", country: "USA", capacity: 65000, surface: "FieldTurf", lat: 36.0909, lon: -115.1833 },
    "Arrowhead Stadium": { fullName: "Arrowhead Stadium", city: "Kansas City, MO", country: "USA", capacity: 76416, surface: "Grass", lat: 39.0489, lon: -94.4839 },
    "NRG Stadium": { fullName: "NRG Stadium", city: "Houston, TX", country: "USA", capacity: 72220, surface: "FieldTurf", lat: 29.6847, lon: -95.4107 },
    "Empower Field": { fullName: "Empower Field at Mile High", city: "Denver, CO", country: "USA", capacity: 76125, surface: "Grass", lat: 39.7439, lon: -105.0201 },
    "Lincoln Financial Field": { fullName: "Lincoln Financial Field", city: "Philadelphia, PA", country: "USA", capacity: 69176, surface: "FieldTurf", lat: 39.9007, lon: -75.1675 },
    "Bank of America Stadium": { fullName: "Bank of America Stadium", city: "Charlotte, NC", country: "USA", capacity: 74867, surface: "Grass", lat: 35.2258, lon: -80.8529 },
    "Lumen Field": { fullName: "Lumen Field", city: "Seattle, WA", country: "USA", capacity: 67000, surface: "FieldTurf", lat: 47.5952, lon: -122.3316 },
    "BC Place": { fullName: "BC Place", city: "Vancouver, BC", country: "Canada", capacity: 54500, surface: "FieldTurf", lat: 49.2768, lon: -123.1115 },
    "BMO Field": { fullName: "BMO Field", city: "Toronto, ON", country: "Canada", capacity: 45736, surface: "FieldTurf", lat: 43.6333, lon: -79.4186 },
    "Estadio Azteca": { fullName: "Estadio Azteca", city: "Mexico City", country: "Mexico", capacity: 87523, surface: "Grass", lat: 19.3029, lon: -99.1505 },
    "Estadio Akron": { fullName: "Estadio Akron", city: "Guadalajara", country: "Mexico", capacity: 48071, surface: "Grass", lat: 20.6857, lon: -103.4669 },
    "Estadio BBVA": { fullName: "Estadio BBVA", city: "Monterrey", country: "Mexico", capacity: 53500, surface: "Grass", lat: 25.6694, lon: -100.4661 },
}

function findStadium(venue: string): StadiumInfo | null {
    if (!venue) return null
    if (STADIUMS[venue]) return STADIUMS[venue]
    const lower = venue.toLowerCase()
    for (const [key, info] of Object.entries(STADIUMS)) {
        if (lower.includes(key.toLowerCase()) || key.toLowerCase().includes(lower))
            return info
    }
    return null
}

interface WeatherData { temp_c: number; temp_f: number; icon: string; desc: string }

const WMO_ICONS: Record<number, { icon: string; desc: string }> = {
    0: { icon: "☀️", desc: "Clear" },
    1: { icon: "🌤", desc: "Mostly Clear" },
    2: { icon: "⛅️", desc: "Partly Cloudy" },
    3: { icon: "☁️", desc: "Overcast" },
    45: { icon: "🌫", desc: "Foggy" },
    51: { icon: "🌦", desc: "Light Drizzle" },
    61: { icon: "🌧", desc: "Light Rain" },
    63: { icon: "🌧", desc: "Rain" },
    80: { icon: "🌦", desc: "Showers" },
    95: { icon: "⛈", desc: "Thunderstorm" },
}
function wmoIcon(code: number) {
    return WMO_ICONS[code] ?? WMO_ICONS[Math.floor(code / 10) * 10] ?? { icon: "🌡", desc: "Unknown" }
}

interface Props { venue: string; round?: string }

export function StadiumCard({ venue, round }: Props) {
    const stadium = findStadium(venue)
    const [weather, setWeather] = useState<WeatherData | null>(null)

    useEffect(() => {
        if (!stadium) return
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${stadium.lat}&longitude=${stadium.lon}&current_weather=true&temperature_unit=fahrenheit`
        fetch(url, { cache: "no-store" })
            .then(r => r.json())
            .then(d => {
                const cw = d.current_weather
                const meta = wmoIcon(cw.weathercode)
                setWeather({
                    temp_f: Math.round(cw.temperature),
                    temp_c: Math.round((cw.temperature - 32) * 5 / 9),
                    icon: meta.icon,
                    desc: meta.desc,
                })
            })
            .catch(() => { })
    }, [stadium?.fullName])

    if (!stadium) {
        if (!venue) return null
        return (
            <div style={{
                background: "var(--bg-2)", border: "1px solid var(--border)",
                borderRadius: "var(--r-lg)", padding: "12px 14px"
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    <span>🏟</span>
                    <span style={{ fontSize: ".84rem", fontWeight: 700, color: "var(--text-1)" }}>{venue}</span>
                </div>
                <div style={{ fontSize: ".72rem", color: "var(--text-3)", fontStyle: "italic" }}>
                    Venue details not available
                </div>
            </div>
        )
    }

    return (
        <div style={{
            background: "var(--bg-2)", border: "1px solid var(--border)",
            borderTop: "2px solid var(--c-data)", borderRadius: "var(--r-lg)",
            overflow: "hidden"
        }}>

            <div style={{ padding: "12px 14px 10px", borderBottom: "1px solid var(--border)" }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
                    <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                            <span style={{ fontSize: ".9rem" }}>📍</span>
                            <span style={{ fontSize: ".86rem", fontWeight: 700, color: "var(--text-1)" }}>
                                {stadium.fullName}
                            </span>
                        </div>
                        <div style={{ fontSize: ".72rem", color: "var(--text-3)", paddingLeft: 22 }}>
                            {stadium.city}, {stadium.country}
                        </div>
                    </div>
                </div>
            </div>

            <div style={{ padding: "10px 14px" }}>

                <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "7px 0", borderBottom: "1px solid var(--border)"
                }}>
                    <span style={{
                        display: "flex", alignItems: "center", gap: 7,
                        fontSize: ".76rem", color: "var(--text-2)"
                    }}>
                        <span>👥</span> Capacity
                    </span>
                    <span style={{
                        fontFamily: "var(--font-mono)", fontSize: ".78rem",
                        fontWeight: 600, color: "var(--text-1)"
                    }}>
                        {stadium.capacity.toLocaleString()}
                    </span>
                </div>

                <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "7px 0", borderBottom: "1px solid var(--border)"
                }}>
                    <span style={{
                        display: "flex", alignItems: "center", gap: 7,
                        fontSize: ".76rem", color: "var(--text-2)"
                    }}>
                        <span>🌿</span> Surface
                    </span>
                    <span style={{
                        fontFamily: "var(--font-mono)", fontSize: ".78rem",
                        fontWeight: 600, color: "var(--text-1)"
                    }}>
                        {stadium.surface}
                    </span>
                </div>

                <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "7px 0"
                }}>
                    <span style={{
                        display: "flex", alignItems: "center", gap: 7,
                        fontSize: ".76rem", color: "var(--text-2)"
                    }}>
                        <span>🌡</span> Weather
                    </span>
                    {weather ? (
                        <span style={{
                            display: "flex", alignItems: "center", gap: 6,
                            fontFamily: "var(--font-mono)", fontSize: ".78rem",
                            fontWeight: 600, color: "var(--text-1)"
                        }}>
                            {weather.temp_f}°F · {weather.temp_c}°C
                            <span style={{ fontSize: "1rem" }}>{weather.icon}</span>
                            <span style={{
                                fontSize: ".7rem", color: "var(--text-3)",
                                fontFamily: "var(--font-body)", fontWeight: 400
                            }}>
                                {weather.desc}
                            </span>
                        </span>
                    ) : (
                        <span style={{
                            fontFamily: "var(--font-mono)", fontSize: ".72rem",
                            color: "var(--text-3)"
                        }}>
                            Fetching…
                        </span>
                    )}
                </div>

            </div>
        </div>
    )
}