"use client"
// SSE connection to GET /matches/{fixtureId}/stream
// "match_update" replaces local state with the new MatchState; "waiting" sets
// isWaiting=true for a loading state
// reconnects automatically (EventSource does this natively)

import { useEffect, useState } from "react"
import type { MatchState } from "@/types/match"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface UseMatchStreamResult {
  state: MatchState | null
  isWaiting: boolean
  error: string | null
}

export function useMatchStream(fixtureId: string): UseMatchStreamResult {
  const [state, setState] = useState<MatchState | null>(null)
  const [isWaiting, setWaiting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!fixtureId) return

    const url = `${API}/matches/${fixtureId}/stream`
    const es = new EventSource(url)

    es.addEventListener("match_update", (e: MessageEvent) => {
      try {
        const data: MatchState = JSON.parse(e.data)
        setState(data)
        setWaiting(false)
        setError(null)
      } catch {
        setError("Failed to parse match update")
      }
    })

    es.addEventListener("waiting", () => {
      setWaiting(true)
    })

    es.addEventListener("error", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data)
        setError(data.message)
      } catch {
        // SSE connection error: EventSource retries automatically
      }
    })

    // native EventSource error (connection drop): no error state, EventSource
    // reconnects silently
    es.onerror = () => { }

    return () => es.close()
  }, [fixtureId])

  return { state, isWaiting, error }
}
