// lib/api.ts
// Shared helper for the handful of /trigger debug endpoints that require a
// shared-secret X-Trigger-Token header once TRIGGER_TOKEN is configured on
// the backend (see backend/api/routes/_security.py). Unset by default —
// fails open on a local/dev box, exactly like the backend does.

const TRIGGER_TOKEN = process.env.NEXT_PUBLIC_TRIGGER_TOKEN ?? ""

export function triggerHeaders(): HeadersInit | undefined {
    return TRIGGER_TOKEN ? { "X-Trigger-Token": TRIGGER_TOKEN } : undefined
}
