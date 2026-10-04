// helper for the /trigger debug endpoints; sends the shared-secret X-Trigger-Token
// header once TRIGGER_TOKEN is set on the backend (see
// backend/api/routes/_security.py); unset by default, so open on a local box like the
// backend

const TRIGGER_TOKEN = process.env.NEXT_PUBLIC_TRIGGER_TOKEN ?? ""

export function triggerHeaders(): HeadersInit | undefined {
    return TRIGGER_TOKEN ? { "X-Trigger-Token": TRIGGER_TOKEN } : undefined
}
