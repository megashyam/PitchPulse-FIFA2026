

const TRIGGER_TOKEN = process.env.NEXT_PUBLIC_TRIGGER_TOKEN ?? ""

export function triggerHeaders(): HeadersInit | undefined {
    return TRIGGER_TOKEN ? { "X-Trigger-Token": TRIGGER_TOKEN } : undefined
}
