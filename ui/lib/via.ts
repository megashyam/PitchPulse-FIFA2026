// Which backend actually produced a narrative (backend `via` field).
export function viaLabel(via?: string | null): string {
    if (!via) return "LLM"
    if (via === "ollama") return "Ollama (local)"
    if (via === "groq") return "Groq"
    if (via.startsWith("template")) return "template"
    return via
}
