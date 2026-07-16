"""
Async LLM generation client with local-first inference and cloud fallback.

Provides a two-tier text generation pipeline:
    1. Local Ollama inference using a self-hosted Mistral model.
    2. Groq API fallback using an OpenAI-compatible chat completion endpoint.

The client prioritizes low-latency local generation while maintaining service
availability through automatic provider fallback. Runtime API key resolution
ensures environment configuration changes are respected without import-order
dependencies.

Used by narrative generation agents to produce concise football intelligence
summaries and match commentary.
"""

import asyncio
import logging
import os
import re
from typing import Optional

import httpx

log = logging.getLogger(__name__)

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
OLLAMA_TIMEOUT = 15.0

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

GROQ_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_FALLBACK_MODELS", "llama-3.3-70b-versatile").split(",")
    if m.strip()
]

GROQ_BASE = "https://api.groq.com/openai/v1"

MAX_TOKENS = 200

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)

_ollama_lock = asyncio.Lock()


def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def _clean(text: str) -> str:
    """Strip reasoning-model <think> blocks"""
    text = _THINK_RE.sub("", text).strip()
    return text


async def generate(
    prompt: str,
    timeout: float = OLLAMA_TIMEOUT,
    max_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
) -> str:
    """
    Generate text using the available inference backend.

    Attempts local Ollama inference first for low-latency generation. If the
    local model is unavailable or fails, automatically falls back to Groq when
    credentials are configured.

    Args:
        prompt: Instruction prompt formatted for the language model.
        timeout: Maximum wait time for local inference requests.

    Returns:
        str: Generated text response. Returns an empty string when all providers
        are unavailable, allowing callers to apply template-based fallback logic.
    """
    text, _ = await generate_with_source(prompt, timeout, max_tokens, num_ctx)
    return text


async def generate_with_source(
    prompt: str,
    timeout: float = OLLAMA_TIMEOUT,
    max_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
) -> tuple[str, str]:

    budget = max_tokens or MAX_TOKENS
    ctx = num_ctx or 1024
    async with _ollama_lock:
        result = await _ollama(prompt, timeout, budget, ctx)
    if result:
        return result, "ollama"

    if _groq_key():
        result = await _groq(prompt, budget)
        if result:
            return result, "groq"
    else:
        log.warning("GROQ_API_KEY not set — skipping Groq fallback")

    log.warning(
        "Both Ollama and Groq unavailable — caller should use template fallback"
    )
    return "", ""


async def _ollama(
    prompt: str, timeout: float, max_tokens: int = MAX_TOKENS, num_ctx: int = 1024
) -> Optional[str]:
    """
    Generate text using a local Ollama model server.

    Sends an asynchronous generation request to the Ollama HTTP API with
    controlled sampling parameters optimized for short narrative responses.

    Args:
        prompt: Model instruction prompt.
        timeout: HTTP request timeout duration.

    Returns:
        Optional[str]: Generated response text, or None when inference fails.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.65,
            "top_p": 0.9,
            "stop": ["\n\n", "[/INST]", "[INST]"],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            resp.raise_for_status()
            text = _clean(resp.json().get("response", ""))
            if text:
                log.debug(f"Ollama generated {len(text.split())} words")
            return text or None
    except httpx.TimeoutException:
        log.warning(f"Ollama timed out ({timeout}s)")
        return None
    except httpx.ConnectError as exc:
        log.warning(f"Ollama connection refused — is `ollama serve` running? ({exc})")
        return None
    except Exception as exc:
        log.warning(f"Ollama error: {exc}")
        return None


async def _groq(prompt: str, max_tokens: int = MAX_TOKENS) -> Optional[str]:
    """
    Generate text using Groq's OpenAI-compatible inference API.

    Converts local instruction-style prompts into chat completion messages and
    requests a concise football analysis response from the configured Groq model.

    Args:
        prompt: Instruction prompt potentially containing [INST] formatting.

    Returns:
        Optional[str]: Generated completion text, or None on API failure.
    """
    clean = prompt.replace("[INST]", "").replace("[/INST]", "").strip()
    models = [GROQ_MODEL] + [m for m in GROQ_FALLBACK_MODELS if m != GROQ_MODEL]

    for i, model in enumerate(models):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a sharp football analyst providing live WC 2026 match "
                        "commentary. Be specific with numbers. Follow the length "
                        "instructions given in the user prompt exactly."
                    ),
                },
                {"role": "user", "content": clean},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.65,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{GROQ_BASE}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {_groq_key()}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                text = _clean(resp.json()["choices"][0]["message"]["content"])
                log.debug(f"Groq ({model}) generated {len(text.split())} words")
                return text or None
        except httpx.HTTPStatusError as exc:
            is_last = i == len(models) - 1
            if exc.response.status_code == 429 and not is_last:
                log.warning(f"Groq {model} rate-limited — trying next model")
                continue
            log.warning(
                f"Groq HTTP {exc.response.status_code} ({model}): {exc.response.text[:200]}"
            )
            return None
        except Exception as exc:
            log.warning(f"Groq error ({model}): {exc}")
            return None
    return None
