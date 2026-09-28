"""
Two-tier text generation client.

Backends:
    - Local Ollama first (mistral:7b-instruct-q4_K_M, keep_alive and a
      capped num_ctx for fast warm calls).
    - Groq fallback when Ollama is unavailable or times out
      (llama-3.1-8b-instant, then llama-3.3-70b-versatile on a 429).

GROQ_API_KEY is read at call time, not at import.
"""

import logging
import os
import re
from typing import Optional

import httpx

from agents.langsmith_tracing import traceable
from agents.llm_queue import Pacer, PriorityGate

log = logging.getLogger(__name__)

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
OLLAMA_TIMEOUT = 15.0

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# Tried in order if GROQ_MODEL returns 429; each Groq model has its own
# rate-limit bucket.
GROQ_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_FALLBACK_MODELS", "llama-3.3-70b-versatile").split(",")
    if m.strip()
]
# Note: a reasoning model here (e.g. qwen/qwen3-*) will burn MAX_TOKENS on a
# <think> block and often return empty content at this short a budget —
# needs `reasoning_format`/`reasoning_effort` handling in _groq() if reused.
GROQ_BASE = "https://api.groq.com/openai/v1"

MAX_TOKENS = 150  # ~2-3 sentences of output

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)

# Ollama runs one generation at a time on this host; Groq allows a few
# concurrent requests but ~30 per minute per model. Both are priority gates
# (agents/llm_queue.py), so live narration jumps the backfill queue. A
# caller's timeout clock starts only once it holds a slot.
OLLAMA_GATE = PriorityGate("ollama", int(os.getenv("OLLAMA_CONCURRENCY", "1")))
GROQ_GATE = PriorityGate("groq", int(os.getenv("GROQ_CONCURRENCY", "2")))
_groq_pacer = Pacer(float(os.getenv("GROQ_MIN_INTERVAL_S", "2.0")))


def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def _clean(text: str) -> str:
    """Strip <think> blocks, including an unclosed one in a truncated reply."""
    text = _THINK_RE.sub("", text).strip()
    return text


async def generate(
    prompt: str,
    timeout: float = OLLAMA_TIMEOUT,
    max_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
) -> str:
    text, _ = await generate_with_source(prompt, timeout, max_tokens, num_ctx)
    return text


@traceable(name="ollama_client.generate_with_source", run_type="llm")
async def generate_with_source(
    prompt: str,
    timeout: float = OLLAMA_TIMEOUT,
    max_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
) -> tuple[str, str]:
    """Like generate(), plus the backend that served it ("groq"/"ollama"/"").

    `max_tokens` overrides MAX_TOKENS; raise `num_ctx` with it, since the
    window must hold prompt and response together.
    """
    budget = max_tokens or MAX_TOKENS
    ctx = num_ctx or 1024
    async with OLLAMA_GATE.slot():
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
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Keeps the model resident in VRAM between calls — without this,
        # Ollama's default 5-min idle unload means any gap between narration
        # calls pays a multi-second reload penalty on top of generation time.
        "keep_alive": "30m",
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.65,
            "top_p": 0.9,
            # "\n\n" as a stop sequence keeps single-paragraph prompts (the
            # common case) from drifting into a second paragraph once they've
            # said what they need to — a mechanical backstop since small
            # models don't reliably self-limit on a soft length instruction.
            "stop": ["\n\n", "[/INST]", "[INST]"],
            # Prompt + response must both fit in this window — kept well
            # below the model's native 4096 for fast prompt eval, but sized
            # to the caller's actual max_tokens budget so a longer requested
            # response doesn't evict earlier context (see num_ctx note above).
            "num_ctx": num_ctx,
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


async def groq_chat(
    messages: list[dict],
    model: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.65,
    timeout: float = 10.0,
) -> str:
    """One gated, paced Groq chat completion. Raises httpx errors."""
    async with GROQ_GATE.slot():
        await _groq_pacer.wait()
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{GROQ_BASE}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={
                    "Authorization": f"Bearer {_groq_key()}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return _clean(resp.json()["choices"][0]["message"]["content"])


async def _groq(prompt: str, max_tokens: int = MAX_TOKENS) -> Optional[str]:
    """Groq chat completion from an [INST] prompt.

    Tries GROQ_MODEL, then GROQ_FALLBACK_MODELS in order on a 429.
    """
    clean = prompt.replace("[INST]", "").replace("[/INST]", "").strip()
    models = [GROQ_MODEL] + [m for m in GROQ_FALLBACK_MODELS if m != GROQ_MODEL]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a sharp football analyst providing live WC 2026 match "
                "commentary. Be specific with numbers. Follow the length "
                "instructions given in the user prompt exactly."
            ),
        },
        {"role": "user", "content": clean},
    ]

    for i, model in enumerate(models):
        try:
            text = await groq_chat(messages, model, max_tokens=max_tokens)
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
