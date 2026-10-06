"""
Text generation client.

Local Ollama with keep_alive and a capped num_ctx for fast warm calls. An
empty result tells the caller to use its template fallback.
"""

import logging
import re
from typing import Optional

import httpx

from langsmith import traceable
from agents.llm_queue import PriorityGate

from config import (
    OLLAMA_BASE,
    OLLAMA_CONCURRENCY,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
)

log = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 15.0

MAX_TOKENS = 150  # ~2-3 sentences of output

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)

# Ollama runs one generation at a time
# priority gate (agents/llm_queue.py): live narration goes ahead of backfill
# a caller's timeout starts once it holds a slot
OLLAMA_GATE = PriorityGate("ollama", OLLAMA_CONCURRENCY)


def _clean(text: str) -> str:
    """Strip <think> blocks, including an unclosed one in a truncated reply."""
    text = _THINK_RE.sub("", text).strip()
    return text


@traceable(name="ollama_client.generate_with_source", run_type="llm")
async def generate_with_source(
    prompt: str,
    timeout: float = OLLAMA_TIMEOUT,
    max_tokens: Optional[int] = None,
    temperature: float = 0.65,
) -> tuple[str, str]:
    """Generated text plus the backend that served it ("ollama"/"").

    `max_tokens` overrides MAX_TOKENS; the window (OLLAMA_NUM_CTX) must hold
    prompt and response together.
    """
    budget = max_tokens or MAX_TOKENS
    async with OLLAMA_GATE.slot():
        result = await _ollama(prompt, timeout, budget, temperature)
    if result:
        return result, "ollama"
    log.warning("Ollama unavailable — caller should use template fallback")
    return "", ""


async def _ollama(
    prompt: str,
    timeout: float,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.65,
) -> Optional[str]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Ollama unloads idle models after 5 min; keep_alive holds it for 30
        "keep_alive": "30m",
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            # stops at the first blank line; small models do not follow length instructions
            "stop": ["\n\n", "[/INST]", "[INST]"],
            # context window for prompt + response; below the native 4096 for faster prompt eval
            # one value for every call, since a change reloads the model
            "num_ctx": OLLAMA_NUM_CTX,
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
