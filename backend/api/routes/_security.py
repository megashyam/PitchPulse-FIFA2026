"""
FastAPI dependency guarding the "/trigger" debug endpoints.

Triggers run Monte Carlo simulations or LLM calls, so they accept GET and
POST but require an `X-Trigger-Token` header matching TRIGGER_TOKEN.

When TRIGGER_TOKEN is unset:
    - Default: allow the request (local dev) and log one warning per process.
    - REQUIRE_TRIGGER_TOKEN=true: fail closed with 503 until a token is set.
"""

import logging
import os

from fastapi import Header, HTTPException

log = logging.getLogger(__name__)

TRIGGER_TOKEN = os.getenv("TRIGGER_TOKEN", "")
REQUIRE_TRIGGER_TOKEN = os.getenv("REQUIRE_TRIGGER_TOKEN", "").lower() in (
    "1",
    "true",
    "yes",
)

_warned_open = False


async def require_trigger_token(
    x_trigger_token: str | None = Header(default=None),
) -> None:
    if not TRIGGER_TOKEN:
        if REQUIRE_TRIGGER_TOKEN:
            # Fail closed: enforcement demanded but no secret configured.
            raise HTTPException(
                status_code=503,
                detail=(
                    "Trigger endpoints are disabled: REQUIRE_TRIGGER_TOKEN is set "
                    "but TRIGGER_TOKEN is not configured."
                ),
            )
        # Dev convenience: allow, but make the open state visible once.
        global _warned_open
        if not _warned_open:
            _warned_open = True
            log.warning(
                "TRIGGER_TOKEN is unset — /trigger debug endpoints are OPEN. "
                "Set TRIGGER_TOKEN (and/or REQUIRE_TRIGGER_TOKEN=true) before "
                "exposing this service beyond localhost."
            )
        return

    if x_trigger_token != TRIGGER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-Trigger-Token header.",
        )
