"""
LangSmith tracing without LangChain.

Uses the `langsmith` SDK's @traceable directly; the match-intel LangGraph
(agents/intel_graph.py) traces natively under the same env vars.

`traceable` is a no-op passthrough unless both are set:
    LANGSMITH_API_KEY
    LANGSMITH_TRACING=true   (or legacy LANGCHAIN_TRACING_V2=true)
"""

from __future__ import annotations

import os


def _tracing_enabled() -> bool:
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    for var in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"):
        if os.getenv(var, "").lower() == "true":
            return True
    return False


if _tracing_enabled():
    from langsmith import traceable
else:

    def traceable(*dargs, **dkwargs):
        """No-op stand-in; usable bare or as @traceable(name=..., run_type=...)."""
        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def _decorator(fn):
            return fn

        return _decorator
