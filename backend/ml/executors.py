"""
Dedicated executors, so CPU-heavy work never contends with request handling
on the asyncio default thread pool.

    SIM_EXECUTOR     one thread; tournament-wide Monte Carlo (predict route)
    EMBED_EXECUTOR   one thread; SentenceTransformer encodes (shared model)
    CF_SIM_EXECUTOR  two threads; counterfactual before/after pairs
    IO_EXECUTOR      two threads; blocking third-party clients (pytrends)
"""

from concurrent.futures import ThreadPoolExecutor

SIM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mc-sim")
EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")

# Counterfactual before/after simulations, separate from SIM_EXECUTOR so a
# tournament-wide sim never blocks them. Two workers run an event's pair
# concurrently (agents/counterfactual_agent.py's asyncio.gather).
CF_SIM_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cf-sim")

# Blocking third-party clients with no async API (pytrends); I/O-bound.
IO_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="blocking-io")
