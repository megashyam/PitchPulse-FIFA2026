"""
Shared RAG helpers: hybrid retrieval and the precedent prompt block.
"""

import asyncio
from typing import List, Optional

from agents.weaviate_client import get_weaviate_client
from ml.embedding_model import get_embed_model
from ml.executors import EMBED_EXECUTOR

PRECEDENT_HEADER = (
    "\n\nHistorical WC precedent — cite ONLY these exact matches if "
    "relevant; do not name any other match, player, minute, or "
    "scoreline not listed here:\n"
)


async def retrieve(
    query: str, collection: str, top_k: int, event_filter: Optional[str] = None
) -> List[str]:
    """Embed `query` on EMBED_EXECUTOR, then hybrid-search `collection`.
    [] when Weaviate isn't ready; other failures raise."""
    wv = get_weaviate_client()
    if not wv.ready:
        return []
    model = get_embed_model()
    qv = await asyncio.get_running_loop().run_in_executor(
        EMBED_EXECUTOR, lambda: model.encode(query, normalize_embeddings=True).tolist()
    )
    docs = await asyncio.to_thread(
        wv.hybrid_search,
        query_vector=qv,
        query_text=query,
        top_k=top_k,
        event_filter=event_filter,
        collection=collection,
    )
    return docs or []


def precedent_block(docs: List[str], n: int, width: int, header: str = PRECEDENT_HEADER) -> str:
    """First `n` docs, each cut to `width` chars, as bullets under `header`."""
    if not docs:
        return ""
    return header + "\n".join(f"• {d[:width]}" for d in docs[:n])
