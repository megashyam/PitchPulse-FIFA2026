"""
Shared Weaviate access layer for narrative and tactical RAG pipelines.

Provides a centralized client wrapper responsible for connection management,
collection initialization, document storage, and hybrid vector/BM25 retrieval
across the NarrativeArcs and TacticalProfiles collections.

The module abstracts Weaviate operations used by downstream agents, including
narrative retrieval, tactical fingerprint matching, collection health checks,
and document counting. Connection parameters are configured through environment
variables to support both local development and containerized deployments.

The client gracefully degrades when Weaviate is unavailable, allowing agents
to continue operating with fallback logic.
"""

import json
import logging
import os
import urllib.request
from typing import List, Optional

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

log = logging.getLogger(__name__)

HYBRID_ALPHA = 0.75

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
_REST_BASE = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"

NARRATIVE_ARCS = "NarrativeArcs"
TACTICAL_PROFILES = "TacticalProfiles"

DEFAULT_COLLECTION = NARRATIVE_ARCS

_SCHEMAS = {
    NARRATIVE_ARCS: [
        ("content", wvc.config.DataType.TEXT),
        ("match_id", wvc.config.DataType.TEXT),
        ("competition", wvc.config.DataType.TEXT),
        ("season", wvc.config.DataType.TEXT),
        ("minute", wvc.config.DataType.INT),
        ("event_type", wvc.config.DataType.TEXT),
    ],
    TACTICAL_PROFILES: [
        ("content", wvc.config.DataType.TEXT),
        ("team", wvc.config.DataType.TEXT),
        ("opponent", wvc.config.DataType.TEXT),
        ("match_id", wvc.config.DataType.TEXT),
        ("competition", wvc.config.DataType.TEXT),
        ("season", wvc.config.DataType.TEXT),
        ("ppda", wvc.config.DataType.NUMBER),
        ("ppda_def_third", wvc.config.DataType.NUMBER),
        ("ppda_mid_third", wvc.config.DataType.NUMBER),
        ("ppda_att_third", wvc.config.DataType.NUMBER),
        ("possession", wvc.config.DataType.NUMBER),
        ("press_intensity", wvc.config.DataType.NUMBER),
    ],
}

_RETURN_PROPS = {
    NARRATIVE_ARCS: ["content", "match_id", "competition", "season"],
    TACTICAL_PROFILES: [
        "content",
        "team",
        "opponent",
        "match_id",
        "competition",
        "season",
        "ppda",
        "ppda_def_third",
        "ppda_mid_third",
        "ppda_att_third",
        "possession",
        "press_intensity",
    ],
}


class WeaviateClient:
    """
    Stateful wrapper around the shared Weaviate retrieval backend.

    Manages the lifecycle of the Weaviate connection, ensures required
    collections exist, and exposes common operations used by RAG agents:
    document counting, hybrid retrieval, and document insertion.

    The wrapper provides a stable interface for narrative and tactical
    pipelines while allowing deployment-specific Weaviate details to remain
    isolated inside this module.
    """

    def __init__(self):
        self._client: Optional[weaviate.WeaviateClient] = None

    def connect(self) -> None:
        try:
            self._client = weaviate.connect_to_local(
                host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT
            )
            log.info(
                f"Weaviate connected ({WEAVIATE_HOST}:{WEAVIATE_PORT}) — "
                f"ready: {self._client.is_ready()}"
            )
            self.ensure_collections()
        except Exception as exc:
            log.error(
                f"Weaviate connection failed ({WEAVIATE_HOST}:{WEAVIATE_PORT}): {exc}. "
                "RAG lookups will be skipped — agents still work via templates."
            )
            self._client = None

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass

    @property
    def ready(self) -> bool:
        try:
            return self._client is not None and self._client.is_ready()
        except Exception:
            return False

    def ensure_collections(self) -> None:
        """Create any missing collections. Safe to call on every startup."""
        if not self._client:
            return
        for name, props in _SCHEMAS.items():
            self._ensure_one(name, props)

    def _ensure_one(self, name: str, props) -> None:
        if self._client.collections.exists(name):
            log.info(f"Weaviate collection '{name}' already exists")
            return
        self._client.collections.create(
            name=name,
            properties=[
                wvc.config.Property(name=p_name, data_type=p_type)
                for p_name, p_type in props
            ],
        )
        log.info(f"Created Weaviate collection '{name}'")

    def get_count(self, collection: str = DEFAULT_COLLECTION) -> int:
        if collection not in _SCHEMAS:
            log.warning(f"get_count: unknown collection '{collection}'")
            return 0
        try:
            q = json.dumps({"query": "{Aggregate{%s{meta{count}}}}" % collection})
            req = urllib.request.Request(
                f"{_REST_BASE}/v1/graphql",
                data=q.encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
                return data["data"]["Aggregate"][collection][0]["meta"]["count"]
        except Exception as exc:
            log.warning(f"count failed for {collection}: {exc}")
            return 0

    def counts(self) -> dict:
        """All collection counts — used by /health and debug routes."""
        return {name: self.get_count(name) for name in _SCHEMAS}

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
        event_filter: Optional[str] = None,
        collection: str = DEFAULT_COLLECTION,
        return_objects: bool = False,
    ):
        """
        Execute hybrid semantic and keyword retrieval against Weaviate.

        Combines vector similarity search with BM25 keyword matching to retrieve
        relevant documents from the selected collection.

        Supports both narrative retrieval (content-only responses) and tactical
        retrieval (full metadata objects including similarity scores).

        Args:
            query_vector: Embedded representation of the search query.
            query_text: Raw text query for keyword retrieval.
            top_k: Maximum number of results returned.
            event_filter: Optional event type filter for narrative documents.
            collection: Target Weaviate collection.
            return_objects: Whether to return metadata objects instead of text.

        Returns:
            list: Retrieved documents or metadata objects. Returns an empty list
            when retrieval is unavailable or fails.
        """
        if not self.ready:
            log.debug("Weaviate not ready — skipping RAG")
            return []

        if collection not in _SCHEMAS:
            log.warning(f"Unknown collection '{collection}' — skipping search")
            return []

        try:
            col = self._client.collections.get(collection)

            filters = None
            if event_filter and collection == NARRATIVE_ARCS:
                filters = wvc.query.Filter.by_property("event_type").equal(event_filter)

            results = col.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=HYBRID_ALPHA,
                limit=top_k,
                filters=filters,
                return_metadata=MetadataQuery(score=True),
                return_properties=_RETURN_PROPS[collection],
            )

            if return_objects:
                out = []
                for obj in results.objects:
                    props = dict(obj.properties)
                    try:
                        props["_score"] = (
                            float(obj.metadata.score)
                            if obj.metadata and obj.metadata.score is not None
                            else None
                        )
                    except Exception:
                        props["_score"] = None
                    out.append(props)
                log.debug(
                    f"Weaviate[{collection}]: {len(out)} objs for '{query_text[:50]}'"
                )
                return out

            docs = [obj.properties["content"] for obj in results.objects]
            log.debug(
                f"Weaviate[{collection}]: {len(docs)} docs for '{query_text[:50]}'"
            )
            return docs

        except Exception as exc:
            log.warning(f"Weaviate search error [{collection}]: {exc}")
            return []

    def insert_document(
        self, collection: str, properties: dict, vector: List[float]
    ) -> bool:
        """
        Insert an embedded document into a Weaviate collection.

        Stores both document metadata and its vector representation through the
        Weaviate REST API.

        Args:
            collection: Destination collection name.
            properties: Structured metadata associated with the document.
            vector: Embedding vector used for semantic retrieval.

        Returns:
            bool: True when insertion succeeds, otherwise False.
        """
        payload = json.dumps(
            {"class": collection, "properties": properties, "vector": vector}
        ).encode()
        request = urllib.request.Request(
            f"{_REST_BASE}/v1/objects",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as r:
                return r.status in (200, 201)
        except Exception as exc:
            log.warning(f"Insert failed [{collection}]: {exc}")
            return False


_client: Optional[WeaviateClient] = None


def get_weaviate_client() -> WeaviateClient:
    global _client
    if _client is None:
        _client = WeaviateClient()
        _client.connect()
    return _client
