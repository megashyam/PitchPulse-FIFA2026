"""
Weaviate client for the NarrativeArcs and tactical collections.

WEAVIATE_HOST / WEAVIATE_PORT / WEAVIATE_GRPC_PORT control both the gRPC
connection and the REST URLs; WEAVIATE_API_KEY enables API-key auth.
"""

import json
import logging
import os
import time
import urllib.request
from typing import List, Optional

import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

log = logging.getLogger(__name__)

HYBRID_ALPHA = 0.75  # 75% vector + 25% BM25
# Collections use one self-provided named vector. Vectors must be written
# under this name: a plain `vector` list (batch) or the legacy REST "vector"
# field is stored but never reaches the HNSW index, which silently turned
# every hybrid query into 0.25 x BM25.
VECTOR_NAME = "default"
READY_TTL_S = 5.0

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
_REST_BASE = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"
# Set when Weaviate runs with API-key auth (docker-compose.yml).
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")


def connect_local() -> weaviate.WeaviateClient:
    auth = Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
        auth_credentials=auth,
    )


def rest_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if WEAVIATE_API_KEY:
        h["Authorization"] = f"Bearer {WEAVIATE_API_KEY}"
    return h

# ── Collection registry ───────────────────────────────────────────────────
#   NarrativeArcs    — historical WC storylines (goals, red cards, momentum).
#                      Used by event_reaction / xg_divergence narration + briefings.
#   TacticalProfiles — per-team-per-match pressing fingerprints (PPDA per zone).
#                      Used by tactical narration + the /tactical route.
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
        "content", "team", "opponent", "match_id", "competition", "season",
        "ppda", "ppda_def_third", "ppda_mid_third", "ppda_att_third",
        "possession", "press_intensity",
    ],
}


class WeaviateClient:

    def __init__(self):
        self._client: Optional[weaviate.WeaviateClient] = None
        self._ready = False
        self._ready_at = float("-inf")

    def connect(self) -> None:
        try:
            self._client = connect_local()
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
        """is_ready(), cached for READY_TTL_S (it is a blocking round-trip)."""
        now = time.monotonic()
        if now - self._ready_at < READY_TTL_S:
            return self._ready
        try:
            self._ready = self._client is not None and self._client.is_ready()
        except Exception:
            self._ready = False
        self._ready_at = now
        return self._ready

    # ── Schema management ─────────────────────────────────────────────────

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
            vector_config=wvc.config.Configure.Vectors.self_provided(name=VECTOR_NAME),
        )
        log.info(f"Created Weaviate collection '{name}'")

    # ── Counts ────────────────────────────────────────────────────────────

    def get_count(self, collection: str = DEFAULT_COLLECTION) -> int:
        if collection not in _SCHEMAS:
            log.warning(f"get_count: unknown collection '{collection}'")
            return 0
        try:
            q = json.dumps({"query": "{Aggregate{%s{meta{count}}}}" % collection})
            req = urllib.request.Request(
                f"{_REST_BASE}/v1/graphql",
                data=q.encode(),
                headers=rest_headers(),
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

    # ── Search ────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
        event_filter: Optional[str] = None,
        collection: str = DEFAULT_COLLECTION,
        return_objects: bool = False,
        alpha: float = HYBRID_ALPHA,
    ):
        """Hybrid BM25 + vector search.

        Defaults to NarrativeArcs, returning content strings; return_objects
        returns the full property dicts.
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
                alpha=alpha,
                target_vector=VECTOR_NAME,
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
                log.debug(f"Weaviate[{collection}]: {len(out)} objs for '{query_text[:50]}'")
                return out

            docs = [obj.properties["content"] for obj in results.objects]
            log.debug(f"Weaviate[{collection}]: {len(docs)} docs for '{query_text[:50]}'")
            return docs

        except Exception as exc:
            log.warning(f"Weaviate search error [{collection}]: {exc}")
            return []

    # ── Insert (REST path — bypasses Python client version quirks) ─────────

    def insert_document(self, collection: str, properties: dict, vector: List[float]) -> bool:
        """Generic insert. Caller supplies the full properties dict."""
        payload = json.dumps(
            {"class": collection, "properties": properties, "vectors": {VECTOR_NAME: vector}}
        ).encode()
        request = urllib.request.Request(
            f"{_REST_BASE}/v1/objects",
            data=payload,
            headers=rest_headers(),
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
