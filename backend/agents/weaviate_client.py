"""
Weaviate client for the NarrativeArcs and tactical collections.

WEAVIATE_HOST / WEAVIATE_PORT / WEAVIATE_GRPC_PORT control the connection.
"""
import logging
import time
from typing import List, Optional

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

from config import (
    HYBRID_ALPHA,
    READY_TTL_S,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_PORT,
)

log = logging.getLogger(__name__)

# Collections use one self-provided named vector. Vectors must be written
# under this name: a plain `vector` list or the legacy REST "vector"
# field is stored but never reaches the HNSW index, which silently turned
# every hybrid query into 0.25 x BM25.
VECTOR_NAME = "default"



def connect_local() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )



# collections
#   NarrativeArcs     historical WC storylines (goals, red cards, momentum); used by
# event_reaction / xg_divergence narration and briefings
#   TacticalProfiles  per-team-per-match pressing fingerprints (PPDA per zone); used by
# tactical narration and /tactical
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
        ("ppda_full", wvc.config.DataType.NUMBER),
        ("pass_accuracy", wvc.config.DataType.NUMBER),
        ("shots", wvc.config.DataType.INT),
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
        "possession", "press_intensity", "ppda_full", "pass_accuracy", "shots",
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

    def recreate(self, name: str) -> None:
        """Drop and recreate one collection (schema changes need a rebuild)."""
        if self._client.collections.exists(name):
            self._client.collections.delete(name)
            log.info(f"Dropped Weaviate collection '{name}'")
        self._ensure_one(name, _SCHEMAS[name])
    def get_count(self, collection: str = DEFAULT_COLLECTION) -> int:
        if collection not in _SCHEMAS:
            log.warning(f"get_count: unknown collection '{collection}'")
            return 0
        try:
            col = self._client.collections.get(collection)
            return col.aggregate.over_all(total_count=True).total_count or 0
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
        alpha: float = HYBRID_ALPHA,
    ):
        """Hybrid BM25 + vector search over `collection` (default NarrativeArcs).

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
                out = [
                    {
                        **obj.properties,
                        "_score": (
                            float(obj.metadata.score)
                            if obj.metadata and obj.metadata.score is not None
                            else None
                        ),
                    }
                    for obj in results.objects
                ]
                log.debug(f"Weaviate[{collection}]: {len(out)} objs for '{query_text[:50]}'")
                return out

            docs = [obj.properties["content"] for obj in results.objects]
            log.debug(f"Weaviate[{collection}]: {len(docs)} docs for '{query_text[:50]}'")
            return docs

        except Exception as exc:
            log.warning(f"Weaviate search error [{collection}]: {exc}")
            return []

    def fetch_all(self, collection: str) -> List[dict]:
        """Every object's properties (no vectors)."""
        if not self.ready or collection not in _SCHEMAS:
            return []
        try:
            col = self._client.collections.get(collection)
            return [
                dict(o.properties)
                for o in col.iterator(return_properties=_RETURN_PROPS[collection])
            ]
        except Exception as exc:
            log.warning(f"Weaviate fetch_all error [{collection}]: {exc}")
            return []

    def insert_many(self, collection: str, props_list: List[dict], vectors) -> int:
        """Batch insert under VECTOR_NAME; returns the number inserted."""
        col = self._client.collections.get(collection)
        with col.batch.dynamic() as batch:
            for props, vec in zip(props_list, vectors):
                batch.add_object(properties=props, vector={VECTOR_NAME: [float(x) for x in vec]})
        failed = len(col.batch.failed_objects)
        if failed:
            log.error(f"{failed} objects failed to insert into {collection}")
        return len(props_list) - failed


_client: Optional[WeaviateClient] = None


def get_weaviate_client() -> WeaviateClient:
    global _client
    if _client is None:
        _client = WeaviateClient()
        _client.connect()
    return _client
