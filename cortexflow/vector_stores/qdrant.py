"""Qdrant vector store backend for CortexFlow.

Requires the ``qdrant-client`` package::

    pip install qdrant-client

If the package is not installed, importing this module will succeed but
instantiating :class:`QdrantBackend` will raise :exc:`ImportError`.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from cortexflow.vector_stores.base import VectorSearchResult, VectorStoreBackend

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None  # type: ignore[assignment,misc]
    QDRANT_AVAILABLE = False


class QdrantBackend(VectorStoreBackend):
    """Vector store backed by `Qdrant <https://qdrant.tech/>`_.

    The backend connects to a Qdrant instance and manages a single
    collection.  The client mode is driven by the config:

    * If ``qdrant_url`` is set, connect to a remote Qdrant server.
    * If ``qdrant_path`` is set, use local on-disk persistence.
    * Otherwise, use an in-memory (ephemeral) Qdrant instance.

    Args:
        config: A ``CortexFlowConfig`` (or compatible) object.
    """

    def __init__(self, config: Any = None) -> None:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Install it with: pip install qdrant-client"
            )

        # Read config
        vs_cfg = getattr(config, "vector_store", None)
        self._collection_name = (
            getattr(vs_cfg, "collection_name", "cortexflow") if vs_cfg else "cortexflow"
        )
        dimension = getattr(vs_cfg, "embedding_dimension", 384) if vs_cfg else 384
        url = getattr(vs_cfg, "qdrant_url", None) if vs_cfg else None
        api_key = getattr(vs_cfg, "qdrant_api_key", None) if vs_cfg else None
        path = getattr(vs_cfg, "qdrant_path", None) if vs_cfg else None

        # Create client
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(location=":memory:")

        # Ensure the collection exists
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=dimension, distance=Distance.COSINE
                ),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_embedding(
        self,
        id: str | int,
        text: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a point into the Qdrant collection."""
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        payload: dict[str, Any] = {"text": text}
        payload["type"] = (metadata or {}).get("type", "knowledge")
        if metadata:
            payload["metadata"] = metadata

        # Qdrant accepts string or int ids; convert to a stable format
        point_id = self._to_point_id(id)

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            ],
        )

    def search(
        self, query_embedding: np.ndarray, max_results: int = 10
    ) -> list[VectorSearchResult]:
        """Search the Qdrant collection for similar vectors."""
        query_embedding = np.asarray(query_embedding, dtype=np.float32).ravel()

        hits = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding.tolist(),
            limit=max_results,
            with_payload=True,
        )

        results: list[VectorSearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                VectorSearchResult(
                    id=str(hit.id),
                    text=payload.get("text", ""),
                    score=float(hit.score),
                    type=payload.get("type", "knowledge"),
                    metadata=payload.get("metadata"),
                )
            )
        return results

    def delete(self, id: str | int) -> bool:
        """Delete a point from the Qdrant collection by ID."""
        point_id = self._to_point_id(id)
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=[point_id],
            )
            return True
        except Exception:
            logger.debug("Qdrant delete failed for id=%s", id, exc_info=True)
            return False

    def count(self) -> int:
        """Return the number of points in the collection."""
        info = self._client.get_collection(collection_name=self._collection_name)
        return info.points_count or 0

    def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            self._client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_point_id(id: str | int) -> str | int:
        """Convert an ID to a Qdrant-compatible point ID.

        Qdrant accepts either unsigned 64-bit integers or UUID strings.
        If the id is already an int, use it directly.  If it is a string
        that looks like an int, convert it.  Otherwise, generate a
        deterministic UUID from the string.
        """
        if isinstance(id, int):
            return id
        try:
            return int(id)
        except (ValueError, TypeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, str(id)))
