"""ChromaDB vector store backend for CortexFlow.

Requires the ``chromadb`` package::

    pip install chromadb

If the package is not installed, importing this module will succeed but
instantiating :class:`ChromaDBBackend` will raise :exc:`ImportError`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cortexflow.vector_stores.base import VectorSearchResult, VectorStoreBackend

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment,misc]
    CHROMADB_AVAILABLE = False


class ChromaDBBackend(VectorStoreBackend):
    """Vector store backed by `ChromaDB <https://www.trychroma.com/>`_.

    The backend creates or connects to a ChromaDB collection.  The client
    mode (in-memory, persistent, or remote) is driven by the config:

    * If ``chromadb_host`` is set, connect to a remote server.
    * If ``chromadb_path`` is set, use persistent local storage.
    * Otherwise, use an ephemeral in-memory client.

    Args:
        config: A ``CortexFlowConfig`` (or compatible) object.
    """

    def __init__(self, config: Any = None) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install it with: pip install chromadb"
            )

        # Read config
        vs_cfg = getattr(config, "vector_store", None)
        collection_name = (
            getattr(vs_cfg, "collection_name", "cortexflow") if vs_cfg else "cortexflow"
        )
        host = getattr(vs_cfg, "chromadb_host", None) if vs_cfg else None
        port = getattr(vs_cfg, "chromadb_port", None) if vs_cfg else None
        path = getattr(vs_cfg, "chromadb_path", None) if vs_cfg else None

        # Create client
        if host:
            self._client = chromadb.HttpClient(host=host, port=port or 8000)
        elif path:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
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
        """Add an embedding to the ChromaDB collection."""
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        doc_metadata = dict(metadata) if metadata else {}
        doc_metadata.setdefault("type", "knowledge")

        self._collection.upsert(
            ids=[str(id)],
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[doc_metadata],
        )

    def search(
        self, query_embedding: np.ndarray, max_results: int = 10
    ) -> list[VectorSearchResult]:
        """Query the ChromaDB collection for similar vectors."""
        query_embedding = np.asarray(query_embedding, dtype=np.float32).ravel()

        if self._collection.count() == 0:
            return []

        n_results = min(max_results, self._collection.count())

        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: list[VectorSearchResult] = []
        if not result["ids"] or not result["ids"][0]:
            return results

        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else [""] * len(ids)
        metadatas = result["metadatas"][0] if result["metadatas"] else [{}] * len(ids)
        distances = result["distances"][0] if result["distances"] else [0.0] * len(ids)

        for doc_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            # ChromaDB returns cosine *distance*; convert to similarity
            score = 1.0 - distance
            item_type = (meta or {}).get("type", "knowledge")
            results.append(
                VectorSearchResult(
                    id=doc_id,
                    text=doc or "",
                    score=float(score),
                    type=item_type,
                    metadata=meta,
                )
            )

        return results

    def delete(self, id: str | int) -> bool:
        """Delete an embedding from the ChromaDB collection."""
        try:
            existing = self._collection.get(ids=[str(id)])
            if not existing["ids"]:
                return False
            self._collection.delete(ids=[str(id)])
            return True
        except Exception:
            logger.debug("ChromaDB delete failed for id=%s", id, exc_info=True)
            return False

    def count(self) -> int:
        """Return the number of items in the collection."""
        return self._collection.count()

    def close(self) -> None:
        """ChromaDB clients do not require explicit cleanup."""
        pass
