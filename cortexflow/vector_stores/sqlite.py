"""SQLite-based vector store backend.

This is the **default** backend and replicates the cosine-similarity-over-
BLOBs logic that was previously embedded directly in
``DenseVectorSearchStrategy.search()`` inside ``cortexflow/knowledge.py``.
It can operate with either an in-memory or on-disk SQLite database.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

from cortexflow.vector_stores.base import VectorSearchResult, VectorStoreBackend

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id   TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'knowledge',
    embedding BLOB NOT NULL,
    metadata TEXT
)
"""


class SQLiteVectorBackend(VectorStoreBackend):
    """Vector store that persists embeddings as BLOBs in SQLite.

    Similarity search uses vectorised numpy cosine similarity -- the same
    algorithm used by the original ``DenseVectorSearchStrategy``.

    Args:
        config: A ``CortexFlowConfig`` (or compatible) object.  The backend
            reads ``config.vector_store.embedding_dimension`` to validate
            vectors and uses ``config.knowledge_store.knowledge_store_path``
            as the database path unless an explicit ``connection`` is given.
        connection: An optional *existing* ``sqlite3.Connection``.  When
            supplied, the backend will use it instead of opening a new one.
            The caller retains ownership -- ``close()`` will **not** close a
            connection that was injected.
    """

    def __init__(
        self,
        config: Any = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        # Resolve embedding dimension
        if config is not None and hasattr(config, "vector_store"):
            self._dimension: int = config.vector_store.embedding_dimension
        else:
            self._dimension = 384  # sensible default (all-MiniLM-L6-v2)

        # Resolve database path
        if connection is not None:
            self._conn = connection
            self._owns_connection = False
        else:
            db_path = ":memory:"
            if config is not None and hasattr(config, "knowledge_store"):
                db_path = getattr(
                    config.knowledge_store, "knowledge_store_path", db_path
                )
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._owns_connection = True

        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

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
        """Store an embedding as a BLOB in the ``vector_embeddings`` table."""
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if embedding.shape[0] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {embedding.shape[0]}"
            )

        blob = embedding.tobytes()
        item_type = (metadata or {}).get("type", "knowledge")
        meta_json = _serialise_metadata(metadata)

        self._conn.execute(
            "INSERT OR REPLACE INTO vector_embeddings (id, text, type, embedding, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(id), text, item_type, blob, meta_json),
        )
        self._conn.commit()

    def search(
        self, query_embedding: np.ndarray, max_results: int = 10
    ) -> list[VectorSearchResult]:
        """Cosine-similarity search over stored BLOB embeddings.

        The implementation mirrors the vectorised numpy approach from
        ``DenseVectorSearchStrategy`` in ``knowledge.py``.
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32).ravel()
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        # Collect all rows in one pass
        cursor = self._conn.execute(
            "SELECT id, text, type, embedding, metadata FROM vector_embeddings"
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        ids: list[str] = []
        texts: list[str] = []
        types: list[str] = []
        metas: list[str | None] = []
        embedding_list: list[np.ndarray] = []

        dim = self._dimension
        for row in rows:
            blob = row["embedding"]
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != dim:
                continue
            ids.append(row["id"])
            texts.append(row["text"])
            types.append(row["type"])
            metas.append(row["metadata"])
            embedding_list.append(vec)

        if not embedding_list:
            return []

        # Vectorised cosine similarity (same as DenseVectorSearchStrategy)
        all_embeddings = np.vstack(embedding_list)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_normalized = all_embeddings / norms

        similarities = all_normalized @ query_normalized  # (N,)

        # Top-k selection
        n_results = min(max_results, len(similarities))
        if n_results >= len(similarities):
            top_indices = np.argsort(similarities)[::-1][:n_results]
        else:
            partitioned = np.argpartition(similarities, -n_results)[-n_results:]
            top_indices = partitioned[np.argsort(similarities[partitioned])[::-1]]

        results: list[VectorSearchResult] = []
        for idx in top_indices:
            meta = _deserialise_metadata(metas[idx])
            results.append(
                VectorSearchResult(
                    id=ids[idx],
                    text=texts[idx],
                    score=float(similarities[idx]),
                    type=types[idx],
                    metadata=meta,
                )
            )

        return results

    def delete(self, id: str | int) -> bool:
        """Delete an embedding by its ID."""
        cursor = self._conn.execute(
            "DELETE FROM vector_embeddings WHERE id = ?", (str(id),)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return the number of stored embeddings."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM vector_embeddings")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection if we own it."""
        if self._owns_connection and self._conn:
            self._conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


def _serialise_metadata(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    import json

    return json.dumps(metadata, default=str)


def _deserialise_metadata(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    import json

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
