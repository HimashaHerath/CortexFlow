"""Pluggable vector store backends for CortexFlow."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortexflow.vector_stores.base import VectorStoreBackend

from cortexflow.vector_stores.base import VectorStoreBackend, VectorSearchResult


def create_vector_store(config) -> "VectorStoreBackend":
    """Factory to create a vector store backend from config.

    The backend type is determined by ``config.vector_store.backend``.  When
    the *config* object does not carry a ``vector_store`` attribute (e.g. a
    plain ``CortexFlowConfig`` built without ``with_vector_store()``), the
    default SQLite backend is used so existing users get zero behavior change.

    Args:
        config: A ``CortexFlowConfig`` (or compatible) object.

    Returns:
        An initialised :class:`VectorStoreBackend` instance.
    """
    backend_type = (
        getattr(config, "vector_store_backend", "sqlite")
        if not hasattr(config, "vector_store")
        else "sqlite"
    )

    if hasattr(config, "vector_store"):
        backend_type = config.vector_store.backend

    if backend_type == "chromadb":
        from cortexflow.vector_stores.chromadb import ChromaDBBackend

        return ChromaDBBackend(config)
    elif backend_type == "qdrant":
        from cortexflow.vector_stores.qdrant import QdrantBackend

        return QdrantBackend(config)
    else:
        from cortexflow.vector_stores.sqlite import SQLiteVectorBackend

        return SQLiteVectorBackend(config)


__all__ = ["VectorStoreBackend", "VectorSearchResult", "create_vector_store"]
