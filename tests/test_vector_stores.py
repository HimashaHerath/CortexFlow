"""Tests for pluggable vector store backends.

Tests cover:
- VectorSearchResult dataclass
- SQLiteVectorBackend: add, search, delete, count
- ChromaDB backend (skip if not installed)
- Qdrant backend (skip if not installed)
- create_vector_store factory
"""
from __future__ import annotations

import numpy as np
import pytest

from cortexflow.vector_stores import VectorSearchResult, create_vector_store
from cortexflow.vector_stores.base import VectorStoreBackend
from cortexflow.vector_stores.sqlite import SQLiteVectorBackend

# Check optional backend availability
try:
    from cortexflow.vector_stores.chromadb import CHROMADB_AVAILABLE
except Exception:
    CHROMADB_AVAILABLE = False

try:
    from cortexflow.vector_stores.qdrant import QDRANT_AVAILABLE
except Exception:
    QDRANT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMENSION = 8  # small dimension for fast tests


def _rand_vec(dim: int = DIMENSION) -> np.ndarray:
    """Return a random unit vector."""
    v = np.random.default_rng(42).random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_config(backend: str = "sqlite", dimension: int = DIMENSION, **kwargs):
    """Build a minimal config-like object for vector store tests."""

    class _VectorStoreCfg:
        def __init__(self):
            self.backend = backend
            self.collection_name = "test_cortexflow"
            self.embedding_dimension = dimension
            self.chromadb_host = kwargs.get("chromadb_host")
            self.chromadb_port = kwargs.get("chromadb_port")
            self.chromadb_path = kwargs.get("chromadb_path")
            self.qdrant_url = kwargs.get("qdrant_url")
            self.qdrant_api_key = kwargs.get("qdrant_api_key")
            self.qdrant_path = kwargs.get("qdrant_path")

    class _KnowledgeStoreCfg:
        knowledge_store_path = ":memory:"

    class _Config:
        vector_store = _VectorStoreCfg()
        knowledge_store = _KnowledgeStoreCfg()

    return _Config()


# ---------------------------------------------------------------------------
# VectorSearchResult dataclass
# ---------------------------------------------------------------------------

class TestVectorSearchResult:
    """Tests for the VectorSearchResult dataclass."""

    def test_defaults(self):
        r = VectorSearchResult(id="a", text="hello", score=0.9)
        assert r.id == "a"
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.type == "knowledge"
        assert r.metadata is None

    def test_with_metadata(self):
        meta = {"source": "wiki"}
        r = VectorSearchResult(id=1, text="x", score=0.5, type="fact", metadata=meta)
        assert r.type == "fact"
        assert r.metadata == {"source": "wiki"}

    def test_equality(self):
        a = VectorSearchResult(id="a", text="t", score=0.8)
        b = VectorSearchResult(id="a", text="t", score=0.8)
        assert a == b


# ---------------------------------------------------------------------------
# SQLiteVectorBackend
# ---------------------------------------------------------------------------

class TestSQLiteVectorBackend:
    """Tests for the default SQLite-based vector store backend."""

    @pytest.fixture()
    def store(self) -> SQLiteVectorBackend:
        """Return an in-memory SQLite vector store."""
        cfg = _make_config("sqlite")
        return SQLiteVectorBackend(cfg)

    def test_is_vector_store_backend(self, store: SQLiteVectorBackend):
        assert isinstance(store, VectorStoreBackend)

    def test_add_and_count(self, store: SQLiteVectorBackend):
        assert store.count() == 0
        store.add_embedding("1", "hello world", _rand_vec())
        assert store.count() == 1

    def test_add_duplicate_replaces(self, store: SQLiteVectorBackend):
        store.add_embedding("1", "first", _rand_vec())
        store.add_embedding("1", "second", _rand_vec())
        assert store.count() == 1

    def test_dimension_mismatch_raises(self, store: SQLiteVectorBackend):
        wrong_dim = np.zeros(DIMENSION + 5, dtype=np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add_embedding("bad", "text", wrong_dim)

    def test_search_empty(self, store: SQLiteVectorBackend):
        results = store.search(_rand_vec())
        assert results == []

    def test_search_returns_sorted_results(self, store: SQLiteVectorBackend):
        rng = np.random.default_rng(99)
        # Add three embeddings: one close to query, two random
        query = rng.random(DIMENSION).astype(np.float32)
        query /= np.linalg.norm(query)

        close_vec = query + rng.normal(0, 0.01, DIMENSION).astype(np.float32)
        close_vec = (close_vec / np.linalg.norm(close_vec)).astype(np.float32)

        far_vec1 = rng.random(DIMENSION).astype(np.float32)
        far_vec1 /= np.linalg.norm(far_vec1)
        far_vec2 = rng.random(DIMENSION).astype(np.float32)
        far_vec2 /= np.linalg.norm(far_vec2)

        store.add_embedding("close", "close item", close_vec)
        store.add_embedding("far1", "far item 1", far_vec1)
        store.add_embedding("far2", "far item 2", far_vec2)

        results = store.search(query, max_results=3)
        assert len(results) == 3
        assert results[0].id == "close"
        # Scores must be in descending order
        assert results[0].score >= results[1].score >= results[2].score

    def test_search_max_results(self, store: SQLiteVectorBackend):
        rng = np.random.default_rng(7)
        for i in range(10):
            v = rng.random(DIMENSION).astype(np.float32)
            v /= np.linalg.norm(v)
            store.add_embedding(str(i), f"item {i}", v)

        results = store.search(_rand_vec(), max_results=3)
        assert len(results) == 3

    def test_search_with_metadata(self, store: SQLiteVectorBackend):
        vec = _rand_vec()
        store.add_embedding("m1", "meta item", vec, metadata={"type": "fact", "source": "test"})
        results = store.search(vec, max_results=1)
        assert len(results) == 1
        assert results[0].type == "fact"
        assert results[0].metadata is not None
        assert results[0].metadata.get("source") == "test"

    def test_delete_existing(self, store: SQLiteVectorBackend):
        store.add_embedding("d1", "del me", _rand_vec())
        assert store.count() == 1
        assert store.delete("d1") is True
        assert store.count() == 0

    def test_delete_nonexistent(self, store: SQLiteVectorBackend):
        assert store.delete("nope") is False

    def test_close(self, store: SQLiteVectorBackend):
        store.close()
        # Calling close again should not raise
        store.close()

    def test_zero_norm_query_returns_empty(self, store: SQLiteVectorBackend):
        store.add_embedding("z1", "item", _rand_vec())
        results = store.search(np.zeros(DIMENSION, dtype=np.float32))
        assert results == []

    def test_no_config_defaults(self):
        """Backend should work with no config at all (defaults)."""
        store = SQLiteVectorBackend()
        vec = np.random.default_rng(0).random(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store.add_embedding("x", "test", vec)
        assert store.count() == 1
        store.close()


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
class TestChromaDBBackend:
    """Tests for the ChromaDB vector store backend."""

    @pytest.fixture()
    def store(self):
        from cortexflow.vector_stores.chromadb import ChromaDBBackend

        cfg = _make_config("chromadb")
        return ChromaDBBackend(cfg)

    def test_is_vector_store_backend(self, store):
        assert isinstance(store, VectorStoreBackend)

    def test_add_and_count(self, store):
        store.add_embedding("1", "hello", _rand_vec())
        assert store.count() >= 1

    def test_search(self, store):
        vec = _rand_vec()
        store.add_embedding("s1", "search me", vec)
        results = store.search(vec, max_results=1)
        assert len(results) >= 1
        assert results[0].id == "s1"

    def test_delete(self, store):
        store.add_embedding("d1", "del", _rand_vec())
        assert store.delete("d1") is True

    def test_delete_nonexistent(self, store):
        assert store.delete("nope") is False


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantBackend:
    """Tests for the Qdrant vector store backend."""

    @pytest.fixture()
    def store(self):
        from cortexflow.vector_stores.qdrant import QdrantBackend

        cfg = _make_config("qdrant")
        return QdrantBackend(cfg)

    def test_is_vector_store_backend(self, store):
        assert isinstance(store, VectorStoreBackend)

    def test_add_and_count(self, store):
        store.add_embedding("1", "hello", _rand_vec())
        assert store.count() >= 1

    def test_search(self, store):
        vec = _rand_vec()
        store.add_embedding("s1", "search me", vec)
        results = store.search(vec, max_results=1)
        assert len(results) >= 1

    def test_delete(self, store):
        store.add_embedding(1, "del", _rand_vec())
        assert store.delete(1) is True


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestCreateVectorStore:
    """Tests for the create_vector_store factory."""

    def test_default_is_sqlite(self):
        cfg = _make_config("sqlite")
        store = create_vector_store(cfg)
        assert isinstance(store, SQLiteVectorBackend)
        store.close()

    def test_unknown_backend_falls_back_to_sqlite(self):
        cfg = _make_config("unknown_backend_xyz")
        store = create_vector_store(cfg)
        assert isinstance(store, SQLiteVectorBackend)
        store.close()

    def test_no_vector_store_attr_defaults_to_sqlite(self):
        """Config without a vector_store attribute should default to SQLite."""

        class _BareCfg:
            pass

        store = create_vector_store(_BareCfg())
        assert isinstance(store, SQLiteVectorBackend)
        store.close()

    @pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
    def test_chromadb_factory(self):
        from cortexflow.vector_stores.chromadb import ChromaDBBackend

        cfg = _make_config("chromadb")
        store = create_vector_store(cfg)
        assert isinstance(store, ChromaDBBackend)
        store.close()

    @pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
    def test_qdrant_factory(self):
        from cortexflow.vector_stores.qdrant import QdrantBackend

        cfg = _make_config("qdrant")
        store = create_vector_store(cfg)
        assert isinstance(store, QdrantBackend)
        store.close()
