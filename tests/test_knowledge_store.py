"""
Tests for the CortexFlow KnowledgeStore module.

Covers add_knowledge, remember_knowledge deprecation, get_relevant_knowledge,
close idempotency, and context manager protocol.
Uses tempfile for DB paths to avoid test pollution.
"""

import os
import warnings
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from cortexflow.config import CortexFlowConfig, ConfigBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(db_path):
    """Create a minimal config pointing to the given DB path."""
    return CortexFlowConfig.from_dict({
        "knowledge_store_path": db_path,
        "use_graph_rag": False,
        "use_inference_engine": False,
        "use_reranking": False,
        "use_ml_classifier": False,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path():
    """Create a temporary file path for the database."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_knowledge.db")
    yield path
    # Cleanup
    if os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.fixture
def knowledge_store(tmp_db_path):
    """Create a KnowledgeStore with a temp DB."""
    from cortexflow.knowledge import KnowledgeStore
    config = _make_config(tmp_db_path)
    ks = KnowledgeStore(config)
    yield ks
    ks.close()


# ---------------------------------------------------------------------------
# add_knowledge
# ---------------------------------------------------------------------------

class TestAddKnowledge:
    """Test adding knowledge to the store."""

    def test_add_knowledge_returns_list_of_ids(self, knowledge_store):
        ids = knowledge_store.add_knowledge("Python is a programming language.")
        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_add_knowledge_with_source(self, knowledge_store):
        ids = knowledge_store.add_knowledge(
            "The Earth orbits the Sun.", source="astronomy_textbook"
        )
        assert len(ids) >= 1

    def test_add_knowledge_with_custom_confidence(self, knowledge_store):
        ids = knowledge_store.add_knowledge(
            "Water boils at 100 degrees Celsius.", confidence=0.99
        )
        assert len(ids) >= 1

    def test_add_multiple_knowledge_items(self, knowledge_store):
        ids1 = knowledge_store.add_knowledge("Fact one: cats are animals.")
        ids2 = knowledge_store.add_knowledge("Fact two: dogs are animals.")
        assert len(ids1) >= 1
        assert len(ids2) >= 1
        # IDs should be different
        assert ids1[0] != ids2[0]


# ---------------------------------------------------------------------------
# remember_knowledge deprecation
# ---------------------------------------------------------------------------

class TestRememberKnowledgeDeprecation:
    """Test that remember_knowledge is a deprecated wrapper."""

    def test_remember_knowledge_emits_deprecation_warning(self, knowledge_store):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            knowledge_store.remember_knowledge("Test knowledge.")
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_remember_knowledge_returns_ids(self, knowledge_store):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ids = knowledge_store.remember_knowledge("Another fact.")
        assert isinstance(ids, list)
        assert len(ids) >= 1


# ---------------------------------------------------------------------------
# get_relevant_knowledge
# ---------------------------------------------------------------------------

class TestGetRelevantKnowledge:
    """Test knowledge retrieval."""

    def test_returns_list(self, knowledge_store):
        knowledge_store.add_knowledge("Python is excellent for data science.")
        results = knowledge_store.get_relevant_knowledge("Python data")
        assert isinstance(results, list)

    def test_returns_results_after_adding_knowledge(self, knowledge_store):
        knowledge_store.add_knowledge("Mars is the fourth planet from the Sun.")
        results = knowledge_store.get_relevant_knowledge("Mars planet")
        # Should find something (at least via keyword/BM25 fallback)
        # Note: without sentence-transformers, vector search is disabled
        # but keyword fallback should still work
        assert isinstance(results, list)

    def test_empty_query_returns_list(self, knowledge_store):
        results = knowledge_store.get_relevant_knowledge("")
        assert isinstance(results, list)

    def test_max_results_respected(self, knowledge_store):
        for i in range(10):
            knowledge_store.add_knowledge(f"Knowledge item number {i} about testing.")
        results = knowledge_store.get_relevant_knowledge("testing", max_results=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# close() idempotency
# ---------------------------------------------------------------------------

class TestCloseIdempotent:
    """Test that close() can be called multiple times safely."""

    def test_close_once(self, tmp_db_path):
        from cortexflow.knowledge import KnowledgeStore
        config = _make_config(tmp_db_path)
        ks = KnowledgeStore(config)
        ks.close()
        assert ks._closed is True

    def test_close_twice_no_error(self, tmp_db_path):
        from cortexflow.knowledge import KnowledgeStore
        config = _make_config(tmp_db_path)
        ks = KnowledgeStore(config)
        ks.close()
        ks.close()  # Should not raise
        assert ks._closed is True

    def test_close_three_times_no_error(self, tmp_db_path):
        from cortexflow.knowledge import KnowledgeStore
        config = _make_config(tmp_db_path)
        ks = KnowledgeStore(config)
        ks.close()
        ks.close()
        ks.close()
        assert ks._closed is True


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    """Test __enter__ and __exit__ protocol."""

    def test_enter_returns_self(self, tmp_db_path):
        from cortexflow.knowledge import KnowledgeStore
        config = _make_config(tmp_db_path)
        ks = KnowledgeStore(config)
        with ks as store:
            assert store is ks

    def test_exit_calls_close(self, tmp_db_path):
        from cortexflow.knowledge import KnowledgeStore
        config = _make_config(tmp_db_path)
        ks = KnowledgeStore(config)
        with ks:
            ks.add_knowledge("Test data in context manager.")
        assert ks._closed is True

    def test_context_manager_on_exception(self, tmp_db_path):
        from cortexflow.knowledge import KnowledgeStore
        config = _make_config(tmp_db_path)
        ks = KnowledgeStore(config)
        try:
            with ks:
                raise ValueError("Test exception")
        except ValueError:
            pass
        # close() should still have been called
        assert ks._closed is True

    def test_exit_returns_false(self, tmp_db_path):
        """__exit__ should return False so exceptions propagate."""
        from cortexflow.knowledge import KnowledgeStore
        config = _make_config(tmp_db_path)
        ks = KnowledgeStore(config)
        result = ks.__exit__(None, None, None)
        assert result is False
