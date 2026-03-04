"""Tests for the CortexFlow <-> CrewAI integration.

CrewAI is not required to be installed -- all tests use a mocked
CortexFlowManager so the CrewAI storage protocol is validated without
the actual ``crewai`` package.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cortexflow.integrations.crewai import CortexFlowCrewStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(retrieve_results=None):
    """Return a mock CortexFlowManager with sensible defaults."""
    manager = MagicMock()
    manager.knowledge_store.retrieve.return_value = retrieve_results or []
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrewStorage:
    """Tests for CortexFlowCrewStorage."""

    # -- save ---------------------------------------------------------------

    def test_save_basic(self):
        manager = _make_manager()
        storage = CortexFlowCrewStorage(manager)
        storage.save("task_1", "The capital of France is Paris.")

        manager.knowledge_store.add_knowledge.assert_called_once_with(
            "The capital of France is Paris.", source="task_1"
        )

    def test_save_with_metadata(self):
        manager = _make_manager()
        storage = CortexFlowCrewStorage(manager)
        storage.save(
            "task_2",
            "Python is great.",
            metadata={"agent": "researcher", "step": "3"},
        )

        call_args = manager.knowledge_store.add_knowledge.call_args
        assert call_args[0][0] == "Python is great."
        source = call_args[1]["source"]
        assert "task_2" in source
        assert "agent=researcher" in source
        assert "step=3" in source

    def test_save_no_knowledge_store(self):
        """Should not raise when knowledge_store is missing."""
        manager = MagicMock(spec=[])  # no knowledge_store
        storage = CortexFlowCrewStorage(manager)
        storage.save("k", "v")  # should not raise

    # -- search -------------------------------------------------------------

    def test_search_basic(self):
        results = [
            {"text": "Paris is the capital", "score": 0.95},
            {"text": "France is in Europe", "score": 0.80},
        ]
        manager = _make_manager(retrieve_results=results)
        storage = CortexFlowCrewStorage(manager)

        found = storage.search("capital of France", limit=5)

        assert len(found) == 2
        assert found[0]["text"] == "Paris is the capital"
        manager.knowledge_store.retrieve.assert_called_once_with(
            "capital of France", max_results=5
        )

    def test_search_with_score_threshold(self):
        results = [
            {"text": "High relevance", "score": 0.9},
            {"text": "Low relevance", "score": 0.2},
        ]
        manager = _make_manager(retrieve_results=results)
        storage = CortexFlowCrewStorage(manager)

        found = storage.search("query", score_threshold=0.5)
        assert len(found) == 1
        assert found[0]["text"] == "High relevance"

    def test_search_empty_results(self):
        manager = _make_manager()
        storage = CortexFlowCrewStorage(manager)
        assert storage.search("nothing here") == []

    def test_search_no_knowledge_store(self):
        manager = MagicMock(spec=[])
        storage = CortexFlowCrewStorage(manager)
        assert storage.search("query") == []

    def test_search_uses_relevance_key(self):
        """Some CortexFlow results use 'relevance' instead of 'score'."""
        results = [
            {"text": "Item", "relevance": 0.85},
        ]
        manager = _make_manager(retrieve_results=results)
        storage = CortexFlowCrewStorage(manager)
        found = storage.search("query", score_threshold=0.5)
        assert len(found) == 1

    # -- reset --------------------------------------------------------------

    def test_reset(self):
        manager = _make_manager()
        storage = CortexFlowCrewStorage(manager)
        storage.reset()
        manager.knowledge_store.clear.assert_called_once()

    def test_reset_no_knowledge_store(self):
        manager = MagicMock(spec=[])
        storage = CortexFlowCrewStorage(manager)
        storage.reset()  # should not raise
