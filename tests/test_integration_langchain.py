"""Tests for the CortexFlow <-> LangChain integration."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, PropertyMock

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from cortexflow.integrations.langchain import (  # noqa: E402
    CortexFlowChatMessageHistory,
    CortexFlowMemory,
    CortexFlowRetriever,
    LANGCHAIN_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(messages=None):
    """Return a mock CortexFlowManager with sensible defaults."""
    manager = MagicMock()
    manager.memory.messages = messages or []
    manager.knowledge_store.retrieve.return_value = []
    return manager


# ---------------------------------------------------------------------------
# CortexFlowChatMessageHistory
# ---------------------------------------------------------------------------


class TestChatMessageHistory:
    """Tests for CortexFlowChatMessageHistory."""

    def test_langchain_available(self):
        assert LANGCHAIN_AVAILABLE is True

    def test_empty_messages(self):
        manager = _make_manager()
        history = CortexFlowChatMessageHistory(manager)
        assert history.messages == []

    def test_messages_converts_roles(self):
        raw = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": "You are helpful"},
        ]
        manager = _make_manager(messages=raw)
        history = CortexFlowChatMessageHistory(manager)

        msgs = history.messages
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Hello"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "Hi there"
        assert isinstance(msgs[2], SystemMessage)
        assert msgs[2].content == "You are helpful"

    def test_add_message_human(self):
        manager = _make_manager()
        history = CortexFlowChatMessageHistory(manager)
        history.add_message(HumanMessage(content="Test"))
        manager.add_message.assert_called_once_with("user", "Test")

    def test_add_message_ai(self):
        manager = _make_manager()
        history = CortexFlowChatMessageHistory(manager)
        history.add_message(AIMessage(content="Response"))
        manager.add_message.assert_called_once_with("assistant", "Response")

    def test_add_user_message(self):
        manager = _make_manager()
        history = CortexFlowChatMessageHistory(manager)
        history.add_user_message("What is CortexFlow?")
        manager.add_message.assert_called_once_with("user", "What is CortexFlow?")

    def test_add_ai_message(self):
        manager = _make_manager()
        history = CortexFlowChatMessageHistory(manager)
        history.add_ai_message("It is a memory framework.")
        manager.add_message.assert_called_once_with(
            "assistant", "It is a memory framework."
        )

    def test_clear(self):
        manager = _make_manager()
        history = CortexFlowChatMessageHistory(manager)
        history.clear()
        manager.clear_memory.assert_called_once()

    def test_session_context_forwarded(self):
        manager = _make_manager()
        CortexFlowChatMessageHistory(manager, session_id="s1", user_id="u1")
        manager.set_session_context.assert_called_once_with("s1", "u1")

    def test_no_session_context_when_none(self):
        manager = _make_manager()
        CortexFlowChatMessageHistory(manager)
        manager.set_session_context.assert_not_called()


# ---------------------------------------------------------------------------
# CortexFlowRetriever
# ---------------------------------------------------------------------------


class TestRetriever:
    """Tests for CortexFlowRetriever."""

    def test_retrieve_returns_documents(self):
        manager = _make_manager()
        manager.knowledge_store.retrieve.return_value = [
            {"text": "CortexFlow manages memory", "score": 0.95, "source": "docs"},
            {"text": "Multi-tier architecture", "score": 0.88, "source": "readme"},
        ]

        retriever = CortexFlowRetriever(manager, max_results=3)
        docs = retriever._get_relevant_documents("memory management")

        assert len(docs) == 2
        assert docs[0].page_content == "CortexFlow manages memory"
        assert docs[0].metadata["score"] == 0.95
        assert docs[0].metadata["source"] == "docs"
        assert docs[1].page_content == "Multi-tier architecture"

        manager.knowledge_store.retrieve.assert_called_once_with(
            "memory management", max_results=3
        )

    def test_retrieve_empty_results(self):
        manager = _make_manager()
        retriever = CortexFlowRetriever(manager)
        docs = retriever._get_relevant_documents("nonexistent topic")
        assert docs == []

    def test_retrieve_uses_content_key_fallback(self):
        """Results may use 'content' instead of 'text'."""
        manager = _make_manager()
        manager.knowledge_store.retrieve.return_value = [
            {"content": "Fallback content", "score": 0.7},
        ]
        retriever = CortexFlowRetriever(manager)
        docs = retriever._get_relevant_documents("query")
        assert docs[0].page_content == "Fallback content"

    def test_retrieve_no_knowledge_store(self):
        manager = MagicMock(spec=[])  # no knowledge_store attribute
        retriever = CortexFlowRetriever(manager)
        docs = retriever._get_relevant_documents("query")
        assert docs == []

    def test_max_results_default(self):
        manager = _make_manager()
        retriever = CortexFlowRetriever(manager)
        assert retriever.max_results == 5


# ---------------------------------------------------------------------------
# CortexFlowMemory
# ---------------------------------------------------------------------------


class TestMemory:
    """Tests for the CortexFlowMemory convenience wrapper."""

    def test_history_property(self):
        manager = _make_manager()
        mem = CortexFlowMemory(manager)
        assert isinstance(mem.history, CortexFlowChatMessageHistory)

    def test_retriever_property(self):
        manager = _make_manager()
        mem = CortexFlowMemory(manager)
        assert isinstance(mem.retriever, CortexFlowRetriever)

    def test_messages_delegation(self):
        raw = [{"role": "user", "content": "Hi"}]
        manager = _make_manager(messages=raw)
        mem = CortexFlowMemory(manager)
        assert len(mem.messages) == 1
        assert isinstance(mem.messages[0], HumanMessage)

    def test_add_user_message_delegation(self):
        manager = _make_manager()
        mem = CortexFlowMemory(manager)
        mem.add_user_message("Hello")
        manager.add_message.assert_called_once_with("user", "Hello")

    def test_add_ai_message_delegation(self):
        manager = _make_manager()
        mem = CortexFlowMemory(manager)
        mem.add_ai_message("Hi!")
        manager.add_message.assert_called_once_with("assistant", "Hi!")

    def test_search_delegation(self):
        manager = _make_manager()
        manager.knowledge_store.retrieve.return_value = [
            {"text": "result", "score": 0.9},
        ]
        mem = CortexFlowMemory(manager, max_results=2)
        docs = mem.search("query")
        assert len(docs) == 1
        assert docs[0].page_content == "result"

    def test_clear_delegation(self):
        manager = _make_manager()
        mem = CortexFlowMemory(manager)
        mem.clear()
        manager.clear_memory.assert_called_once()

    def test_session_forwarded_to_history(self):
        manager = _make_manager()
        mem = CortexFlowMemory(manager, session_id="sess", user_id="usr")
        assert mem.history.session_id == "sess"
        assert mem.history.user_id == "usr"
        manager.set_session_context.assert_called_once_with("sess", "usr")
