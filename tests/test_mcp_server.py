"""Tests for cortexflow.mcp_server."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, PropertyMock

import pytest

from cortexflow.mcp_server import (
    add_knowledge,
    add_memory,
    create_session,
    get_conversation_context,
    get_emotional_state,
    get_relationship_state,
    get_user_profile,
    manage_persona,
    search_memory,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def mock_manager():
    """Return a MagicMock that quacks like CortexFlowManager."""
    mgr = MagicMock()

    # add_message returns a message dict
    mgr.add_message.return_value = {
        "role": "user",
        "content": "hello",
        "metadata": {},
    }

    # knowledge_store.retrieve returns search results
    mgr.knowledge_store.retrieve.return_value = [
        {"text": "result 1", "score": 0.9},
    ]

    # add_knowledge returns list of IDs
    mgr.add_knowledge.return_value = [1, 2]

    # get_conversation_context returns context dict
    mgr.get_conversation_context.return_value = {
        "messages": [{"role": "user", "content": "hi"}],
        "knowledge": [],
    }

    # Companion AI: disabled by default (None)
    type(mgr).user_profile_manager = PropertyMock(return_value=None)
    type(mgr).emotion_tracker = PropertyMock(return_value=None)
    type(mgr).relationship_tracker = PropertyMock(return_value=None)
    type(mgr).persona_manager = PropertyMock(return_value=None)
    mgr.get_current_session.return_value = None

    return mgr


@pytest.fixture()
def mock_ctx(mock_manager):
    """Return a mock Context whose lifespan_context holds the mock manager."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context.manager = mock_manager
    return ctx


# ------------------------------------------------------------------
# Tool tests
# ------------------------------------------------------------------

class TestAddMemory:
    async def test_adds_message(self, mock_ctx, mock_manager):
        result = await add_memory("user", "hello", mock_ctx)
        mock_manager.add_message.assert_called_once_with("user", "hello", None)
        data = json.loads(result)
        assert data["role"] == "user"
        assert data["content"] == "hello"

    async def test_with_metadata(self, mock_ctx, mock_manager):
        meta = {"source": "test"}
        await add_memory("assistant", "hi", mock_ctx, metadata=meta)
        mock_manager.add_message.assert_called_once_with("assistant", "hi", meta)


class TestSearchMemory:
    async def test_returns_results(self, mock_ctx, mock_manager):
        result = await search_memory("hello", mock_ctx)
        mock_manager.knowledge_store.retrieve.assert_called_once_with(
            "hello", max_results=5
        )
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["text"] == "result 1"

    async def test_custom_max_results(self, mock_ctx, mock_manager):
        await search_memory("test", mock_ctx, max_results=10)
        mock_manager.knowledge_store.retrieve.assert_called_once_with(
            "test", max_results=10
        )


class TestAddKnowledge:
    async def test_stores_knowledge(self, mock_ctx, mock_manager):
        result = await add_knowledge("The sky is blue", mock_ctx)
        mock_manager.add_knowledge.assert_called_once_with(
            "The sky is blue", source=None, confidence=None
        )
        data = json.loads(result)
        assert data["stored_ids"] == [1, 2]

    async def test_with_source_and_confidence(self, mock_ctx, mock_manager):
        await add_knowledge("fact", mock_ctx, source="wiki", confidence=0.95)
        mock_manager.add_knowledge.assert_called_once_with(
            "fact", source="wiki", confidence=0.95
        )


class TestGetConversationContext:
    async def test_returns_context(self, mock_ctx, mock_manager):
        result = await get_conversation_context(mock_ctx)
        mock_manager.get_conversation_context.assert_called_once_with(max_tokens=None)
        data = json.loads(result)
        assert "messages" in data

    async def test_with_max_tokens(self, mock_ctx, mock_manager):
        await get_conversation_context(mock_ctx, max_tokens=1024)
        mock_manager.get_conversation_context.assert_called_once_with(max_tokens=1024)


class TestGetUserProfile:
    async def test_not_enabled(self, mock_ctx):
        result = await get_user_profile("u1", mock_ctx)
        data = json.loads(result)
        assert "error" in data

    async def test_returns_profile(self, mock_ctx, mock_manager):
        profile_mock = MagicMock()
        profile_mock.to_dict.return_value = {"user_id": "u1", "name": "Alice"}
        pm = MagicMock()
        pm.get_profile.return_value = profile_mock
        type(mock_manager).user_profile_manager = PropertyMock(return_value=pm)

        result = await get_user_profile("u1", mock_ctx)
        data = json.loads(result)
        assert data["user_id"] == "u1"
        assert data["name"] == "Alice"


class TestGetEmotionalState:
    async def test_not_enabled(self, mock_ctx):
        result = await get_emotional_state(mock_ctx)
        data = json.loads(result)
        assert "error" in data

    async def test_returns_state(self, mock_ctx, mock_manager):
        state_mock = MagicMock()
        state_mock.to_dict.return_value = {
            "primary_emotion": "joy",
            "intensity": 0.8,
        }
        tracker = MagicMock()
        tracker.get_current_state.return_value = state_mock
        type(mock_manager).emotion_tracker = PropertyMock(return_value=tracker)

        result = await get_emotional_state(mock_ctx)
        data = json.loads(result)
        assert data["primary_emotion"] == "joy"


class TestGetRelationshipState:
    async def test_not_enabled(self, mock_ctx):
        result = await get_relationship_state("u1", mock_ctx)
        data = json.loads(result)
        assert "error" in data

    async def test_returns_state(self, mock_ctx, mock_manager):
        state_mock = MagicMock()
        state_mock.to_dict.return_value = {
            "user_id": "u1",
            "persona_id": "default",
            "stage": "INTRODUCTION",
        }
        tracker = MagicMock()
        tracker.get_state.return_value = state_mock
        type(mock_manager).relationship_tracker = PropertyMock(return_value=tracker)

        result = await get_relationship_state("u1", mock_ctx)
        data = json.loads(result)
        assert data["stage"] == "INTRODUCTION"


class TestManagePersona:
    async def test_not_enabled(self, mock_ctx):
        result = await manage_persona("list", mock_ctx)
        data = json.loads(result)
        assert "error" in data

    async def test_list(self, mock_ctx, mock_manager):
        persona_mock = MagicMock()
        persona_mock.to_dict.return_value = {"persona_id": "p1", "name": "Buddy"}
        pm = MagicMock()
        pm.list_personas.return_value = [persona_mock]
        type(mock_manager).persona_manager = PropertyMock(return_value=pm)

        result = await manage_persona("list", mock_ctx)
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["persona_id"] == "p1"

    async def test_get(self, mock_ctx, mock_manager):
        persona_mock = MagicMock()
        persona_mock.to_dict.return_value = {"persona_id": "p1", "name": "Buddy"}
        pm = MagicMock()
        pm.get_persona.return_value = persona_mock
        type(mock_manager).persona_manager = PropertyMock(return_value=pm)

        result = await manage_persona("get", mock_ctx, persona_id="p1")
        data = json.loads(result)
        assert data["persona_id"] == "p1"

    async def test_get_missing_id(self, mock_ctx, mock_manager):
        pm = MagicMock()
        type(mock_manager).persona_manager = PropertyMock(return_value=pm)

        result = await manage_persona("get", mock_ctx)
        data = json.loads(result)
        assert "error" in data

    async def test_set(self, mock_ctx, mock_manager):
        persona_mock = MagicMock()
        pm = MagicMock()
        pm.get_persona.return_value = persona_mock
        type(mock_manager).persona_manager = PropertyMock(return_value=pm)

        result = await manage_persona("set", mock_ctx, persona_id="p1")
        data = json.loads(result)
        assert data["active_persona"] == "p1"

    async def test_unknown_action(self, mock_ctx, mock_manager):
        pm = MagicMock()
        type(mock_manager).persona_manager = PropertyMock(return_value=pm)

        result = await manage_persona("delete", mock_ctx)
        data = json.loads(result)
        assert "error" in data
        assert "Unknown action" in data["error"]


class TestCreateSession:
    async def test_sessions_not_enabled(self, mock_ctx, mock_manager):
        mock_manager.create_session.side_effect = RuntimeError(
            "Sessions are not enabled."
        )
        result = await create_session(mock_ctx)
        data = json.loads(result)
        assert "error" in data

    async def test_creates_session(self, mock_ctx, mock_manager):
        session_mock = MagicMock()
        session_mock.to_dict.return_value = {
            "session_id": "s1",
            "user_id": "u1",
            "is_active": True,
        }
        mock_manager.create_session.return_value = session_mock

        result = await create_session(mock_ctx, user_id="u1")
        mock_manager.create_session.assert_called_once_with("u1", metadata=None)
        data = json.loads(result)
        assert data["session_id"] == "s1"

    async def test_default_user_id(self, mock_ctx, mock_manager):
        session_mock = MagicMock()
        session_mock.to_dict.return_value = {"session_id": "s2", "user_id": "default"}
        mock_manager.create_session.return_value = session_mock

        await create_session(mock_ctx)
        mock_manager.create_session.assert_called_once_with("default", metadata=None)
