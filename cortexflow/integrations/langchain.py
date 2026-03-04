"""
CortexFlow <-> LangChain integration.

Provides:
- ``CortexFlowChatMessageHistory`` -- LangChain-compatible chat message history
  backed by :class:`~cortexflow.manager.CortexFlowManager`.
- ``CortexFlowRetriever`` -- LangChain ``BaseRetriever`` that queries the
  CortexFlow knowledge store.
- ``CortexFlowMemory`` -- convenience wrapper that bundles history + retriever.

All three classes gracefully degrade when ``langchain-core`` is not installed:
stub base classes are substituted so the module can still be imported for type
checking or documentation generation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("cortexflow.integrations.langchain")

# ---------------------------------------------------------------------------
# Optional LangChain imports
# ---------------------------------------------------------------------------

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.retrievers import BaseRetriever

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Provide thin stubs so the module can be imported without langchain.
    class BaseMessage:  # type: ignore[no-redef]
        """Stub for ``langchain_core.messages.BaseMessage``."""

        def __init__(self, content: str = "", **kwargs: Any) -> None:
            self.content = content
            self.type: str = kwargs.get("type", "base")

    class HumanMessage(BaseMessage):  # type: ignore[no-redef]
        """Stub for ``langchain_core.messages.HumanMessage``."""

        def __init__(self, content: str = "", **kwargs: Any) -> None:
            super().__init__(content=content, **kwargs)
            self.type = "human"

    class AIMessage(BaseMessage):  # type: ignore[no-redef]
        """Stub for ``langchain_core.messages.AIMessage``."""

        def __init__(self, content: str = "", **kwargs: Any) -> None:
            super().__init__(content=content, **kwargs)
            self.type = "ai"

    class SystemMessage(BaseMessage):  # type: ignore[no-redef]
        """Stub for ``langchain_core.messages.SystemMessage``."""

        def __init__(self, content: str = "", **kwargs: Any) -> None:
            super().__init__(content=content, **kwargs)
            self.type = "system"

    class BaseChatMessageHistory:  # type: ignore[no-redef]
        """Stub for ``langchain_core.chat_history.BaseChatMessageHistory``."""

    class BaseRetriever:  # type: ignore[no-redef]
        """Stub for ``langchain_core.retrievers.BaseRetriever``."""

    class Document:  # type: ignore[no-redef]
        """Stub for ``langchain_core.documents.Document``."""

        def __init__(
            self, page_content: str = "", metadata: dict[str, Any] | None = None
        ) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    class CallbackManagerForRetrieverRun:  # type: ignore[no-redef]
        """Stub for ``langchain_core.callbacks.CallbackManagerForRetrieverRun``."""


# ---------------------------------------------------------------------------
# Role <-> Message-type mapping helpers
# ---------------------------------------------------------------------------

_ROLE_TO_MESSAGE_CLASS: dict[str, type] = {
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}


def _role_to_message(role: str, content: str) -> BaseMessage:
    """Convert a CortexFlow role/content pair to a LangChain ``BaseMessage``."""
    cls = _ROLE_TO_MESSAGE_CLASS.get(role, HumanMessage)
    return cls(content=content)


def _message_to_role(message: BaseMessage) -> str:
    """Convert a LangChain ``BaseMessage`` to the CortexFlow role string."""
    msg_type = getattr(message, "type", None)
    if msg_type in ("human",):
        return "user"
    if msg_type in ("ai",):
        return "assistant"
    if msg_type in ("system",):
        return "system"
    # Fallback: inspect class name
    cls_name = type(message).__name__.lower()
    if "human" in cls_name:
        return "user"
    if "ai" in cls_name or "assistant" in cls_name:
        return "assistant"
    if "system" in cls_name:
        return "system"
    return "user"


# ---------------------------------------------------------------------------
# CortexFlowChatMessageHistory
# ---------------------------------------------------------------------------


class CortexFlowChatMessageHistory(BaseChatMessageHistory):
    """LangChain ``BaseChatMessageHistory`` backed by a CortexFlowManager.

    Parameters
    ----------
    manager:
        An initialised :class:`~cortexflow.manager.CortexFlowManager` instance.
    session_id:
        Optional session identifier.  If the manager has session support
        enabled, this will be forwarded to ``set_session_context``.
    user_id:
        Optional user identifier, forwarded alongside *session_id*.
    """

    def __init__(
        self,
        manager: Any,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.manager = manager
        self.session_id = session_id
        self.user_id = user_id

        # If the manager supports session scoping, apply it.
        if session_id is not None or user_id is not None:
            if hasattr(manager, "set_session_context"):
                manager.set_session_context(session_id, user_id)

    # -- BaseChatMessageHistory interface -----------------------------------

    @property
    def messages(self) -> list[BaseMessage]:  # noqa: D401
        """Return the conversation history as a list of ``BaseMessage``."""
        raw_messages: list[dict[str, Any]] = getattr(
            self.manager, "memory", None
        ) and self.manager.memory.messages or []
        return [_role_to_message(m["role"], m["content"]) for m in raw_messages]

    def add_message(self, message: BaseMessage) -> None:
        """Persist *message* through the underlying CortexFlowManager."""
        role = _message_to_role(message)
        content = message.content if hasattr(message, "content") else str(message)
        self.manager.add_message(role, content)

    def add_user_message(self, message: str) -> None:
        """Convenience: add a user message by raw string."""
        self.manager.add_message("user", message)

    def add_ai_message(self, message: str) -> None:
        """Convenience: add an AI/assistant message by raw string."""
        self.manager.add_message("assistant", message)

    def clear(self) -> None:
        """Clear the conversation memory."""
        self.manager.clear_memory()


# ---------------------------------------------------------------------------
# CortexFlowRetriever
# ---------------------------------------------------------------------------


class CortexFlowRetriever(BaseRetriever):
    """LangChain ``BaseRetriever`` that queries the CortexFlow knowledge store.

    Parameters
    ----------
    manager:
        An initialised :class:`~cortexflow.manager.CortexFlowManager` instance
        whose ``knowledge_store`` will be used for retrieval.
    max_results:
        Maximum number of documents to return per query.
    """

    # Pydantic v1/v2-safe field declarations.  When langchain is not
    # installed these are just plain instance attributes set in __init__.
    manager: Any = None  # type: ignore[assignment]
    max_results: int = 5

    def __init__(self, manager: Any, max_results: int = 5, **kwargs: Any) -> None:
        if LANGCHAIN_AVAILABLE:
            # Let BaseRetriever (pydantic) handle field assignment.
            super().__init__(manager=manager, max_results=max_results, **kwargs)
        else:
            self.manager = manager
            self.max_results = max_results

    # -- BaseRetriever interface --------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Search the CortexFlow knowledge store and return ``Document`` objects."""
        knowledge_store = getattr(self.manager, "knowledge_store", None)
        if knowledge_store is None:
            logger.warning(
                "CortexFlowRetriever: manager has no knowledge_store attribute."
            )
            return []

        raw_results: list[dict[str, Any]] = knowledge_store.retrieve(
            query, max_results=self.max_results
        )

        documents: list[Document] = []
        for item in raw_results:
            text = item.get("text", item.get("content", ""))
            metadata: dict[str, Any] = {
                k: v for k, v in item.items() if k not in ("text", "content")
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents


# ---------------------------------------------------------------------------
# CortexFlowMemory
# ---------------------------------------------------------------------------


class CortexFlowMemory:
    """Convenience wrapper that bundles a chat history and a retriever.

    This is *not* a LangChain ``BaseMemory`` subclass -- it simply holds
    references to a :class:`CortexFlowChatMessageHistory` and a
    :class:`CortexFlowRetriever` for easy access.

    Parameters
    ----------
    manager:
        An initialised :class:`~cortexflow.manager.CortexFlowManager`.
    session_id / user_id:
        Forwarded to the underlying ``CortexFlowChatMessageHistory``.
    max_results:
        Forwarded to the underlying ``CortexFlowRetriever``.
    """

    def __init__(
        self,
        manager: Any,
        session_id: str | None = None,
        user_id: str | None = None,
        max_results: int = 5,
    ) -> None:
        self.manager = manager
        self._history = CortexFlowChatMessageHistory(
            manager, session_id=session_id, user_id=user_id
        )
        self._retriever = CortexFlowRetriever(manager, max_results=max_results)

    # -- Public properties --------------------------------------------------

    @property
    def history(self) -> CortexFlowChatMessageHistory:
        """The underlying chat message history."""
        return self._history

    @property
    def retriever(self) -> CortexFlowRetriever:
        """The underlying knowledge retriever."""
        return self._retriever

    # -- Convenience delegation ---------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:  # noqa: D401
        """Shortcut for ``self.history.messages``."""
        return self._history.messages

    def add_user_message(self, message: str) -> None:
        """Shortcut for ``self.history.add_user_message``."""
        self._history.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        """Shortcut for ``self.history.add_ai_message``."""
        self._history.add_ai_message(message)

    def search(self, query: str) -> list[Document]:
        """Shortcut for ``self.retriever._get_relevant_documents``."""
        return self._retriever._get_relevant_documents(query)

    def clear(self) -> None:
        """Clear the conversation memory."""
        self._history.clear()
