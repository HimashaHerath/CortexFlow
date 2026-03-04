"""
Core interfaces for CortexFlow.
All components should implement these interfaces for consistency and extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

__all__ = [
    "ContextProvider",
    "MemoryTierInterface",
    "KnowledgeStoreInterface",
    "LLMProviderInterface",
    "SessionAwareInterface",
    "AsyncContextProvider",
    "AsyncKnowledgeStoreInterface",
]


class ContextProvider(ABC):
    """Base interface for context providers."""

    @abstractmethod
    def add_message(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Add a message to the context."""
        pass

    @abstractmethod
    def get_context(self) -> dict[str, Any]:
        """Get the current context for model consumption."""
        pass

    @abstractmethod
    def clear_context(self) -> None:
        """Clear all context data."""
        pass


class MemoryTierInterface(ABC):
    """Interface for memory tiers."""

    @abstractmethod
    def add_content(self, content: Any, importance: float) -> bool:
        """Add content to this tier."""
        pass

    @abstractmethod
    def get_content(self) -> Any:
        """Get all content from this tier."""
        pass

    @abstractmethod
    def update_size(self, new_size: int) -> bool:
        """Update the size/capacity of this tier."""
        pass


class KnowledgeStoreInterface(ABC):
    """Interface for knowledge stores."""

    @abstractmethod
    def add_knowledge(
        self, text: str, source: str | None = None, confidence: float = 0.95
    ) -> list[int]:
        """Store knowledge in the system.

        Args:
            text: Text to store as knowledge.
            source: Optional source identifier.
            confidence: Confidence score for the stored facts.

        Returns:
            List of IDs for the stored knowledge items.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Retrieve knowledge relevant to the query."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored knowledge."""
        pass


class LLMProviderInterface(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    def generate(
        self, messages: list[dict[str, str]], model: str = None, **kwargs
    ) -> str:
        """Generate a response from a list of role/content messages."""
        pass

    @abstractmethod
    def generate_from_prompt(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate a response from a raw string prompt."""
        pass

    @abstractmethod
    def generate_stream(
        self, messages: list[dict[str, str]], model: str = None, **kwargs
    ) -> Iterator[str]:
        """Stream a response from a list of role/content messages."""
        pass

    async def agenerate(
        self, messages: list[dict[str, str]], model: str = None, **kwargs
    ) -> str:
        """Async generate a response. Default delegates to sync."""
        return self.generate(messages, model=model, **kwargs)

    async def agenerate_from_prompt(
        self, prompt: str, model: str = None, **kwargs
    ) -> str:
        """Async generate from prompt. Default delegates to sync."""
        return self.generate_from_prompt(prompt, model=model, **kwargs)


class SessionAwareInterface(ABC):
    """Mixin for components that can scope operations to a session/user."""

    @abstractmethod
    def set_session_context(self, session_id: str | None, user_id: str | None) -> None:
        """Set the active session/user scope for subsequent operations."""

    @abstractmethod
    def get_session_context(self) -> dict[str, Any]:
        """Return the currently active session/user context."""


class AsyncContextProvider(ABC):
    """Async variant of ContextProvider."""

    @abstractmethod
    async def add_message(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    async def get_context(self) -> dict[str, Any]:
        pass

    @abstractmethod
    async def clear_context(self) -> None:
        pass


class AsyncKnowledgeStoreInterface(ABC):
    """Async variant of KnowledgeStoreInterface."""

    @abstractmethod
    async def add_knowledge(
        self, text: str, source: str | None = None, confidence: float = 0.95
    ) -> list[int]:
        pass

    @abstractmethod
    async def retrieve(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass
