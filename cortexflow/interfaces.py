"""
Core interfaces for CortexFlow.
All components should implement these interfaces for consistency and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator, Union


class ContextProvider(ABC):
    """Base interface for context providers."""
    
    @abstractmethod
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add a message to the context."""
        pass
    
    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
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
    def remember(self, text: str, source: Optional[str] = None) -> List[int]:
        """Store knowledge in the system."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve knowledge relevant to the query."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored knowledge."""
        pass


class LLMProviderInterface(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> Iterator[str]:
        """Generate text as a stream."""
        pass 