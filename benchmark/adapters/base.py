"""
Base adapter interface for benchmarking different RAG systems.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BenchmarkSystemAdapter(ABC):
    """Base adapter interface for benchmarking RAG systems."""
    
    def __init__(self, model: str = "llama3.2", verbose: bool = False):
        """
        Initialize the adapter.
        
        Args:
            model: The LLM model to use
            verbose: Whether to output verbose logs
        """
        self.model = model
        self.verbose = verbose
    
    @abstractmethod
    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        pass
    
    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        pass
    
    @abstractmethod
    def query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Submit a query to the system.
        
        Args:
            query: The user query
            context: Optional context documents
            
        Returns:
            The system's response
        """
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        pass
    
    @abstractmethod
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        pass 