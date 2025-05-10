"""
Adapter for benchmarking AdaptiveContext.
"""
from typing import Dict, Any, List, Optional
import time

from adaptive_context import AdaptiveContextManager
from adaptive_context.config import AdaptiveContextConfig

from benchmark.adapters.base import BenchmarkSystemAdapter


class AdaptiveContextAdapter(BenchmarkSystemAdapter):
    """Adapter for the AdaptiveContext system."""
    
    def __init__(self, model: str = "llama3.2", verbose: bool = False):
        """
        Initialize the AdaptiveContext adapter.
        
        Args:
            model: The LLM model to use
            verbose: Whether to output verbose logs
        """
        super().__init__(model, verbose)
        
        # Initialize AdaptiveContext with benchmark config
        self.config = AdaptiveContextConfig(
            default_model=model,
            active_token_limit=4000,
            working_token_limit=6000,
            archive_token_limit=10000,
            use_dynamic_weighting=True,
            use_graph_rag=True,
            enable_multi_hop_queries=True,
            max_graph_hops=3
        )
        
        self.manager = AdaptiveContextManager(self.config)
        self.conversation_start_time = time.time()
    
    def initialize_conversation(self) -> None:
        """Initialize a new conversation."""
        # Clear memory using the correct method
        self.manager.clear_memory()
        self.conversation_start_time = time.time()
        if self.verbose:
            print("AdaptiveContext: Conversation initialized")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        self.manager.add_message(role, content)
        if self.verbose:
            print(f"AdaptiveContext: Added {role} message: {content[:30]}...")
    
    def query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Submit a query to the system.
        
        Args:
            query: The user query
            context: Optional context documents
            
        Returns:
            The system's response
        """
        # Add any context documents to knowledge store
        if context:
            for i, doc in enumerate(context):
                self.manager.knowledge_store.remember_explicit(
                    text=doc,
                    source=f"benchmark_context_{i}",
                    confidence=0.9
                )
                if self.verbose:
                    print(f"AdaptiveContext: Added context document {i}: {doc[:30]}...")
        
        # Add the user query as a message
        self.add_message("user", query)
        
        # Get response
        response = self.manager.generate_response()
        
        if self.verbose:
            print(f"AdaptiveContext: Query: {query}")
            print(f"AdaptiveContext: Response: {response[:50]}...")
        
        return response
    
    def get_token_count(self, tier_name: str) -> int:
        """
        Get token count for a specific memory tier.
        
        Args:
            tier_name: Name of the tier ("active", "working", "archive")
            
        Returns:
            Token count for the tier
        """
        stats = self.manager.memory.get_stats()
        if "tiers" in stats and tier_name in stats["tiers"]:
            return stats["tiers"][tier_name]["used"]
        return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        # Get memory stats from tiers
        active_tokens = self.get_token_count("active")
        working_tokens = self.get_token_count("working")
        archive_tokens = self.get_token_count("archive")
        total_tokens = active_tokens + working_tokens + archive_tokens
        
        # Get knowledge store stats - estimate based on a query
        knowledge_items = self.manager.knowledge_store.get_relevant_knowledge("test query", max_results=100)
        knowledge_count = len(knowledge_items)
        
        # Get dynamic weighting stats if available
        weighting_stats = {}
        if hasattr(self.manager, "weighting_engine") and self.manager.weighting_engine:
            weighting_stats = self.manager.weighting_engine.get_stats()
        
        stats = {
            "total_tokens": total_tokens,
            "active_tokens": active_tokens,
            "working_tokens": working_tokens,
            "archive_tokens": archive_tokens,
            "knowledge_count": knowledge_count,
            "conversation_duration": time.time() - self.conversation_start_time,
            "weighting": weighting_stats
        }
        
        if self.verbose:
            print(f"AdaptiveContext: Memory stats: {stats}")
            
        return stats
    
    def flush_memory(self) -> None:
        """Flush the system's memory."""
        self.manager.clear_memory()
        if self.verbose:
            print("AdaptiveContext: Memory flushed") 