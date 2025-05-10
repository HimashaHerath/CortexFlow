import os
import logging
from typing import Dict, Any

class AdaptiveContextConfig:
    """Configuration for the AdaptiveContext system."""
    
    def __init__(self, **kwargs):
        """
        Initialize with default configuration.
        
        Args:
            **kwargs: Override default configuration with provided values
        """
        # Memory tiers configuration (tokens)
        self.active_tier_tokens = kwargs.get("active_tier_tokens", 4096)
        self.working_tier_tokens = kwargs.get("working_tier_tokens", 8192)
        self.archive_tier_tokens = kwargs.get("archive_tier_tokens", 16384)
        
        # Importance thresholds (0-1)
        self.working_importance_threshold = kwargs.get("working_importance_threshold", 0.3)
        self.archive_importance_threshold = kwargs.get("archive_importance_threshold", 0.1)
        
        # Token counting method
        self.token_counting_method = kwargs.get("token_counting_method", "basic")  # "basic", "tiktoken", "ollama"
        
        # Memory compression
        self.enable_compression = kwargs.get("enable_compression", True)
        self.compression_threshold = kwargs.get("compression_threshold", 0.8)  # When tier is this full, compress
        self.compression_target = kwargs.get("compression_target", 0.6)  # Target fullness after compression
        
        # Classification parameters
        self.classifier_model = kwargs.get("classifier_model", "gpt-3.5-turbo")
        self.classifier_temperature = kwargs.get("classifier_temperature", 0.1)
        
        # Vector embeddings
        self.vector_embedding_model = kwargs.get("vector_embedding_model", "all-MiniLM-L6-v2")
        
        # Ollama settings
        self.ollama_host = kwargs.get("ollama_host", "http://localhost:11434")
        self.default_model = kwargs.get("default_model", "gemma:7b")
        
        # Persistence paths
        self.storage_path = kwargs.get("storage_path", os.path.expanduser("~/.adaptive_context"))
        self.knowledge_store_path = kwargs.get("knowledge_store_path", os.path.join(self.storage_path, "knowledge.db"))
        
        # Enable vector embeddings for retrieval
        self.use_vector_embeddings = kwargs.get("use_vector_embeddings", True)
        
        # Enable result re-ranking
        self.use_reranking = kwargs.get("use_reranking", True)
        
        # GraphRAG configuration
        self.use_graph_rag = kwargs.get("use_graph_rag", True)
        self.graph_weight = kwargs.get("graph_weight", 0.3)
        self.enable_multi_hop_queries = kwargs.get("enable_multi_hop_queries", True)
        self.max_graph_hops = kwargs.get("max_graph_hops", 3)
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdaptiveContextConfig':
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            AdaptiveContextConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "active_tier_tokens": self.active_tier_tokens,
            "working_tier_tokens": self.working_tier_tokens,
            "archive_tier_tokens": self.archive_tier_tokens,
            "working_importance_threshold": self.working_importance_threshold,
            "archive_importance_threshold": self.archive_importance_threshold,
            "token_counting_method": self.token_counting_method,
            "enable_compression": self.enable_compression,
            "compression_threshold": self.compression_threshold,
            "compression_target": self.compression_target,
            "classifier_model": self.classifier_model,
            "classifier_temperature": self.classifier_temperature,
            "vector_embedding_model": self.vector_embedding_model,
            "ollama_host": self.ollama_host,
            "default_model": self.default_model,
            "storage_path": self.storage_path,
            "knowledge_store_path": self.knowledge_store_path,
            "use_vector_embeddings": self.use_vector_embeddings,
            "use_reranking": self.use_reranking,
            "use_graph_rag": self.use_graph_rag,
            "graph_weight": self.graph_weight,
            "enable_multi_hop_queries": self.enable_multi_hop_queries,
            "max_graph_hops": self.max_graph_hops
        }
    
    def log_config(self):
        """Log the current configuration."""
        logging.info("AdaptiveContext Configuration:")
        for key, value in self.to_dict().items():
            logging.info(f"  {key}: {value}") 