import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class AdaptiveContextConfig:
    """Configuration for AdaptiveContext."""
    
    # Memory settings
    active_token_limit: int = 4096
    working_token_limit: int = 8192
    archive_token_limit: int = 16384
    
    # Knowledge store settings
    knowledge_store_path: str = "adaptive_context.db"
    retrieval_type: str = "hybrid"
    trust_marker: str = "📚" 
    use_reranking: bool = True
    rerank_top_k: int = 15
    
    # Vector embedding settings
    vector_model: str = "all-MiniLM-L6-v2"
    
    # Graph RAG settings
    use_graph_rag: bool = False
    enable_multi_hop_queries: bool = False
    max_graph_hops: int = 3
    graph_weight: float = 0.5
    
    # Chain of Agents settings
    use_chain_of_agents: bool = False
    chain_complexity_threshold: int = 5  # Minimum number of words to trigger CoA
    chain_agent_count: int = 3  # Number of agents in the chain
    
    # Self-Reflection settings
    use_self_reflection: bool = False
    reflection_relevance_threshold: float = 0.6  # Minimum relevance score for knowledge items
    reflection_confidence_threshold: float = 0.7  # Minimum confidence for consistency checks
    
    # Dynamic Memory Weighting settings
    use_dynamic_weighting: bool = False
    dynamic_weighting_learning_rate: float = 0.1  # Learning rate for weight adjustments
    dynamic_weighting_min_tier_size: int = 1000  # Minimum token size for any tier
    dynamic_weighting_default_ratios: Dict[str, float] = field(default_factory=lambda: {
        "active": 0.25,
        "working": 0.35,
        "archive": 0.40
    })
    
    # LLM Integration
    default_model: str = "gemma3:1b"
    ollama_host: str = "http://localhost:11434"
    
    # Conversation style
    conversation_style: str = "casual"
    system_persona: str = "helpful assistant"
    
    # Classifier settings
    use_ml_classifier: bool = False
    classifier_model: str = "all-MiniLM-L6-v2"
    classifier_threshold: float = 0.7
    
    # Advanced settings
    verbose_logging: bool = False
    debug_mode: bool = False
    
    # Optional custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize any derived settings after creation."""
        # Ensure path is absolute
        if not os.path.isabs(self.knowledge_store_path):
            # If relative, make it relative to the current working directory
            self.knowledge_store_path = os.path.join(os.getcwd(), self.knowledge_store_path)
            
        # Set default values for any properties that might not be set
        # This ensures backward compatibility when adding new properties
        if not hasattr(self, 'use_graph_rag'):
            self.use_graph_rag = False
            
        if not hasattr(self, 'enable_multi_hop_queries'):
            self.enable_multi_hop_queries = False
            
        if not hasattr(self, 'max_graph_hops'):
            self.max_graph_hops = 3
            
        if not hasattr(self, 'graph_weight'):
            self.graph_weight = 0.5
            
        if not hasattr(self, 'use_ml_classifier'):
            self.use_ml_classifier = False
            
        # Chain of Agents backward compatibility
        if not hasattr(self, 'use_chain_of_agents'):
            self.use_chain_of_agents = False
            
        if not hasattr(self, 'chain_complexity_threshold'):
            self.chain_complexity_threshold = 5
            
        if not hasattr(self, 'chain_agent_count'):
            self.chain_agent_count = 3
            
        # Self-Reflection backward compatibility
        if not hasattr(self, 'use_self_reflection'):
            self.use_self_reflection = False
            
        if not hasattr(self, 'reflection_relevance_threshold'):
            self.reflection_relevance_threshold = 0.6
            
        if not hasattr(self, 'reflection_confidence_threshold'):
            self.reflection_confidence_threshold = 0.7
            
        # Dynamic Weighting backward compatibility
        if not hasattr(self, 'use_dynamic_weighting'):
            self.use_dynamic_weighting = False
            
        if not hasattr(self, 'dynamic_weighting_learning_rate'):
            self.dynamic_weighting_learning_rate = 0.1
            
        if not hasattr(self, 'dynamic_weighting_min_tier_size'):
            self.dynamic_weighting_min_tier_size = 1000
            
        if not hasattr(self, 'dynamic_weighting_default_ratios'):
            self.dynamic_weighting_default_ratios = {
                "active": 0.25,
                "working": 0.35,
                "archive": 0.40
            }
    
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
            "active_token_limit": self.active_token_limit,
            "working_token_limit": self.working_token_limit,
            "archive_token_limit": self.archive_token_limit,
            "knowledge_store_path": self.knowledge_store_path,
            "retrieval_type": self.retrieval_type,
            "trust_marker": self.trust_marker,
            "use_reranking": self.use_reranking,
            "rerank_top_k": self.rerank_top_k,
            "vector_model": self.vector_model,
            "use_graph_rag": self.use_graph_rag,
            "enable_multi_hop_queries": self.enable_multi_hop_queries,
            "max_graph_hops": self.max_graph_hops,
            "graph_weight": self.graph_weight,
            "use_chain_of_agents": self.use_chain_of_agents,
            "chain_complexity_threshold": self.chain_complexity_threshold,
            "chain_agent_count": self.chain_agent_count,
            "use_self_reflection": self.use_self_reflection,
            "reflection_relevance_threshold": self.reflection_relevance_threshold,
            "reflection_confidence_threshold": self.reflection_confidence_threshold,
            "use_dynamic_weighting": self.use_dynamic_weighting,
            "dynamic_weighting_learning_rate": self.dynamic_weighting_learning_rate,
            "dynamic_weighting_min_tier_size": self.dynamic_weighting_min_tier_size,
            "dynamic_weighting_default_ratios": self.dynamic_weighting_default_ratios,
            "default_model": self.default_model,
            "ollama_host": self.ollama_host,
            "conversation_style": self.conversation_style,
            "system_persona": self.system_persona,
            "use_ml_classifier": self.use_ml_classifier,
            "classifier_model": self.classifier_model,
            "classifier_threshold": self.classifier_threshold,
            "verbose_logging": self.verbose_logging,
            "debug_mode": self.debug_mode,
            "custom_config": self.custom_config
        }
    
    def log_config(self):
        """Log the current configuration."""
        print("AdaptiveContext Configuration:")
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}") 