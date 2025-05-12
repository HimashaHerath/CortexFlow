"""
CortexFlow Configuration module.

This module provides the configuration class for the CortexFlow system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class CortexFlowConfig:
    """Configuration for CortexFlow."""
    
    # Memory settings
    active_token_limit: int = 4096
    working_token_limit: int = 8192
    archive_token_limit: int = 16384
    
    # Knowledge store settings
    knowledge_store_path: str = "cortexflow.db"
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
    
    # Ontology settings
    use_ontology: bool = False
    enable_ontology_evolution: bool = True
    ontology_confidence_threshold: float = 0.7
    
    # Metadata framework settings
    track_provenance: bool = True
    track_confidence: bool = True
    track_temporal: bool = True
    
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
    
    # Uncertainty Handling settings
    use_uncertainty_handling: bool = False
    auto_detect_contradictions: bool = True
    default_contradiction_strategy: str = "weighted"  # auto, recency, confidence, reliability, weighted, or keep_both
    recency_weight: float = 0.6  # Weight for recency in contradiction resolution
    reliability_weight: float = 0.4  # Weight for source reliability in contradiction resolution
    confidence_threshold: float = 0.7  # Threshold for high confidence assertions
    uncertainty_representation: str = "confidence"  # confidence, distribution, or both
    reason_with_incomplete_info: bool = True  # Whether to attempt reasoning with incomplete information
    
    # Performance Optimization settings
    use_performance_optimization: bool = False
    use_graph_partitioning: bool = False  # Enable graph partitioning for efficient storage and retrieval
    graph_partition_method: str = "louvain"  # Graph partitioning method: louvain, spectral, modularity
    target_partition_count: int = 5  # Target number of partitions (if applicable to method)
    use_multihop_indexing: bool = False  # Enable indexing strategies for multi-hop queries
    max_indexed_hops: int = 2  # Maximum number of hops to index for quick retrieval
    use_query_planning: bool = True  # Enable query planning system for optimizing reasoning paths
    use_reasoning_cache: bool = True  # Enable caching for common reasoning patterns
    reasoning_cache_max_size: int = 1000  # Maximum number of reasoning patterns to cache
    query_cache_max_size: int = 500  # Maximum number of query plans to cache
    cache_ttl: int = 3600  # Cache time-to-live in seconds (0 for no expiration)
    
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
    
    # Inference Engine settings
    use_inference_engine: bool = False
    max_inference_depth: int = 5
    inference_confidence_threshold: float = 0.6
    max_forward_chain_iterations: int = 3
    abductive_reasoning_enabled: bool = True
    max_abductive_hypotheses: int = 5
    
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
            
        # Ontology backward compatibility
        if not hasattr(self, 'use_ontology'):
            self.use_ontology = False
            
        if not hasattr(self, 'enable_ontology_evolution'):
            self.enable_ontology_evolution = True
            
        if not hasattr(self, 'ontology_confidence_threshold'):
            self.ontology_confidence_threshold = 0.7
            
        # Metadata framework backward compatibility
        if not hasattr(self, 'track_provenance'):
            self.track_provenance = True
            
        if not hasattr(self, 'track_confidence'):
            self.track_confidence = True
            
        if not hasattr(self, 'track_temporal'):
            self.track_temporal = True
            
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
        
        # Uncertainty Handling backward compatibility
        if not hasattr(self, 'use_uncertainty_handling'):
            self.use_uncertainty_handling = False
            
        if not hasattr(self, 'auto_detect_contradictions'):
            self.auto_detect_contradictions = True
            
        if not hasattr(self, 'default_contradiction_strategy'):
            self.default_contradiction_strategy = "weighted"
            
        if not hasattr(self, 'recency_weight'):
            self.recency_weight = 0.6
            
        if not hasattr(self, 'reliability_weight'):
            self.reliability_weight = 0.4
            
        if not hasattr(self, 'confidence_threshold'):
            self.confidence_threshold = 0.7
            
        if not hasattr(self, 'uncertainty_representation'):
            self.uncertainty_representation = "confidence"
            
        if not hasattr(self, 'reason_with_incomplete_info'):
            self.reason_with_incomplete_info = True
            
        # Performance Optimization backward compatibility
        if not hasattr(self, 'use_performance_optimization'):
            self.use_performance_optimization = False
            
        if not hasattr(self, 'use_graph_partitioning'):
            self.use_graph_partitioning = False
            
        if not hasattr(self, 'graph_partition_method'):
            self.graph_partition_method = "louvain"
            
        if not hasattr(self, 'target_partition_count'):
            self.target_partition_count = 5
            
        if not hasattr(self, 'use_multihop_indexing'):
            self.use_multihop_indexing = False
            
        if not hasattr(self, 'max_indexed_hops'):
            self.max_indexed_hops = 2
            
        if not hasattr(self, 'use_query_planning'):
            self.use_query_planning = True
            
        if not hasattr(self, 'use_reasoning_cache'):
            self.use_reasoning_cache = True
            
        if not hasattr(self, 'reasoning_cache_max_size'):
            self.reasoning_cache_max_size = 1000
            
        if not hasattr(self, 'query_cache_max_size'):
            self.query_cache_max_size = 500
            
        if not hasattr(self, 'cache_ttl'):
            self.cache_ttl = 3600
        
        # Inference Engine backward compatibility
        if not hasattr(self, 'use_inference_engine'):
            self.use_inference_engine = False
            
        if not hasattr(self, 'max_inference_depth'):
            self.max_inference_depth = 5
            
        if not hasattr(self, 'inference_confidence_threshold'):
            self.inference_confidence_threshold = 0.6
            
        if not hasattr(self, 'max_forward_chain_iterations'):
            self.max_forward_chain_iterations = 3
            
        if not hasattr(self, 'abductive_reasoning_enabled'):
            self.abductive_reasoning_enabled = True
            
        if not hasattr(self, 'max_abductive_hypotheses'):
            self.max_abductive_hypotheses = 5
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CortexFlowConfig':
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            CortexFlowConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = {
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
            "use_ontology": self.use_ontology,
            "enable_ontology_evolution": self.enable_ontology_evolution,
            "ontology_confidence_threshold": self.ontology_confidence_threshold,
            "track_provenance": self.track_provenance,
            "track_confidence": self.track_confidence,
            "track_temporal": self.track_temporal,
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
            "custom_config": self.custom_config,
            "use_inference_engine": self.use_inference_engine,
            "max_inference_depth": self.max_inference_depth,
            "inference_confidence_threshold": self.inference_confidence_threshold,
            "max_forward_chain_iterations": self.max_forward_chain_iterations,
            "abductive_reasoning_enabled": self.abductive_reasoning_enabled,
            "max_abductive_hypotheses": self.max_abductive_hypotheses,
            "use_uncertainty_handling": self.use_uncertainty_handling,
            "auto_detect_contradictions": self.auto_detect_contradictions,
            "default_contradiction_strategy": self.default_contradiction_strategy,
            "recency_weight": self.recency_weight,
            "reliability_weight": self.reliability_weight,
            "confidence_threshold": self.confidence_threshold,
            "uncertainty_representation": self.uncertainty_representation,
            "reason_with_incomplete_info": self.reason_with_incomplete_info,
            "use_performance_optimization": self.use_performance_optimization,
            "use_graph_partitioning": self.use_graph_partitioning,
            "graph_partition_method": self.graph_partition_method,
            "target_partition_count": self.target_partition_count,
            "use_multihop_indexing": self.use_multihop_indexing,
            "max_indexed_hops": self.max_indexed_hops,
            "use_query_planning": self.use_query_planning,
            "use_reasoning_cache": self.use_reasoning_cache,
            "reasoning_cache_max_size": self.reasoning_cache_max_size,
            "query_cache_max_size": self.query_cache_max_size,
            "cache_ttl": self.cache_ttl,
        }
        return config_dict
    
    def log_config(self):
        """Log the current configuration."""
        print("CortexFlow Configuration:")
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}") 