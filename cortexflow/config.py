"""
CortexFlow Configuration module.

This module provides the configuration class for the CortexFlow system.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, List

@dataclass
class MemoryConfig:
    """Memory configuration settings."""
    active_token_limit: int = 4096
    working_token_limit: int = 8192
    archive_token_limit: int = 16384
    use_dynamic_weighting: bool = False
    dynamic_weighting_learning_rate: float = 0.1
    dynamic_weighting_min_tier_size: int = 1000
    dynamic_weighting_default_ratios: Dict[str, float] = field(default_factory=lambda: {
        "active": 0.25,
        "working": 0.35,
        "archive": 0.40
    })

@dataclass
class KnowledgeStoreConfig:
    """Knowledge store configuration settings."""
    knowledge_store_path: str = "cortexflow.db"
    retrieval_type: str = "hybrid"
    trust_marker: str = "📚" 
    use_reranking: bool = True
    rerank_top_k: int = 15
    vector_model: str = "all-MiniLM-L6-v2"

@dataclass
class GraphRagConfig:
    """Graph RAG configuration settings."""
    use_graph_rag: bool = False
    enable_multi_hop_queries: bool = False
    max_graph_hops: int = 3
    graph_weight: float = 0.5
    use_graph_partitioning: bool = False
    graph_partition_method: str = "louvain"
    target_partition_count: int = 5
    use_multihop_indexing: bool = False
    max_indexed_hops: int = 2

@dataclass
class OntologyConfig:
    """Ontology configuration settings."""
    use_ontology: bool = False
    enable_ontology_evolution: bool = True
    ontology_confidence_threshold: float = 0.7

@dataclass
class MetadataConfig:
    """Metadata framework configuration settings."""
    track_provenance: bool = True
    track_confidence: bool = True
    track_temporal: bool = True

@dataclass
class AgentConfig:
    """Chain of Agents configuration settings."""
    use_chain_of_agents: bool = False
    chain_complexity_threshold: int = 5
    chain_agent_count: int = 3

@dataclass
class ReflectionConfig:
    """Self-Reflection configuration settings."""
    use_self_reflection: bool = False
    reflection_relevance_threshold: float = 0.6
    reflection_confidence_threshold: float = 0.7

@dataclass
class UncertaintyConfig:
    """Uncertainty Handling configuration settings."""
    use_uncertainty_handling: bool = False
    auto_detect_contradictions: bool = True
    default_contradiction_strategy: str = "weighted"
    recency_weight: float = 0.6
    reliability_weight: float = 0.4
    confidence_threshold: float = 0.7
    uncertainty_representation: str = "confidence"
    reason_with_incomplete_info: bool = True

@dataclass
class PerformanceConfig:
    """Performance Optimization configuration settings."""
    use_performance_optimization: bool = False
    use_query_planning: bool = True
    use_reasoning_cache: bool = True
    reasoning_cache_max_size: int = 1000
    query_cache_max_size: int = 500
    cache_ttl: int = 3600

@dataclass
class LLMConfig:
    """LLM Integration configuration settings."""
    default_model: str = "gemma3:1b"
    ollama_host: str = "http://localhost:11434"
    conversation_style: str = "casual"
    system_persona: str = "helpful assistant"

@dataclass
class ClassifierConfig:
    """ML Classifier configuration settings."""
    use_ml_classifier: bool = False
    classifier_model: str = "all-MiniLM-L6-v2"
    classifier_threshold: float = 0.7

@dataclass
class InferenceConfig:
    """Inference Engine configuration settings."""
    use_inference_engine: bool = False
    max_inference_depth: int = 5
    inference_confidence_threshold: float = 0.6
    max_forward_chain_iterations: int = 3
    abductive_reasoning_enabled: bool = True
    max_abductive_hypotheses: int = 5

@dataclass
class CortexFlowConfig:
    """Configuration for CortexFlow."""
    
    # Configuration sections
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    knowledge_store: KnowledgeStoreConfig = field(default_factory=KnowledgeStoreConfig)
    graph_rag: GraphRagConfig = field(default_factory=GraphRagConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Debug and logging settings
    verbose_logging: bool = False
    debug_mode: bool = False
    
    # Optional custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize any derived settings after creation."""
        # Ensure path is absolute
        if not os.path.isabs(self.knowledge_store.knowledge_store_path):
            # If relative, make it relative to the current working directory
            self.knowledge_store.knowledge_store_path = os.path.join(
                os.getcwd(), self.knowledge_store.knowledge_store_path
            )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CortexFlowConfig':
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            CortexFlowConfig instance
        """
        # Create config sections
        memory_dict = {k: v for k, v in config_dict.items() 
                      if k in [f.name for f in fields(MemoryConfig)]}
        knowledge_dict = {k: v for k, v in config_dict.items() 
                         if k in [f.name for f in fields(KnowledgeStoreConfig)]}
        graph_dict = {k: v for k, v in config_dict.items() 
                     if k in [f.name for f in fields(GraphRagConfig)]}
        ontology_dict = {k: v for k, v in config_dict.items() 
                        if k in [f.name for f in fields(OntologyConfig)]}
        metadata_dict = {k: v for k, v in config_dict.items() 
                        if k in [f.name for f in fields(MetadataConfig)]}
        agent_dict = {k: v for k, v in config_dict.items() 
                     if k in [f.name for f in fields(AgentConfig)]}
        reflection_dict = {k: v for k, v in config_dict.items() 
                          if k in [f.name for f in fields(ReflectionConfig)]}
        uncertainty_dict = {k: v for k, v in config_dict.items() 
                           if k in [f.name for f in fields(UncertaintyConfig)]}
        performance_dict = {k: v for k, v in config_dict.items() 
                           if k in [f.name for f in fields(PerformanceConfig)]}
        llm_dict = {k: v for k, v in config_dict.items() 
                   if k in [f.name for f in fields(LLMConfig)]}
        classifier_dict = {k: v for k, v in config_dict.items() 
                          if k in [f.name for f in fields(ClassifierConfig)]}
        inference_dict = {k: v for k, v in config_dict.items() 
                         if k in [f.name for f in fields(InferenceConfig)]}
        
        # Create top-level config dict with just the top-level items
        top_config_dict = {
            "verbose_logging": config_dict.get("verbose_logging", False),
            "debug_mode": config_dict.get("debug_mode", False),
            "custom_config": config_dict.get("custom_config", {})
        }
        
        # Create section objects if any fields were provided
        if memory_dict:
            top_config_dict["memory"] = MemoryConfig(**memory_dict)
        if knowledge_dict:
            top_config_dict["knowledge_store"] = KnowledgeStoreConfig(**knowledge_dict)
        if graph_dict:
            top_config_dict["graph_rag"] = GraphRagConfig(**graph_dict)
        if ontology_dict:
            top_config_dict["ontology"] = OntologyConfig(**ontology_dict)
        if metadata_dict:
            top_config_dict["metadata"] = MetadataConfig(**metadata_dict)
        if agent_dict:
            top_config_dict["agents"] = AgentConfig(**agent_dict)
        if reflection_dict:
            top_config_dict["reflection"] = ReflectionConfig(**reflection_dict)
        if uncertainty_dict:
            top_config_dict["uncertainty"] = UncertaintyConfig(**uncertainty_dict)
        if performance_dict:
            top_config_dict["performance"] = PerformanceConfig(**performance_dict)
        if llm_dict:
            top_config_dict["llm"] = LLMConfig(**llm_dict)
        if classifier_dict:
            top_config_dict["classifier"] = ClassifierConfig(**classifier_dict)
        if inference_dict:
            top_config_dict["inference"] = InferenceConfig(**inference_dict)
        
        return cls(**top_config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = {
            # Memory settings
            "active_token_limit": self.memory.active_token_limit,
            "working_token_limit": self.memory.working_token_limit,
            "archive_token_limit": self.memory.archive_token_limit,
            "use_dynamic_weighting": self.memory.use_dynamic_weighting,
            "dynamic_weighting_learning_rate": self.memory.dynamic_weighting_learning_rate,
            "dynamic_weighting_min_tier_size": self.memory.dynamic_weighting_min_tier_size,
            "dynamic_weighting_default_ratios": self.memory.dynamic_weighting_default_ratios,
            
            # Knowledge store settings
            "knowledge_store_path": self.knowledge_store.knowledge_store_path,
            "retrieval_type": self.knowledge_store.retrieval_type,
            "trust_marker": self.knowledge_store.trust_marker,
            "use_reranking": self.knowledge_store.use_reranking,
            "rerank_top_k": self.knowledge_store.rerank_top_k,
            "vector_model": self.knowledge_store.vector_model,
            
            # Graph RAG settings
            "use_graph_rag": self.graph_rag.use_graph_rag,
            "enable_multi_hop_queries": self.graph_rag.enable_multi_hop_queries,
            "max_graph_hops": self.graph_rag.max_graph_hops,
            "graph_weight": self.graph_rag.graph_weight,
            "use_graph_partitioning": self.graph_rag.use_graph_partitioning,
            "graph_partition_method": self.graph_rag.graph_partition_method,
            "target_partition_count": self.graph_rag.target_partition_count,
            "use_multihop_indexing": self.graph_rag.use_multihop_indexing,
            "max_indexed_hops": self.graph_rag.max_indexed_hops,
            
            # Ontology settings
            "use_ontology": self.ontology.use_ontology,
            "enable_ontology_evolution": self.ontology.enable_ontology_evolution,
            "ontology_confidence_threshold": self.ontology.ontology_confidence_threshold,
            
            # Metadata framework settings
            "track_provenance": self.metadata.track_provenance,
            "track_confidence": self.metadata.track_confidence,
            "track_temporal": self.metadata.track_temporal,
            
            # Chain of Agents settings
            "use_chain_of_agents": self.agents.use_chain_of_agents,
            "chain_complexity_threshold": self.agents.chain_complexity_threshold,
            "chain_agent_count": self.agents.chain_agent_count,
            
            # Self-Reflection settings
            "use_self_reflection": self.reflection.use_self_reflection,
            "reflection_relevance_threshold": self.reflection.reflection_relevance_threshold,
            "reflection_confidence_threshold": self.reflection.reflection_confidence_threshold,
            
            # Uncertainty Handling settings
            "use_uncertainty_handling": self.uncertainty.use_uncertainty_handling,
            "auto_detect_contradictions": self.uncertainty.auto_detect_contradictions,
            "default_contradiction_strategy": self.uncertainty.default_contradiction_strategy,
            "recency_weight": self.uncertainty.recency_weight,
            "reliability_weight": self.uncertainty.reliability_weight,
            "confidence_threshold": self.uncertainty.confidence_threshold,
            "uncertainty_representation": self.uncertainty.uncertainty_representation,
            "reason_with_incomplete_info": self.uncertainty.reason_with_incomplete_info,
            
            # Performance Optimization settings
            "use_performance_optimization": self.performance.use_performance_optimization,
            "use_query_planning": self.performance.use_query_planning,
            "use_reasoning_cache": self.performance.use_reasoning_cache,
            "reasoning_cache_max_size": self.performance.reasoning_cache_max_size,
            "query_cache_max_size": self.performance.query_cache_max_size,
            "cache_ttl": self.performance.cache_ttl,
            
            # LLM Integration
            "default_model": self.llm.default_model,
            "ollama_host": self.llm.ollama_host,
            "conversation_style": self.llm.conversation_style,
            "system_persona": self.llm.system_persona,
            
            # Classifier settings
            "use_ml_classifier": self.classifier.use_ml_classifier,
            "classifier_model": self.classifier.classifier_model,
            "classifier_threshold": self.classifier.classifier_threshold,
            
            # Inference Engine settings
            "use_inference_engine": self.inference.use_inference_engine,
            "max_inference_depth": self.inference.max_inference_depth,
            "inference_confidence_threshold": self.inference.inference_confidence_threshold,
            "max_forward_chain_iterations": self.inference.max_forward_chain_iterations,
            "abductive_reasoning_enabled": self.inference.abductive_reasoning_enabled,
            "max_abductive_hypotheses": self.inference.max_abductive_hypotheses,
            
            # Debug and logging settings
            "verbose_logging": self.verbose_logging,
            "debug_mode": self.debug_mode,
            
            # Custom config
            "custom_config": self.custom_config,
        }
        return config_dict
    
    def log_config(self):
        """Log the current configuration."""
        print("CortexFlow Configuration:")
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}") 

class ConfigBuilder:
    """Builder class for CortexFlowConfig to allow fluent configuration."""
    
    def __init__(self):
        self._memory = MemoryConfig()
        self._knowledge_store = KnowledgeStoreConfig()
        self._graph_rag = GraphRagConfig()
        self._ontology = OntologyConfig()
        self._metadata = MetadataConfig()
        self._agents = AgentConfig()
        self._reflection = ReflectionConfig()
        self._uncertainty = UncertaintyConfig()
        self._performance = PerformanceConfig()
        self._llm = LLMConfig()
        self._classifier = ClassifierConfig()
        self._inference = InferenceConfig()
        self._verbose_logging = False
        self._debug_mode = False
        self._custom_config = {}
    
    def with_memory(self, **kwargs) -> 'ConfigBuilder':
        """Configure memory settings."""
        for key, value in kwargs.items():
            setattr(self._memory, key, value)
        return self
    
    def with_knowledge_store(self, **kwargs) -> 'ConfigBuilder':
        """Configure knowledge store settings."""
        for key, value in kwargs.items():
            setattr(self._knowledge_store, key, value)
        return self
    
    def with_graph_rag(self, **kwargs) -> 'ConfigBuilder':
        """Configure graph RAG settings."""
        for key, value in kwargs.items():
            setattr(self._graph_rag, key, value)
        return self
    
    def with_ontology(self, **kwargs) -> 'ConfigBuilder':
        """Configure ontology settings."""
        for key, value in kwargs.items():
            setattr(self._ontology, key, value)
        return self
    
    def with_metadata(self, **kwargs) -> 'ConfigBuilder':
        """Configure metadata framework settings."""
        for key, value in kwargs.items():
            setattr(self._metadata, key, value)
        return self
    
    def with_agents(self, **kwargs) -> 'ConfigBuilder':
        """Configure chain of agents settings."""
        for key, value in kwargs.items():
            setattr(self._agents, key, value)
        return self
    
    def with_reflection(self, **kwargs) -> 'ConfigBuilder':
        """Configure self-reflection settings."""
        for key, value in kwargs.items():
            setattr(self._reflection, key, value)
        return self
    
    def with_uncertainty(self, **kwargs) -> 'ConfigBuilder':
        """Configure uncertainty handling settings."""
        for key, value in kwargs.items():
            setattr(self._uncertainty, key, value)
        return self
    
    def with_performance(self, **kwargs) -> 'ConfigBuilder':
        """Configure performance optimization settings."""
        for key, value in kwargs.items():
            setattr(self._performance, key, value)
        return self
    
    def with_llm(self, **kwargs) -> 'ConfigBuilder':
        """Configure LLM integration settings."""
        for key, value in kwargs.items():
            setattr(self._llm, key, value)
        return self
    
    def with_classifier(self, **kwargs) -> 'ConfigBuilder':
        """Configure classifier settings."""
        for key, value in kwargs.items():
            setattr(self._classifier, key, value)
        return self
    
    def with_inference(self, **kwargs) -> 'ConfigBuilder':
        """Configure inference engine settings."""
        for key, value in kwargs.items():
            setattr(self._inference, key, value)
        return self
    
    def with_debug(self, verbose_logging: bool = None, debug_mode: bool = None) -> 'ConfigBuilder':
        """Configure debug and logging settings."""
        if verbose_logging is not None:
            self._verbose_logging = verbose_logging
        if debug_mode is not None:
            self._debug_mode = debug_mode
        return self
    
    def with_custom_config(self, custom_config: Dict[str, Any]) -> 'ConfigBuilder':
        """Configure custom settings."""
        self._custom_config = custom_config
        return self
    
    def build(self) -> CortexFlowConfig:
        """Build the final configuration object."""
        return CortexFlowConfig(
            memory=self._memory,
            knowledge_store=self._knowledge_store,
            graph_rag=self._graph_rag,
            ontology=self._ontology,
            metadata=self._metadata,
            agents=self._agents,
            reflection=self._reflection,
            uncertainty=self._uncertainty,
            performance=self._performance,
            llm=self._llm,
            classifier=self._classifier,
            inference=self._inference,
            verbose_logging=self._verbose_logging,
            debug_mode=self._debug_mode,
            custom_config=self._custom_config
        ) 