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
    # Backend selection
    backend: str = "ollama"           # "ollama" or "vertex_ai"
    # Vertex AI fields (None = read from env vars at runtime)
    vertex_project_id: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_api_key: Optional[str] = None
    vertex_credentials_path: Optional[str] = None
    vertex_model: str = "gemini-1.5-flash"

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

    def __getattr__(self, name: str):
        """Proxy flat attribute access to nested config sections (backward compat)."""
        for sub_name in ('memory', 'knowledge_store', 'graph_rag', 'ontology', 'metadata',
                         'agents', 'reflection', 'uncertainty', 'performance', 'llm',
                         'classifier', 'inference'):
            try:
                sub = object.__getattribute__(self, sub_name)
                if name in {f.name for f in fields(type(sub))}:
                    return getattr(sub, name)
            except AttributeError:
                continue
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @staticmethod
    def _extract_section(config_dict: Dict[str, Any], config_class) -> Dict[str, Any]:
        """Extract keys from config_dict that belong to the given dataclass."""
        valid_fields = {f.name for f in fields(config_class)}
        return {k: v for k, v in config_dict.items() if k in valid_fields}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CortexFlowConfig':
        """
        Create configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            CortexFlowConfig instance
        """
        # Mapping of CortexFlowConfig field names to their dataclass types
        _section_map = {
            "memory": MemoryConfig,
            "knowledge_store": KnowledgeStoreConfig,
            "graph_rag": GraphRagConfig,
            "ontology": OntologyConfig,
            "metadata": MetadataConfig,
            "agents": AgentConfig,
            "reflection": ReflectionConfig,
            "uncertainty": UncertaintyConfig,
            "performance": PerformanceConfig,
            "llm": LLMConfig,
            "classifier": ClassifierConfig,
            "inference": InferenceConfig,
        }

        # Create top-level config dict with just the top-level items
        top_config_dict = {
            "verbose_logging": config_dict.get("verbose_logging", False),
            "debug_mode": config_dict.get("debug_mode", False),
            "custom_config": config_dict.get("custom_config", {})
        }

        # Create section objects if any matching fields were provided
        for section_name, config_class in _section_map.items():
            section_dict = cls._extract_section(config_dict, config_class)
            if section_dict:
                top_config_dict[section_name] = config_class(**section_dict)

        return cls(**top_config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a flat dictionary.

        Returns:
            Dictionary representation of configuration
        """
        config_dict: Dict[str, Any] = {}

        # Flatten all nested dataclass sections into a single dict
        for section_field in fields(self):
            value = getattr(self, section_field.name)
            if hasattr(value, '__dataclass_fields__'):
                # It's a nested dataclass section -- flatten its fields
                for f in fields(value):
                    config_dict[f.name] = getattr(value, f.name)
            else:
                # Top-level scalar field
                config_dict[section_field.name] = value

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
    
    def _set_section(self, section_name: str, **kwargs) -> 'ConfigBuilder':
        """Apply keyword arguments to the named config section."""
        section = getattr(self, section_name)
        for key, value in kwargs.items():
            setattr(section, key, value)
        return self

    def with_memory(self, **kwargs) -> 'ConfigBuilder':
        """Configure memory settings."""
        return self._set_section('_memory', **kwargs)

    def with_knowledge_store(self, **kwargs) -> 'ConfigBuilder':
        """Configure knowledge store settings."""
        return self._set_section('_knowledge_store', **kwargs)

    def with_graph_rag(self, **kwargs) -> 'ConfigBuilder':
        """Configure graph RAG settings."""
        return self._set_section('_graph_rag', **kwargs)

    def with_ontology(self, **kwargs) -> 'ConfigBuilder':
        """Configure ontology settings."""
        return self._set_section('_ontology', **kwargs)

    def with_metadata(self, **kwargs) -> 'ConfigBuilder':
        """Configure metadata framework settings."""
        return self._set_section('_metadata', **kwargs)

    def with_agents(self, **kwargs) -> 'ConfigBuilder':
        """Configure chain of agents settings."""
        return self._set_section('_agents', **kwargs)

    def with_reflection(self, **kwargs) -> 'ConfigBuilder':
        """Configure self-reflection settings."""
        return self._set_section('_reflection', **kwargs)

    def with_uncertainty(self, **kwargs) -> 'ConfigBuilder':
        """Configure uncertainty handling settings."""
        return self._set_section('_uncertainty', **kwargs)

    def with_performance(self, **kwargs) -> 'ConfigBuilder':
        """Configure performance optimization settings."""
        return self._set_section('_performance', **kwargs)

    def with_llm(self, **kwargs) -> 'ConfigBuilder':
        """Configure LLM integration settings."""
        return self._set_section('_llm', **kwargs)

    def with_vertex_ai(self, project_id=None, location=None, default_model="gemini-1.5-flash",
                       api_key=None, credentials_path=None) -> 'ConfigBuilder':
        """Configure Vertex AI as the LLM backend."""
        self._llm.backend = "vertex_ai"
        self._llm.default_model = default_model
        self._llm.vertex_model = default_model
        if project_id:
            self._llm.vertex_project_id = project_id
        if location:
            self._llm.vertex_location = location
        if api_key:
            self._llm.vertex_api_key = api_key
        if credentials_path:
            self._llm.vertex_credentials_path = credentials_path
        return self

    def with_classifier(self, **kwargs) -> 'ConfigBuilder':
        """Configure classifier settings."""
        return self._set_section('_classifier', **kwargs)

    def with_inference(self, **kwargs) -> 'ConfigBuilder':
        """Configure inference engine settings."""
        return self._set_section('_inference', **kwargs)
    
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