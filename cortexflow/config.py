"""
CortexFlow Configuration module.

This module provides the configuration class for the CortexFlow system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class MemoryConfig:
    """Memory configuration settings."""

    active_token_limit: int = 4096
    working_token_limit: int = 8192
    archive_token_limit: int = 16384
    use_dynamic_weighting: bool = False
    dynamic_weighting_learning_rate: float = 0.1
    dynamic_weighting_min_tier_size: int = 1000
    dynamic_weighting_default_ratios: dict[str, float] = field(
        default_factory=lambda: {"active": 0.25, "working": 0.35, "archive": 0.40}
    )
    use_fact_extraction: bool = False


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
    backend: str = "ollama"  # "ollama" or "vertex_ai"
    # Vertex AI fields (None = read from env vars at runtime)
    vertex_project_id: str | None = None
    vertex_location: str | None = None
    vertex_api_key: str | None = None
    vertex_credentials_path: str | None = None
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


# ---- Companion AI config sections (Phase 1-3) ----


@dataclass
class SessionConfig:
    """Session management configuration."""

    enable_sessions: bool = False
    default_user_id: str = "default"
    session_ttl: int = 86400  # seconds
    max_sessions_per_user: int = 10
    session_db_path: str = ":memory:"


@dataclass
class EmotionConfig:
    """Emotion tracking configuration."""

    use_emotion_tracking: bool = False
    emotion_detector: str = "rule"  # "rule" or "llm"
    emotion_window_size: int = 20
    emotion_influence_on_response: float = 0.5


@dataclass
class PersonaConfig:
    """Persona management configuration."""

    use_personas: bool = False
    default_persona_id: str | None = None
    persona_db_path: str = ":memory:"


@dataclass
class RelationshipConfig:
    """Relationship tracking configuration."""

    use_relationship_tracking: bool = False
    relationship_db_path: str = ":memory:"


@dataclass
class EventConfig:
    """Event system configuration."""

    use_events: bool = False


# ---- Phase 4: Temporal & Episodic config sections ----


@dataclass
class TemporalConfig:
    """Temporal fact management configuration."""

    use_temporal_facts: bool = False
    temporal_db_path: str = ":memory:"


@dataclass
class EpisodicConfig:
    """Episodic memory configuration."""

    use_episodic_memory: bool = False
    episodic_db_path: str = ":memory:"
    auto_summarize_on_session_close: bool = True


@dataclass
class SafetyConfig:
    """Safety pipeline configuration."""

    use_safety_pipeline: bool = False
    enable_pii_detection: bool = True
    enable_boundary_enforcement: bool = True
    custom_blocked_patterns: list[str] = field(default_factory=list)
    block_on_safety_violation: bool = False  # If True, block message entirely


@dataclass
class VectorStoreConfig:
    """Vector store backend configuration."""

    backend: str = "sqlite"  # "sqlite", "chromadb", "qdrant"
    collection_name: str = "cortexflow"
    # ChromaDB settings
    chromadb_host: str | None = None
    chromadb_port: int | None = None
    chromadb_path: str | None = None  # persistence path
    # Qdrant settings
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str | None = None  # local persistence path
    embedding_dimension: int = 384


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

    # Companion AI sections (all disabled by default for backward compat)
    session: SessionConfig = field(default_factory=SessionConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    relationship: RelationshipConfig = field(default_factory=RelationshipConfig)

    # Event system
    events: EventConfig = field(default_factory=EventConfig)

    # Temporal & Episodic memory
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)

    # Safety pipeline
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Vector store backend
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    # Debug and logging settings
    verbose_logging: bool = False
    debug_mode: bool = False

    # Optional custom configuration
    custom_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize any derived settings after creation."""
        # Ensure path is absolute
        if not os.path.isabs(self.knowledge_store.knowledge_store_path):
            # If relative, make it relative to the current working directory
            self.knowledge_store.knowledge_store_path = os.path.join(
                os.getcwd(), self.knowledge_store.knowledge_store_path
            )

    # Pre-computed mapping: flat field name -> (section_attr_name, field_name)
    # Built once per class, not per instance.
    _FIELD_MAP: dict[str, str] | None = None

    @classmethod
    def _build_field_map(cls) -> dict[str, str]:
        """Build a mapping from flat field names to their section attribute names."""
        if cls._FIELD_MAP is not None:
            return cls._FIELD_MAP
        mapping: dict[str, str] = {}
        for section_name in (
            "memory",
            "knowledge_store",
            "graph_rag",
            "ontology",
            "metadata",
            "agents",
            "reflection",
            "uncertainty",
            "performance",
            "llm",
            "classifier",
            "inference",
            "session",
            "emotion",
            "persona",
            "relationship",
            "events",
            "temporal",
            "episodic",
            "safety",
            "vector_store",
        ):
            section_cls = {f.name: f for f in fields(cls)}[section_name].default_factory  # type: ignore[union-attr]
            for f in fields(section_cls):
                # First-write wins -- earlier sections have priority (matches old behavior)
                if f.name not in mapping:
                    mapping[f.name] = section_name
        cls._FIELD_MAP = mapping
        return mapping

    def __getattr__(self, name: str):
        """Proxy flat attribute access to nested config sections (backward compat).

        Uses a class-level cached field map so lookups are O(1) after the
        first access instead of traversing all 12 sub-configs every time.
        """
        field_map = CortexFlowConfig._build_field_map()
        section_name = field_map.get(name)
        if section_name is not None:
            sub = object.__getattribute__(self, section_name)
            return getattr(sub, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @staticmethod
    def _extract_section(config_dict: dict[str, Any], config_class) -> dict[str, Any]:
        """Extract keys from config_dict that belong to the given dataclass."""
        valid_fields = {f.name for f in fields(config_class)}
        return {k: v for k, v in config_dict.items() if k in valid_fields}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CortexFlowConfig:
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
            "session": SessionConfig,
            "emotion": EmotionConfig,
            "persona": PersonaConfig,
            "relationship": RelationshipConfig,
            "events": EventConfig,
            "temporal": TemporalConfig,
            "episodic": EpisodicConfig,
            "safety": SafetyConfig,
            "vector_store": VectorStoreConfig,
        }

        # Create top-level config dict with just the top-level items
        top_config_dict = {
            "verbose_logging": config_dict.get("verbose_logging", False),
            "debug_mode": config_dict.get("debug_mode", False),
            "custom_config": config_dict.get("custom_config", {}),
        }

        # Create section objects if any matching fields were provided
        for section_name, config_class in _section_map.items():
            section_dict = cls._extract_section(config_dict, config_class)
            if section_dict:
                top_config_dict[section_name] = config_class(**section_dict)

        return cls(**top_config_dict)

    # Fields that must never appear in logs or serialized output
    _SENSITIVE_FIELDS = frozenset(
        {
            "vertex_api_key",
            "vertex_credentials_path",
            "qdrant_api_key",
        }
    )

    def to_dict(self, *, redact: bool = False) -> dict[str, Any]:
        """
        Convert configuration to a flat dictionary.

        Args:
            redact: If True, replace sensitive values with ``"***"``.

        Returns:
            Dictionary representation of configuration
        """
        config_dict: dict[str, Any] = {}

        # Flatten all nested dataclass sections into a single dict
        for section_field in fields(self):
            value = getattr(self, section_field.name)
            if hasattr(value, "__dataclass_fields__"):
                # It's a nested dataclass section -- flatten its fields
                for f in fields(value):
                    val = getattr(value, f.name)
                    if redact and f.name in self._SENSITIVE_FIELDS and val:
                        val = "***"
                    config_dict[f.name] = val
            else:
                # Top-level scalar field
                config_dict[section_field.name] = value

        return config_dict

    def log_config(self):
        """Log the current configuration (sensitive fields are redacted)."""
        import logging as _logging

        _logger = _logging.getLogger("cortexflow")
        _logger.info("CortexFlow Configuration:")
        for key, value in self.to_dict(redact=True).items():
            _logger.info("  %s: %s", key, value)


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
        self._session = SessionConfig()
        self._emotion = EmotionConfig()
        self._persona = PersonaConfig()
        self._relationship = RelationshipConfig()
        self._events = EventConfig()
        self._temporal = TemporalConfig()
        self._episodic = EpisodicConfig()
        self._safety = SafetyConfig()
        self._vector_store = VectorStoreConfig()
        self._verbose_logging = False
        self._debug_mode = False
        self._custom_config = {}

    def _set_section(self, section_name: str, **kwargs) -> ConfigBuilder:
        """Apply keyword arguments to the named config section."""
        section = getattr(self, section_name)
        for key, value in kwargs.items():
            setattr(section, key, value)
        return self

    def with_memory(self, **kwargs) -> ConfigBuilder:
        """Configure memory settings."""
        return self._set_section("_memory", **kwargs)

    def with_knowledge_store(self, **kwargs) -> ConfigBuilder:
        """Configure knowledge store settings."""
        return self._set_section("_knowledge_store", **kwargs)

    def with_graph_rag(self, **kwargs) -> ConfigBuilder:
        """Configure graph RAG settings."""
        return self._set_section("_graph_rag", **kwargs)

    def with_ontology(self, **kwargs) -> ConfigBuilder:
        """Configure ontology settings."""
        return self._set_section("_ontology", **kwargs)

    def with_metadata(self, **kwargs) -> ConfigBuilder:
        """Configure metadata framework settings."""
        return self._set_section("_metadata", **kwargs)

    def with_agents(self, **kwargs) -> ConfigBuilder:
        """Configure chain of agents settings."""
        return self._set_section("_agents", **kwargs)

    def with_reflection(self, **kwargs) -> ConfigBuilder:
        """Configure self-reflection settings."""
        return self._set_section("_reflection", **kwargs)

    def with_uncertainty(self, **kwargs) -> ConfigBuilder:
        """Configure uncertainty handling settings."""
        return self._set_section("_uncertainty", **kwargs)

    def with_performance(self, **kwargs) -> ConfigBuilder:
        """Configure performance optimization settings."""
        return self._set_section("_performance", **kwargs)

    def with_llm(self, **kwargs) -> ConfigBuilder:
        """Configure LLM integration settings."""
        return self._set_section("_llm", **kwargs)

    def with_vertex_ai(
        self,
        project_id=None,
        location=None,
        default_model="gemini-1.5-flash",
        api_key=None,
        credentials_path=None,
    ) -> ConfigBuilder:
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

    def with_classifier(self, **kwargs) -> ConfigBuilder:
        """Configure classifier settings."""
        return self._set_section("_classifier", **kwargs)

    def with_inference(self, **kwargs) -> ConfigBuilder:
        """Configure inference engine settings."""
        return self._set_section("_inference", **kwargs)

    def with_sessions(self, **kwargs) -> ConfigBuilder:
        """Enable and configure session management."""
        self._session.enable_sessions = True
        return self._set_section("_session", **kwargs)

    def with_emotions(self, **kwargs) -> ConfigBuilder:
        """Enable and configure emotion tracking."""
        self._emotion.use_emotion_tracking = True
        return self._set_section("_emotion", **kwargs)

    def with_persona(self, **kwargs) -> ConfigBuilder:
        """Enable and configure persona management."""
        self._persona.use_personas = True
        return self._set_section("_persona", **kwargs)

    def with_relationship(self, **kwargs) -> ConfigBuilder:
        """Enable and configure relationship tracking."""
        self._relationship.use_relationship_tracking = True
        return self._set_section("_relationship", **kwargs)

    def with_events(self, **kwargs) -> ConfigBuilder:
        """Enable and configure event system."""
        self._events.use_events = True
        return self._set_section("_events", **kwargs)

    def with_temporal(self, **kwargs) -> ConfigBuilder:
        """Enable and configure temporal fact management."""
        self._temporal.use_temporal_facts = True
        return self._set_section("_temporal", **kwargs)

    def with_episodic(self, **kwargs) -> ConfigBuilder:
        """Enable and configure episodic memory."""
        self._episodic.use_episodic_memory = True
        return self._set_section("_episodic", **kwargs)

    def with_safety(self, **kwargs) -> ConfigBuilder:
        """Enable and configure the safety pipeline."""
        self._safety.use_safety_pipeline = True
        return self._set_section("_safety", **kwargs)

    def with_vector_store(self, **kwargs) -> ConfigBuilder:
        """Configure vector store backend."""
        return self._set_section("_vector_store", **kwargs)

    def with_fact_extraction(self, enabled: bool = True) -> ConfigBuilder:
        """Enable or disable personal fact extraction for deep memory recall."""
        self._memory.use_fact_extraction = enabled
        return self

    def with_debug(
        self, verbose_logging: bool = None, debug_mode: bool = None
    ) -> ConfigBuilder:
        """Configure debug and logging settings."""
        if verbose_logging is not None:
            self._verbose_logging = verbose_logging
        if debug_mode is not None:
            self._debug_mode = debug_mode
        return self

    def with_custom_config(self, custom_config: dict[str, Any]) -> ConfigBuilder:
        """Configure custom settings."""
        self._custom_config = custom_config
        return self

    def build(self) -> CortexFlowConfig:
        """Build the final configuration object."""
        # Invalidate the cached field map so new sections are included
        CortexFlowConfig._FIELD_MAP = None

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
            session=self._session,
            emotion=self._emotion,
            persona=self._persona,
            relationship=self._relationship,
            events=self._events,
            temporal=self._temporal,
            episodic=self._episodic,
            safety=self._safety,
            vector_store=self._vector_store,
            verbose_logging=self._verbose_logging,
            debug_mode=self._debug_mode,
            custom_config=self._custom_config,
        )
