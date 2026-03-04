"""
CortexFlow: Multi-tier memory optimization for LLMs with cognitive-inspired architecture.

This package provides tools to optimize context windows by implementing a
cognitive-inspired multi-tier memory system with dynamic allocation.
"""

from cortexflow.async_manager import AsyncCortexFlowManager
from cortexflow.config import (
    AgentConfig,
    ClassifierConfig,
    ConfigBuilder,
    CortexFlowConfig,
    EmotionConfig,
    EpisodicConfig,
    EventConfig,
    GraphRagConfig,
    InferenceConfig,
    KnowledgeStoreConfig,
    LLMConfig,
    MemoryConfig,
    MetadataConfig,
    OntologyConfig,
    PerformanceConfig,
    PersonaConfig,
    ReflectionConfig,
    RelationshipConfig,
    SafetyConfig,
    # Companion AI configs
    SessionConfig,
    # Temporal & Episodic configs
    TemporalConfig,
    UncertaintyConfig,
    VectorStoreConfig,
)
from cortexflow.emotion import (
    EmotionalState,
    EmotionDetector,
    EmotionTracker,
    LLMEmotionDetector,
    RuleBasedEmotionDetector,
)
from cortexflow.episodic_memory import Episode, EpisodicMemoryStore
from cortexflow.events import Event, EventBus, EventHandler, EventType
from cortexflow.interfaces import (
    AsyncContextProvider,
    AsyncKnowledgeStoreInterface,
    ContextProvider,
    KnowledgeStoreInterface,
    LLMProviderInterface,
    MemoryTierInterface,
    SessionAwareInterface,
)
from cortexflow.knowledge import KnowledgeStore
from cortexflow.llm_client import (
    LiteLLMClient,
    LLMClient,
    OllamaClient,
    VertexAIClient,
    create_llm_client,
)
from cortexflow.manager import CortexFlowManager, configure_logging
from cortexflow.memory import ConversationMemory
from cortexflow.persona import PersonaDefinition, PersonaManager
from cortexflow.relationship import (
    RelationshipStage,
    RelationshipState,
    RelationshipTracker,
)
from cortexflow.safety import SafetyLevel, SafetyPipeline, SafetyResult, SafetyRule

# Companion AI modules
from cortexflow.session import SessionContext, SessionManager
from cortexflow.temporal import TemporalFact, TemporalManager
from cortexflow.user_profile import UserProfile, UserProfileManager
from cortexflow.user_store import UserStore
from cortexflow.vector_stores import (
    VectorSearchResult,
    VectorStoreBackend,
    create_vector_store,
)
from cortexflow.version import __version__

__all__ = [
    # Core
    "CortexFlowManager",
    "AsyncCortexFlowManager",
    "configure_logging",
    # Config
    "CortexFlowConfig",
    "ConfigBuilder",
    "MemoryConfig",
    "KnowledgeStoreConfig",
    "GraphRagConfig",
    "LLMConfig",
    "AgentConfig",
    "ReflectionConfig",
    "UncertaintyConfig",
    "PerformanceConfig",
    "OntologyConfig",
    "MetadataConfig",
    "ClassifierConfig",
    "InferenceConfig",
    "SessionConfig",
    "EmotionConfig",
    "PersonaConfig",
    "RelationshipConfig",
    "EventConfig",
    "TemporalConfig",
    "EpisodicConfig",
    # Storage & Memory
    "ConversationMemory",
    "KnowledgeStore",
    # LLM
    "LLMClient",
    "OllamaClient",
    "VertexAIClient",
    "LiteLLMClient",
    "create_llm_client",
    # Interfaces
    "ContextProvider",
    "MemoryTierInterface",
    "KnowledgeStoreInterface",
    "LLMProviderInterface",
    "SessionAwareInterface",
    "AsyncContextProvider",
    "AsyncKnowledgeStoreInterface",
    # Session & User
    "SessionContext",
    "SessionManager",
    "UserStore",
    # Emotion
    "EmotionalState",
    "EmotionDetector",
    "RuleBasedEmotionDetector",
    "LLMEmotionDetector",
    "EmotionTracker",
    # User Profile
    "UserProfile",
    "UserProfileManager",
    # Persona
    "PersonaDefinition",
    "PersonaManager",
    # Relationship
    "RelationshipStage",
    "RelationshipState",
    "RelationshipTracker",
    # Events
    "EventType",
    "Event",
    "EventBus",
    "EventHandler",
    # Temporal & Episodic
    "TemporalFact",
    "TemporalManager",
    "Episode",
    "EpisodicMemoryStore",
    # Vector Stores
    "VectorStoreBackend",
    "VectorSearchResult",
    "VectorStoreConfig",
    "create_vector_store",
    # Safety
    "SafetyConfig",
    "SafetyLevel",
    "SafetyResult",
    "SafetyRule",
    "SafetyPipeline",
    # Version
    "__version__",
]
