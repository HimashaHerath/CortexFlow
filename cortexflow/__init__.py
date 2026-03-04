"""
CortexFlow: Multi-tier memory optimization for LLMs with cognitive-inspired architecture.

This package provides tools to optimize context windows by implementing a
cognitive-inspired multi-tier memory system with dynamic allocation.
"""

from cortexflow.manager import CortexFlowManager, configure_logging
from cortexflow.config import (
    CortexFlowConfig,
    ConfigBuilder,
    MemoryConfig,
    KnowledgeStoreConfig,
    GraphRagConfig,
    LLMConfig,
    AgentConfig,
    ReflectionConfig,
    UncertaintyConfig,
    PerformanceConfig,
    OntologyConfig,
    MetadataConfig,
    ClassifierConfig,
    InferenceConfig,
)
from cortexflow.memory import ConversationMemory
from cortexflow.knowledge import KnowledgeStore
from cortexflow.llm_client import (
    LLMClient,
    OllamaClient,
    VertexAIClient,
    LiteLLMClient,
    create_llm_client,
)
from cortexflow.interfaces import (
    ContextProvider,
    MemoryTierInterface,
    KnowledgeStoreInterface,
    LLMProviderInterface,
)
from cortexflow.version import __version__

__all__ = [
    # Core
    "CortexFlowManager",
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
    # Version
    "__version__",
]