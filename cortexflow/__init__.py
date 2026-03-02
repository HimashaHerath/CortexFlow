"""
CortexFlow: Multi-tier memory optimization for LLMs with cognitive-inspired architecture.

This package provides tools to optimize context windows by implementing a
cognitive-inspired multi-tier memory system with dynamic allocation.
"""

from cortexflow.manager import CortexFlowManager
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
from cortexflow.version import __version__

__all__ = [
    "CortexFlowManager",
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
    "ConversationMemory",
    "__version__",
] 