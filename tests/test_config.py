"""
Tests for the CortexFlow configuration system.

Covers CortexFlowConfig defaults, ConfigBuilder fluent API, backward-compat
__getattr__ proxy, from_dict/to_dict roundtrip, and dead flag removal.
"""

import os
from dataclasses import fields

import pytest

from cortexflow.config import (
    ConfigBuilder,
    CortexFlowConfig,
    PerformanceConfig,
)

# ---------------------------------------------------------------------------
# CortexFlowConfig defaults
# ---------------------------------------------------------------------------


class TestCortexFlowConfigDefaults:
    """Verify that CortexFlowConfig creates sensible defaults."""

    def test_default_memory_limits(self):
        config = CortexFlowConfig()
        assert config.memory.active_token_limit == 4096
        assert config.memory.working_token_limit == 8192
        assert config.memory.archive_token_limit == 16384

    def test_default_llm_backend_is_ollama(self):
        config = CortexFlowConfig()
        assert config.llm.backend == "ollama"
        assert config.llm.default_model == "gemma3:1b"

    def test_default_debug_flags_are_off(self):
        config = CortexFlowConfig()
        assert config.verbose_logging is False
        assert config.debug_mode is False

    def test_default_graph_rag_is_disabled(self):
        config = CortexFlowConfig()
        assert config.graph_rag.use_graph_rag is False

    def test_default_custom_config_is_empty(self):
        config = CortexFlowConfig()
        assert config.custom_config == {}

    def test_knowledge_store_path_becomes_absolute(self):
        config = CortexFlowConfig()
        assert os.path.isabs(config.knowledge_store.knowledge_store_path)

    def test_all_sections_are_dataclasses(self):
        config = CortexFlowConfig()
        section_names = [
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
        ]
        for name in section_names:
            section = getattr(config, name)
            assert hasattr(section, "__dataclass_fields__"), (
                f"Section '{name}' is not a dataclass"
            )


# ---------------------------------------------------------------------------
# __getattr__ backward-compat proxy
# ---------------------------------------------------------------------------


class TestGetAttrProxy:
    """Test that flat attribute access is proxied to nested sections."""

    def test_ollama_host_resolves_to_llm_section(self):
        config = CortexFlowConfig()
        assert config.ollama_host == config.llm.ollama_host

    def test_active_token_limit_resolves_to_memory_section(self):
        config = CortexFlowConfig()
        assert config.active_token_limit == config.memory.active_token_limit

    def test_use_graph_rag_resolves_to_graph_rag_section(self):
        config = CortexFlowConfig()
        assert config.use_graph_rag == config.graph_rag.use_graph_rag

    def test_use_chain_of_agents_resolves_to_agents_section(self):
        config = CortexFlowConfig()
        assert config.use_chain_of_agents == config.agents.use_chain_of_agents

    def test_use_self_reflection_resolves_to_reflection_section(self):
        config = CortexFlowConfig()
        assert config.use_self_reflection == config.reflection.use_self_reflection

    def test_nonexistent_attribute_raises_attribute_error(self):
        config = CortexFlowConfig()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = config.completely_fake_attribute

    def test_proxy_reflects_mutations(self):
        config = CortexFlowConfig()
        config.llm.ollama_host = "http://custom:9999"
        assert config.ollama_host == "http://custom:9999"


# ---------------------------------------------------------------------------
# ConfigBuilder
# ---------------------------------------------------------------------------


class TestConfigBuilder:
    """Test the fluent ConfigBuilder API."""

    def test_build_returns_cortexflow_config(self):
        config = ConfigBuilder().build()
        assert isinstance(config, CortexFlowConfig)

    def test_with_memory_sets_values(self):
        config = (
            ConfigBuilder()
            .with_memory(active_token_limit=1024, working_token_limit=2048)
            .build()
        )
        assert config.memory.active_token_limit == 1024
        assert config.memory.working_token_limit == 2048

    def test_with_llm_sets_backend(self):
        config = (
            ConfigBuilder()
            .with_llm(backend="ollama", default_model="llama3:8b")
            .build()
        )
        assert config.llm.backend == "ollama"
        assert config.llm.default_model == "llama3:8b"

    def test_with_graph_rag_enables_feature(self):
        config = (
            ConfigBuilder().with_graph_rag(use_graph_rag=True, max_graph_hops=5).build()
        )
        assert config.graph_rag.use_graph_rag is True
        assert config.graph_rag.max_graph_hops == 5

    def test_with_agents_sets_values(self):
        config = (
            ConfigBuilder()
            .with_agents(use_chain_of_agents=True, chain_agent_count=5)
            .build()
        )
        assert config.agents.use_chain_of_agents is True
        assert config.agents.chain_agent_count == 5

    def test_with_reflection_sets_threshold(self):
        config = (
            ConfigBuilder()
            .with_reflection(
                use_self_reflection=True, reflection_relevance_threshold=0.8
            )
            .build()
        )
        assert config.reflection.use_self_reflection is True
        assert config.reflection.reflection_relevance_threshold == 0.8

    def test_with_performance_sets_cache_size(self):
        config = ConfigBuilder().with_performance(reasoning_cache_max_size=500).build()
        assert config.performance.reasoning_cache_max_size == 500

    def test_with_debug_sets_flags(self):
        config = (
            ConfigBuilder().with_debug(verbose_logging=True, debug_mode=True).build()
        )
        assert config.verbose_logging is True
        assert config.debug_mode is True

    def test_with_custom_config(self):
        config = ConfigBuilder().with_custom_config({"my_key": 42}).build()
        assert config.custom_config == {"my_key": 42}

    def test_builder_chaining_returns_self(self):
        builder = ConfigBuilder()
        result = builder.with_memory(active_token_limit=100)
        assert result is builder

    def test_with_vertex_ai_sets_backend(self):
        config = (
            ConfigBuilder()
            .with_vertex_ai(
                project_id="my-project",
                location="us-east1",
                default_model="gemini-1.5-flash",
                api_key="test-key",
                credentials_path="/tmp/creds.json",
            )
            .build()
        )
        assert config.llm.backend == "vertex_ai"
        assert config.llm.vertex_project_id == "my-project"
        assert config.llm.vertex_location == "us-east1"
        assert config.llm.vertex_model == "gemini-1.5-flash"
        assert config.llm.vertex_api_key == "test-key"
        assert config.llm.vertex_credentials_path == "/tmp/creds.json"

    def test_with_vertex_ai_without_optional_params(self):
        config = ConfigBuilder().with_vertex_ai().build()
        assert config.llm.backend == "vertex_ai"
        # Project ID should remain None when not passed
        assert config.llm.vertex_project_id is None


# ---------------------------------------------------------------------------
# from_dict / to_dict roundtrip
# ---------------------------------------------------------------------------


class TestDictRoundtrip:
    """Test from_dict() and to_dict() roundtrip."""

    def test_to_dict_contains_flat_keys(self):
        config = CortexFlowConfig()
        d = config.to_dict()
        # Should contain flattened keys from nested sections
        assert "active_token_limit" in d
        assert "ollama_host" in d
        assert "use_graph_rag" in d
        assert "verbose_logging" in d

    def test_from_dict_creates_valid_config(self):
        d = {
            "active_token_limit": 2048,
            "default_model": "llama3:8b",
            "use_graph_rag": True,
            "verbose_logging": True,
        }
        config = CortexFlowConfig.from_dict(d)
        assert config.memory.active_token_limit == 2048
        assert config.llm.default_model == "llama3:8b"
        assert config.graph_rag.use_graph_rag is True
        assert config.verbose_logging is True

    def test_roundtrip_preserves_values(self):
        original = (
            ConfigBuilder()
            .with_memory(active_token_limit=512, working_token_limit=1024)
            .with_llm(default_model="test-model")
            .with_debug(verbose_logging=True)
            .build()
        )
        d = original.to_dict()
        restored = CortexFlowConfig.from_dict(d)
        assert restored.memory.active_token_limit == 512
        assert restored.memory.working_token_limit == 1024
        assert restored.llm.default_model == "test-model"
        assert restored.verbose_logging is True

    def test_from_dict_with_empty_dict_returns_defaults(self):
        config = CortexFlowConfig.from_dict({})
        assert config.memory.active_token_limit == 4096
        assert config.llm.backend == "ollama"


# ---------------------------------------------------------------------------
# Dead flags removal verification
# ---------------------------------------------------------------------------


class TestDeadFlagsRemoved:
    """Verify that deprecated/dead config flags no longer exist."""

    def test_use_query_planning_not_on_performance_config(self):
        perf_field_names = {f.name for f in fields(PerformanceConfig)}
        assert "use_query_planning" not in perf_field_names

    def test_use_reasoning_cache_not_on_performance_config(self):
        perf_field_names = {f.name for f in fields(PerformanceConfig)}
        assert "use_reasoning_cache" not in perf_field_names

    def test_accessing_dead_flag_via_proxy_raises(self):
        config = CortexFlowConfig()
        with pytest.raises(AttributeError):
            _ = config.use_query_planning
