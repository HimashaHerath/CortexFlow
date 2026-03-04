"""
Tests for the CortexFlow Agent Chain module.

Covers _is_complex_query heuristic, simple-query skip behavior,
agent error handling, and full chain processing with mocked LLM.
"""

from unittest.mock import MagicMock, patch

import pytest

from cortexflow.agent_chain import (
    AgentChainManager,
    AnalyzerAgent,
    SynthesizerAgent,
)
from cortexflow.config import ConfigBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_knowledge_store():
    """Create a mock knowledge store for agent chain tests."""
    ks = MagicMock()
    ks.get_relevant_knowledge.return_value = [
        {"text": "Python is a programming language.", "score": 0.9},
        {"text": "Machine learning uses algorithms.", "score": 0.8},
    ]
    return ks


@pytest.fixture
def config():
    return (
        ConfigBuilder()
        .with_agents(use_chain_of_agents=True, chain_complexity_threshold=5)
        .build()
    )


@pytest.fixture
def chain_manager(config, mock_knowledge_store):
    """Create an AgentChainManager with mocked LLM calls."""
    with patch("cortexflow.agent_chain.create_llm_client") as mock_llm_factory:
        mock_client = MagicMock()
        mock_client.generate_from_prompt.return_value = "Mocked LLM response."
        mock_client.batch_generate_from_prompts.return_value = ["Mocked batch response."]
        mock_llm_factory.return_value = mock_client
        manager = AgentChainManager(config, mock_knowledge_store)
    return manager


# ---------------------------------------------------------------------------
# _is_complex_query
# ---------------------------------------------------------------------------

class TestIsComplexQuery:
    """Test the complexity heuristic for query routing."""

    def test_short_simple_query_is_not_complex(self, chain_manager):
        assert chain_manager._is_complex_query("Hello") is False

    def test_short_question_is_not_complex(self, chain_manager):
        assert chain_manager._is_complex_query("What time is it?") is False

    def test_long_query_is_complex(self, chain_manager):
        long_query = "Please tell me about the history of computer science " \
                     "and all the major breakthroughs that happened in the field " \
                     "throughout the twentieth century and beyond"
        assert chain_manager._is_complex_query(long_query) is True

    def test_query_with_analysis_word_is_complex(self, chain_manager):
        assert chain_manager._is_complex_query("explain quantum mechanics") is True
        assert chain_manager._is_complex_query("analyze this data set") is True
        assert chain_manager._is_complex_query("compare Python and Java") is True
        assert chain_manager._is_complex_query("why does the sun rise") is True

    def test_query_with_conjunction_and_length_is_complex(self, chain_manager):
        query = "Tell me about Python and explain how it differs from Java but is similar to Ruby"
        assert chain_manager._is_complex_query(query) is True

    def test_conjunction_alone_not_enough_if_short(self, chain_manager):
        # "and" present but total words <= 8
        assert chain_manager._is_complex_query("cats and dogs") is False

    def test_how_keyword_makes_complex(self, chain_manager):
        assert chain_manager._is_complex_query("how does photosynthesis work") is True

    def test_evaluate_keyword_makes_complex(self, chain_manager):
        assert chain_manager._is_complex_query("evaluate this approach") is True


# ---------------------------------------------------------------------------
# Simple queries return skipped
# ---------------------------------------------------------------------------

class TestSimpleQuerySkip:
    """Test that simple queries skip the agent chain."""

    def test_simple_query_returns_skipped_true(self, chain_manager):
        result = chain_manager.process_query("Hello")
        assert result["skipped"] is True
        assert result["answer"] is None
        assert result["total_processing_time"] == 0.0

    def test_simple_query_has_empty_agent_chain(self, chain_manager):
        result = chain_manager.process_query("Hi there")
        assert result["agent_chain"] == []

    def test_simple_query_has_reason(self, chain_manager):
        result = chain_manager.process_query("Yes")
        assert "reason" in result
        assert "complexity" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Agent error handling
# ---------------------------------------------------------------------------

class TestAgentErrorHandling:
    """Test that agent failures produce proper error dicts."""

    def test_analyzer_without_history_returns_error(self, config, mock_knowledge_store):
        with patch("cortexflow.agent_chain.create_llm_client") as mock_llm:
            mock_llm.return_value = MagicMock()
            agent = AnalyzerAgent(config, mock_knowledge_store)

        result = agent.process("test query", {}, agent_history=None)
        assert result["status"] == "error"
        assert "No previous agent history" in str(result["analysis_results"])

    def test_synthesizer_without_enough_history_returns_error(self, config, mock_knowledge_store):
        with patch("cortexflow.agent_chain.create_llm_client") as mock_llm:
            mock_llm.return_value = MagicMock()
            agent = SynthesizerAgent(config, mock_knowledge_store)

        result = agent.process("test query", {}, agent_history=[{"one": "item"}])
        assert result["status"] == "error"
        assert "Insufficient information" in result["answer"]

    def test_fallback_result_on_explorer_failure(self, chain_manager):
        """When explorer raises, the chain manager generates a fallback."""
        error = RuntimeError("LLM down")
        fallback = chain_manager._generate_fallback_result(
            chain_manager.agents[0],  # Explorer
            "test query",
            {},
            [],
            error,
        )
        assert fallback["status"] == "error"
        assert "LLM down" in fallback["error"]

    def test_fallback_result_on_synthesizer_failure(self, chain_manager):
        error = RuntimeError("Synthesis failed")
        fallback = chain_manager._generate_fallback_result(
            chain_manager.agents[2],  # Synthesizer
            "test query",
            {},
            [{"exploration_results": {"exploration_text": "some data"}}],
            error,
        )
        assert fallback["status"] == "error"
        assert "answer" in fallback

    def test_can_continue_after_explorer_failure_without_items(self, chain_manager):
        result = {"knowledge_items": []}
        can_continue = chain_manager._can_continue_after_failure(
            chain_manager.agents[0], result
        )
        assert can_continue is False

    def test_can_continue_after_analyzer_fallback(self, chain_manager):
        result = {"analysis_results": {"status": "fallback"}}
        can_continue = chain_manager._can_continue_after_failure(
            chain_manager.agents[1], result
        )
        assert can_continue is True


# ---------------------------------------------------------------------------
# Full chain processing (mocked LLM)
# ---------------------------------------------------------------------------

class TestFullChainProcessing:
    """Test end-to-end processing with mocked LLM."""

    def test_complex_query_runs_all_agents(self, chain_manager):
        result = chain_manager.process_query(
            "explain in detail how machine learning algorithms work "
            "and compare supervised and unsupervised approaches"
        )
        assert "skipped" not in result or result.get("skipped") is not True
        assert result["total_processing_time"] > 0
        assert len(result["agent_chain"]) > 0

    def test_result_contains_answer(self, chain_manager):
        result = chain_manager.process_query(
            "analyze the differences between Python and Java programming languages"
        )
        assert "answer" in result

    def test_each_agent_has_processing_time(self, chain_manager):
        result = chain_manager.process_query(
            "explain why neural networks are effective at pattern recognition"
        )
        for agent_result in result["agent_chain"]:
            assert "processing_time" in agent_result
            assert agent_result["processing_time"] >= 0
