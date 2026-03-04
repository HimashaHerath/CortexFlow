"""
Tests for the CortexFlow Reflection module.

Covers _extract_claims, _compute_kb_support_ratio, verify_knowledge_relevance
filtering, and check_response_consistency with mocked LLM.
"""

from unittest.mock import MagicMock, patch

import pytest

from cortexflow.config import ConfigBuilder
from cortexflow.reflection import ReflectionEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a ReflectionEngine with mocked LLM client."""
    config = (
        ConfigBuilder()
        .with_reflection(
            use_self_reflection=True,
            reflection_relevance_threshold=0.6,
            reflection_confidence_threshold=0.7,
        )
        .build()
    )
    with patch("cortexflow.reflection.create_llm_client") as mock_llm:
        mock_client = MagicMock()
        mock_client.generate_from_prompt.return_value = "Mocked LLM response"
        mock_llm.return_value = mock_client
        eng = ReflectionEngine(config)
    return eng


# ---------------------------------------------------------------------------
# _extract_claims
# ---------------------------------------------------------------------------


class TestExtractClaims:
    """Test sentence-based claim extraction."""

    def test_splits_into_sentences(self, engine):
        text = "The sky is blue. Water flows downhill. Plants need sunlight to grow."
        claims = engine._extract_claims(text)
        assert len(claims) >= 2

    def test_filters_short_sentences(self, engine):
        text = "Yes. No. Maybe. The temperature is 72 degrees Fahrenheit today."
        claims = engine._extract_claims(text)
        # Only the long sentence should survive
        assert len(claims) >= 1
        assert any("temperature" in c for c in claims)

    def test_filters_non_substantive_fragments(self, engine):
        text = (
            "I think this is interesting. Let me explain. Here is the data. "
            "The algorithm processes data in three distinct phases."
        )
        claims = engine._extract_claims(text)
        # "I think...", "Let me...", "Here is..." should be filtered
        substantive = [c for c in claims if "algorithm" in c]
        assert len(substantive) >= 1

    def test_empty_text_returns_empty(self, engine):
        claims = engine._extract_claims("")
        assert claims == []

    def test_single_long_sentence(self, engine):
        text = "The quick brown fox jumps over the lazy dog near the riverbank."
        claims = engine._extract_claims(text)
        assert len(claims) >= 1


# ---------------------------------------------------------------------------
# _compute_kb_support_ratio
# ---------------------------------------------------------------------------


class TestComputeKBSupportRatio:
    """Test knowledge base support ratio computation."""

    def test_all_claims_supported(self, engine):
        claims = ["Python is a popular programming language used worldwide."]
        kb_items = [
            {
                "text": "Python is a popular programming language used by millions worldwide."
            }
        ]
        ratio, details = engine._compute_kb_support_ratio(claims, kb_items)
        assert ratio == 1.0
        assert details[0]["has_kb_support"] is True

    def test_no_claims_supported(self, engine):
        claims = ["Jupiter orbits around the distant galaxy core."]
        kb_items = [{"text": "Python is a programming language."}]
        ratio, details = engine._compute_kb_support_ratio(claims, kb_items)
        assert ratio == 0.0
        assert details[0]["has_kb_support"] is False

    def test_empty_claims_returns_full_support(self, engine):
        ratio, details = engine._compute_kb_support_ratio([], [])
        assert ratio == 1.0
        assert details == []

    def test_partial_support(self, engine):
        claims = [
            "Python is a popular programming language used worldwide.",
            "Jupiter orbits around the distant galaxy core.",
        ]
        kb_items = [
            {
                "text": "Python is a popular programming language used by millions worldwide."
            }
        ]
        ratio, details = engine._compute_kb_support_ratio(claims, kb_items)
        assert 0.0 < ratio < 1.0

    def test_stopwords_excluded_from_matching(self, engine):
        # Claims with mostly stopwords should not match
        claims = ["The is are was were to of in for on with at by from."]
        kb_items = [{"text": "The is are was were to of in for on with at by from."}]
        ratio, details = engine._compute_kb_support_ratio(claims, kb_items)
        # After removing stopwords, no meaningful overlap
        assert details[0]["has_kb_support"] is False

    def test_details_contain_claim_text(self, engine):
        claims = ["Machine learning enables computers to learn patterns."]
        kb_items = [
            {"text": "Machine learning helps computers learn patterns from data."}
        ]
        ratio, details = engine._compute_kb_support_ratio(claims, kb_items)
        assert "claim" in details[0]
        assert "has_kb_support" in details[0]


# ---------------------------------------------------------------------------
# verify_knowledge_relevance
# ---------------------------------------------------------------------------


class TestVerifyKnowledgeRelevance:
    """Test knowledge relevance filtering."""

    def test_empty_items_returns_empty(self, engine):
        result = engine.verify_knowledge_relevance("test query", [])
        assert result == []

    def test_filters_items_below_threshold(self, engine):
        """Mock LLM to return low scores for some items."""
        engine.llm_client.generate_from_prompt.return_value = """
        [
            {"item": 1, "score": 0.9, "explanation": "Highly relevant"},
            {"item": 2, "score": 0.2, "explanation": "Not relevant"}
        ]
        """
        items = [
            {"text": "Relevant knowledge about Python."},
            {"text": "Unrelated knowledge about cooking."},
        ]
        result = engine.verify_knowledge_relevance("Python programming", items)
        # Item 2 has score 0.2 < threshold 0.6, should be filtered
        assert len(result) == 1
        assert result[0]["relevance_score"] == 0.9

    def test_all_items_pass_threshold(self, engine):
        engine.llm_client.generate_from_prompt.return_value = """
        [
            {"item": 1, "score": 0.8, "explanation": "Relevant"},
            {"item": 2, "score": 0.7, "explanation": "Relevant"}
        ]
        """
        items = [
            {"text": "Item one."},
            {"text": "Item two."},
        ]
        result = engine.verify_knowledge_relevance("query", items)
        assert len(result) == 2

    def test_parse_error_returns_original_items(self, engine):
        """If LLM returns garbage, the parse error handler assigns default 0.5 scores.
        With threshold=0.6, items scored 0.5 will be filtered out.
        Verify the parse fallback itself by checking the _parse_relevance_response output."""
        engine.llm_client.generate_from_prompt.return_value = "not valid json at all"
        items = [{"text": "Some knowledge."}]
        # _parse_relevance_response should assign 0.5 defaults on error
        response = engine.llm_client.generate_from_prompt("test")
        parsed = engine._parse_relevance_response(response, items)
        assert len(parsed) == 1
        assert parsed[0]["relevance_score"] == 0.5


# ---------------------------------------------------------------------------
# check_response_consistency (mocked LLM)
# ---------------------------------------------------------------------------


class TestCheckResponseConsistency:
    """Test consistency checking with mocked LLM."""

    def test_consistent_response(self, engine):
        engine.llm_client.generate_from_prompt.return_value = """
        {
            "is_consistent": true,
            "confidence": 0.95,
            "issues": [],
            "reasoning": "All claims supported."
        }
        """
        result = engine.check_response_consistency(
            query="What is Python?",
            response="Python is a popular programming language used for web development.",
            knowledge_items=[
                {
                    "text": "Python is a popular programming language used for web development and data science."
                }
            ],
        )
        assert result["is_consistent"] is True
        assert "kb_support_ratio" in result

    def test_inconsistent_response_has_issues(self, engine):
        engine.llm_client.generate_from_prompt.return_value = """
        {
            "is_consistent": false,
            "confidence": 0.8,
            "issues": ["Claim about Python being created in 2020 is incorrect."],
            "reasoning": "Historical date is wrong."
        }
        """
        result = engine.check_response_consistency(
            query="When was Python created?",
            response="Python was created in 2020 by Guido van Rossum.",
            knowledge_items=[
                {"text": "Python was created in 1991 by Guido van Rossum."}
            ],
        )
        assert result["is_consistent"] is False
        assert len(result["issues"]) > 0

    def test_kb_support_ratio_included(self, engine):
        engine.llm_client.generate_from_prompt.return_value = """
        {"is_consistent": true, "confidence": 0.9, "issues": [], "reasoning": "OK"}
        """
        result = engine.check_response_consistency(
            query="test",
            response="Machine learning algorithms process data efficiently.",
            knowledge_items=[
                {
                    "text": "Machine learning algorithms can process large amounts of data efficiently."
                }
            ],
        )
        assert "kb_support_ratio" in result
        assert 0.0 <= result["kb_support_ratio"] <= 1.0

    def test_claim_details_included(self, engine):
        engine.llm_client.generate_from_prompt.return_value = """
        {"is_consistent": true, "confidence": 0.9, "issues": [], "reasoning": "OK"}
        """
        result = engine.check_response_consistency(
            query="test",
            response="Neural networks learn from data. They use backpropagation for training.",
            knowledge_items=[
                {"text": "Neural networks learn from data using backpropagation."}
            ],
        )
        assert "claim_details" in result

    def test_llm_parse_error_returns_default(self, engine):
        engine.llm_client.generate_from_prompt.return_value = "unparseable garbage"
        result = engine.check_response_consistency(
            query="test",
            response="Some response about things.",
            knowledge_items=[{"text": "Some knowledge."}],
        )
        # Default on error is is_consistent=True
        assert result["is_consistent"] is True
        assert result["confidence"] == 0.5


# ---------------------------------------------------------------------------
# revise_response
# ---------------------------------------------------------------------------


class TestReviseResponse:
    """Test response revision."""

    def test_consistent_response_not_revised(self, engine):
        original = "Python is great."
        result = engine.revise_response(
            query="test",
            original_response=original,
            knowledge_items=[],
            consistency_result={"is_consistent": True},
        )
        assert result == original

    def test_inconsistent_response_is_revised(self, engine):
        engine.llm_client.generate_from_prompt.return_value = (
            "Revised: Python was created in 1991."
        )
        result = engine.revise_response(
            query="When was Python created?",
            original_response="Python was created in 2020.",
            knowledge_items=[{"text": "Python was created in 1991."}],
            consistency_result={
                "is_consistent": False,
                "issues": ["Wrong creation date."],
            },
        )
        assert "1991" in result
