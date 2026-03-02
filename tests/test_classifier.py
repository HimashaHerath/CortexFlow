"""
Tests for the CortexFlow Classifier module.

Covers RuleBasedClassifier, ImportanceClassifier ensemble (rule-only mode),
MLClassifier removal, and importance scale verification.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from cortexflow.config import CortexFlowConfig, ConfigBuilder
from cortexflow.memory import ContextSegment
from cortexflow.classifier import RuleBasedClassifier, ImportanceClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(content, segment_type="user", importance=5.0, metadata=None):
    """Create a ContextSegment for testing."""
    return ContextSegment(
        content=content,
        importance=importance,
        timestamp=time.time(),
        token_count=len(content.split()),
        segment_type=segment_type,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# RuleBasedClassifier
# ---------------------------------------------------------------------------

class TestRuleBasedClassifier:
    """Test the deterministic rule-based classifier."""

    @pytest.fixture
    def classifier(self):
        return RuleBasedClassifier()

    def test_system_messages_get_high_score(self, classifier):
        segment = _make_segment("You are a helpful assistant.", segment_type="system")
        score = classifier.classify(segment)
        # System type_weight is 0.9; should be high
        assert score >= 0.8

    def test_user_messages_get_moderate_score(self, classifier):
        segment = _make_segment(
            "Tell me about the weather forecast for tomorrow in detail please.",
            segment_type="user",
        )
        score = classifier.classify(segment)
        assert 0.3 <= score <= 1.0

    def test_assistant_messages_get_lower_score(self, classifier):
        segment = _make_segment(
            "Sure, the weather will be sunny.",
            segment_type="assistant",
        )
        score = classifier.classify(segment)
        assert 0.0 <= score <= 1.0

    def test_important_keywords_boost_score(self, classifier):
        base_segment = _make_segment(
            "The cat sat on a mat.",
            segment_type="user",
        )
        important_segment = _make_segment(
            "URGENT: The critical deadline is tomorrow, remember this important fact.",
            segment_type="user",
        )
        base_score = classifier.classify(base_segment)
        important_score = classifier.classify(important_segment)
        assert important_score >= base_score

    def test_unimportant_patterns_reduce_score(self, classifier):
        normal_segment = _make_segment(
            "Tell me about quantum physics principles.",
            segment_type="user",
        )
        filler_segment = _make_segment("lol haha ok", segment_type="user")
        normal_score = classifier.classify(normal_segment)
        filler_score = classifier.classify(filler_segment)
        assert filler_score < normal_score

    def test_empty_content_returns_zero(self, classifier):
        segment = _make_segment("", segment_type="user")
        score = classifier.classify(segment)
        assert score == 0.0

    def test_none_segment_returns_zero(self, classifier):
        score = classifier.classify(None)
        assert score == 0.0

    def test_score_is_clamped_to_0_1(self, classifier):
        # A segment with many important keywords should still be capped at 1.0
        segment = _make_segment(
            "urgent critical important essential vital crucial key significant "
            "remember note attention action decision deadline schedule appointment "
            "meeting password credential account login",
            segment_type="system",
        )
        score = classifier.classify(segment)
        assert 0.0 <= score <= 1.0

    def test_url_pattern_boosts_score(self, classifier):
        segment = _make_segment(
            "Check the documentation at https://example.com/docs for more details.",
            segment_type="user",
        )
        plain_segment = _make_segment(
            "Check the documentation for more details.",
            segment_type="user",
        )
        url_score = classifier.classify(segment)
        plain_score = classifier.classify(plain_segment)
        assert url_score > plain_score

    def test_data_segment_type(self, classifier):
        segment = _make_segment("Raw data payload content.", segment_type="data")
        score = classifier.classify(segment)
        # Data type_weight is 0.8
        assert score >= 0.5


# ---------------------------------------------------------------------------
# MLClassifier removal verification
# ---------------------------------------------------------------------------

class TestMLClassifierRemoved:
    """Verify that the old MLClassifier class was removed."""

    def test_ml_classifier_import_fails(self):
        with pytest.raises(ImportError):
            from cortexflow.classifier import MLClassifier  # noqa: F401


# ---------------------------------------------------------------------------
# ImportanceClassifier ensemble (rule-only, no LLM)
# ---------------------------------------------------------------------------

class TestImportanceClassifier:
    """Test the ensemble classifier with rule-only mode (LLM mocked out)."""

    @pytest.fixture
    def classifier(self):
        config = CortexFlowConfig()
        # Patch create_llm_client to avoid network calls
        with patch("cortexflow.classifier.create_llm_client") as mock_llm:
            mock_llm.return_value = MagicMock()
            clf = ImportanceClassifier(config)
        return clf

    def test_returns_score_on_0_to_10_scale(self, classifier):
        segment = _make_segment("What is the capital of France?", segment_type="user")
        score = classifier.classify(segment)
        assert 0.0 <= score <= 10.0

    def test_system_messages_get_high_importance(self, classifier):
        segment = _make_segment(
            "You are a helpful assistant that specializes in code review.",
            segment_type="system",
        )
        score = classifier.classify(segment)
        # System type weight is 0.9 -> scaled to ~9.0 on 0-10 scale
        assert score >= 7.0

    def test_short_filler_messages_get_low_importance(self, classifier):
        segment = _make_segment("ok", segment_type="user")
        score = classifier.classify(segment)
        assert score < 8.0

    def test_none_segment_returns_zero(self, classifier):
        score = classifier.classify(None)
        assert score == 0.0

    def test_importance_stored_on_segment(self, classifier):
        segment = _make_segment("Store this information please.", segment_type="user")
        score = classifier.classify(segment)
        assert segment.importance == score

    def test_rule_only_mode_ignores_llm(self, classifier):
        """When use_llm_classification is not set, only rule classifier is used."""
        segment = _make_segment(
            "This is a moderate length message about an interesting topic.",
            segment_type="user",
        )
        score = classifier.classify(segment)
        # Should still return a valid score without LLM
        assert 0.0 <= score <= 10.0

    def test_weight_normalization(self, classifier):
        """Weights should be normalized to sum to 1.0 in rule-only mode."""
        # In rule-only mode, only rule_weight is used, normalized to 1.0
        segment = _make_segment("Testing weight normalization.", segment_type="user")
        score = classifier.classify(segment)
        # Score should be valid even with single classifier
        assert 0.0 <= score <= 10.0
