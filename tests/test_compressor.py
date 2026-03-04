"""
Tests for the CortexFlow Compressor module.

Covers TruncationCompressor, ExtractiveSummarizer, ContextCompressor, and
compress_segment with different importance levels.
"""

import time

import pytest

from cortexflow.compressor import (
    ContextCompressor,
    ExtractiveSummarizer,
    TruncationCompressor,
)
from cortexflow.memory import ContextSegment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LONG_TEXT = (
    "The development of artificial intelligence has been one of the most significant "
    "technological advances of the 21st century. Machine learning algorithms have "
    "transformed industries ranging from healthcare to finance. Deep learning models "
    "can now recognize images, translate languages, and generate human-like text. "
    "Researchers continue to push the boundaries of what is possible with neural "
    "networks. The ethical implications of these technologies are being actively "
    "debated by policymakers and academics around the world. Companies are investing "
    "billions of dollars in AI research and development. The future of artificial "
    "intelligence remains both exciting and uncertain."
)


def _make_segment(content, importance=5.0, segment_type="user"):
    """Create a ContextSegment for testing."""
    return ContextSegment(
        content=content,
        importance=importance,
        timestamp=time.time(),
        token_count=len(content.split()),
        segment_type=segment_type,
        metadata={},
    )


# ---------------------------------------------------------------------------
# TruncationCompressor
# ---------------------------------------------------------------------------


class TestTruncationCompressor:
    """Test the simple truncation compressor."""

    @pytest.fixture
    def compressor(self):
        return TruncationCompressor()

    def test_compresses_text_to_target_ratio(self, compressor):
        result = compressor.compress(LONG_TEXT, 0.5)
        # Result should be shorter than original
        assert len(result) < len(LONG_TEXT)

    def test_adds_ellipsis(self, compressor):
        result = compressor.compress(LONG_TEXT, 0.3)
        assert result.endswith("...")

    def test_ratio_1_returns_original(self, compressor):
        result = compressor.compress(LONG_TEXT, 1.0)
        assert result == LONG_TEXT

    def test_ratio_above_1_returns_original(self, compressor):
        result = compressor.compress(LONG_TEXT, 1.5)
        assert result == LONG_TEXT

    def test_empty_string_returns_empty(self, compressor):
        result = compressor.compress("", 0.5)
        assert result == ""

    def test_output_respects_target_length(self, compressor):
        result = compressor.compress(LONG_TEXT, 0.4)
        # Truncated portion (minus ellipsis) should be approximately target
        target_len = int(len(LONG_TEXT) * 0.4)
        # The result includes "..." so it's target_len + 3
        assert len(result) == target_len + 3


# ---------------------------------------------------------------------------
# ExtractiveSummarizer
# ---------------------------------------------------------------------------


class TestExtractiveSummarizer:
    """Test keyword-based extractive summarization."""

    @pytest.fixture
    def summarizer(self):
        return ExtractiveSummarizer()

    def test_compresses_text(self, summarizer):
        result = summarizer.compress(LONG_TEXT, 0.5)
        assert len(result) < len(LONG_TEXT)

    def test_ratio_1_returns_original(self, summarizer):
        result = summarizer.compress(LONG_TEXT, 1.0)
        assert result == LONG_TEXT

    def test_empty_returns_original(self, summarizer):
        result = summarizer.compress("", 0.5)
        assert result == ""

    def test_extract_keywords_returns_list(self, summarizer):
        keywords = summarizer.extract_keywords(LONG_TEXT)
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_extract_keywords_excludes_stop_words(self, summarizer):
        keywords = summarizer.extract_keywords(LONG_TEXT)
        for kw in keywords:
            assert kw not in summarizer.stop_words

    def test_rank_sentences_returns_scored_tuples(self, summarizer):
        keywords = summarizer.extract_keywords(LONG_TEXT)
        scored = summarizer.rank_sentences(LONG_TEXT, keywords)
        assert isinstance(scored, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in scored)

    def test_very_short_text_falls_back_to_truncation(self, summarizer):
        short_text = "Hello world."
        result = summarizer.compress(short_text, 0.3)
        # Should not crash; may fall back to truncation
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------


class TestContextCompressor:
    """Test the main ContextCompressor that orchestrates strategies."""

    def test_create_default_works_without_config(self):
        compressor = ContextCompressor.create_default()
        assert compressor is not None
        assert compressor.config is None
        assert compressor._abstractive is None

    def test_create_default_has_extractive_and_truncation(self):
        compressor = ContextCompressor.create_default()
        assert compressor.truncation is not None
        assert compressor.extractive is not None

    def test_compress_segment_returns_new_segment(self):
        compressor = ContextCompressor.create_default()
        segment = _make_segment(LONG_TEXT, importance=5.0)
        result = compressor.compress_segment(segment, 0.5)
        assert isinstance(result, ContextSegment)
        assert result is not segment  # New instance

    def test_compressed_content_is_smaller(self):
        compressor = ContextCompressor.create_default()
        segment = _make_segment(LONG_TEXT, importance=3.0)
        result = compressor.compress_segment(segment, 0.5)
        assert len(result.content) < len(segment.content)

    def test_high_importance_gets_minimal_compression(self):
        compressor = ContextCompressor.create_default()
        segment = _make_segment(LONG_TEXT, importance=9.0)
        result = compressor.compress_segment(segment, 0.3)
        # High importance (>=8.0) should use at least 0.9 ratio
        # So the output should be almost as long as original
        assert len(result.content) >= len(segment.content) * 0.8

    def test_low_importance_uses_aggressive_compression(self):
        compressor = ContextCompressor.create_default()
        low_segment = _make_segment(LONG_TEXT, importance=2.0)
        high_segment = _make_segment(LONG_TEXT, importance=9.0)
        low_result = compressor.compress_segment(low_segment, 0.4)
        high_result = compressor.compress_segment(high_segment, 0.4)
        assert len(low_result.content) < len(high_result.content)

    def test_code_segment_uses_truncation(self):
        code_text = "def hello():\n    print('hello world')\n    return True\n" * 10
        compressor = ContextCompressor.create_default()
        segment = _make_segment(code_text, importance=5.0, segment_type="code")
        result = compressor.compress_segment(segment, 0.5)
        # Code should be truncated (ends with ...)
        assert result.content.endswith("...")

    def test_compressed_segment_has_metadata(self):
        compressor = ContextCompressor.create_default()
        segment = _make_segment(LONG_TEXT, importance=5.0)
        result = compressor.compress_segment(segment, 0.5)
        assert result.metadata.get("compressed") is True
        assert "original_length" in result.metadata

    def test_compressed_segment_preserves_importance(self):
        compressor = ContextCompressor.create_default()
        segment = _make_segment(LONG_TEXT, importance=7.5)
        result = compressor.compress_segment(segment, 0.5)
        assert result.importance == 7.5

    def test_compressed_segment_preserves_type(self):
        compressor = ContextCompressor.create_default()
        segment = _make_segment(LONG_TEXT, importance=5.0, segment_type="assistant")
        result = compressor.compress_segment(segment, 0.5)
        assert result.segment_type == "assistant"

    def test_progressive_compress_under_budget_returns_unchanged(self):
        compressor = ContextCompressor.create_default()
        segments = [
            _make_segment("Short text.", importance=5.0),
            _make_segment("Another short text.", importance=5.0),
        ]
        # Set a large budget that's already met
        result = compressor.progressive_compress(segments, target_token_count=1000)
        assert len(result) == len(segments)
        # Content should be unchanged
        for orig, res in zip(segments, result):
            assert orig.content == res.content
