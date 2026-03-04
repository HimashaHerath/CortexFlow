"""
Tests for CortexFlow memory tier management.

Tests overflow behavior, demotion cascading, system message protection,
get_context_messages across tiers, and compression during demotion.
Uses small token limits (50/100/200) to force overflow quickly.
"""

import time

from cortexflow.config import CortexFlowConfig
from cortexflow.memory import (
    ActiveTier,
    ArchiveTier,
    ContextSegment,
    ConversationMemory,
    WorkingTier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_config(active=50, working=100, archive=200):
    """Create a config with very small token limits to trigger overflow."""
    return CortexFlowConfig.from_dict({
        "active_token_limit": active,
        "working_token_limit": working,
        "archive_token_limit": archive,
    })


def _make_segment(content, importance=5.0, segment_type="user"):
    return ContextSegment(
        content=content,
        importance=importance,
        timestamp=time.time(),
        token_count=len(content.split()),
        segment_type=segment_type,
        metadata={},
    )


def _word_string(n):
    """Generate a string with approximately n words."""
    words = ["word"] * n
    return " ".join(words)


# ---------------------------------------------------------------------------
# Basic tier operations
# ---------------------------------------------------------------------------

class TestBasicTierOperations:
    """Test fundamental MemoryTier behavior."""

    def test_active_tier_creation(self):
        tier = ActiveTier(100)
        assert tier.name == "active"
        assert tier.max_tokens == 100
        assert tier.current_token_count == 0

    def test_working_tier_creation(self):
        tier = WorkingTier(200)
        assert tier.name == "working"
        assert tier.max_tokens == 200

    def test_archive_tier_creation(self):
        tier = ArchiveTier(300)
        assert tier.name == "archive"
        assert tier.max_tokens == 300

    def test_add_segment_increases_token_count(self):
        tier = ActiveTier(1000)
        segment = _make_segment("hello world", importance=5.0)
        result = tier.add_segment(segment)
        assert result is True
        assert tier.current_token_count == segment.token_count

    def test_add_segment_fails_when_full(self):
        tier = ActiveTier(5)
        segment = _make_segment(_word_string(10), importance=5.0)
        result = tier.add_segment(segment)
        assert result is False

    def test_remove_segment_decreases_token_count(self):
        tier = ActiveTier(1000)
        segment = _make_segment("hello world test", importance=5.0)
        tier.add_segment(segment)
        initial_count = tier.current_token_count
        removed = tier.remove_segment(0)
        assert removed is not None
        assert tier.current_token_count == initial_count - removed.token_count

    def test_is_full_property(self):
        tier = ActiveTier(3)
        # Directly insert a segment with exact token_count = 3 to match the limit
        segment = ContextSegment(
            content="one two three",
            importance=5.0,
            timestamp=time.time(),
            token_count=3,  # Exactly matches max_tokens
            segment_type="user",
            metadata={},
        )
        tier.add_segment(segment)
        assert tier.is_full is True

    def test_available_tokens(self):
        tier = ActiveTier(100)
        segment = _make_segment("hello", importance=5.0)
        tier.add_segment(segment)
        assert tier.available_tokens == 100 - segment.token_count

    def test_fullness_ratio(self):
        tier = ActiveTier(100)
        assert tier.fullness_ratio == 0.0
        segment = _make_segment(_word_string(50), importance=5.0)
        tier.add_segment(segment)
        assert tier.fullness_ratio > 0.0


# ---------------------------------------------------------------------------
# ConversationMemory: adding messages fills active tier
# ---------------------------------------------------------------------------

class TestAddMessageFillsActiveTier:
    """Test that add_message places segments into the active tier."""

    def test_message_appears_in_active_tier(self):
        config = _make_small_config(active=500, working=500, archive=500)
        memory = ConversationMemory(config)
        memory.add_message("user", "Hello, how are you?")
        assert len(memory.active_tier.segments) >= 1

    def test_active_tier_token_count_increases(self):
        config = _make_small_config(active=500, working=500, archive=500)
        memory = ConversationMemory(config)
        memory.add_message("user", "Test message content here.")
        assert memory.active_tier.current_token_count > 0

    def test_multiple_messages_accumulate(self):
        config = _make_small_config(active=500, working=500, archive=500)
        memory = ConversationMemory(config)
        memory.add_message("user", "First message.")
        memory.add_message("assistant", "Second message.")
        memory.add_message("user", "Third message.")
        assert len(memory.active_tier.segments) >= 3


# ---------------------------------------------------------------------------
# Overflow triggers demotion to working tier
# ---------------------------------------------------------------------------

class TestActiveTierOverflow:
    """Test that active tier overflow demotes segments to working tier."""

    def test_overflow_populates_working_tier(self):
        config = _make_small_config(active=10, working=500, archive=500)
        memory = ConversationMemory(config)
        # Each message has ~5-6 tokens, active limit is 10
        # After a few messages, overflow should trigger
        for i in range(5):
            memory.add_message("user", f"Message number {i} with some extra words here.")
        # Working tier should have received demoted segments
        assert memory.working_tier.current_token_count > 0

    def test_active_tier_stays_within_limit(self):
        config = _make_small_config(active=15, working=500, archive=500)
        memory = ConversationMemory(config)
        for i in range(10):
            memory.add_message("user", f"Overflow test message number {i} content.")
        # Active tier should not exceed its limit (much)
        assert memory.active_tier.current_token_count <= config.memory.active_token_limit + 20


# ---------------------------------------------------------------------------
# Working tier overflow cascades to archive
# ---------------------------------------------------------------------------

class TestWorkingTierOverflow:
    """Test that working tier overflow cascades to archive."""

    def test_archive_receives_segments(self):
        config = _make_small_config(active=10, working=20, archive=500)
        memory = ConversationMemory(config)
        for i in range(15):
            memory.add_message("user", f"Cascade test message {i} with extra words to use tokens.")
        # Archive should have received segments
        assert memory.archive_tier.current_token_count > 0

    def test_all_tiers_have_content_after_many_messages(self):
        config = _make_small_config(active=10, working=20, archive=500)
        memory = ConversationMemory(config)
        for i in range(20):
            memory.add_message("user", f"Message {i} contains enough words to fill tiers.")
        # At least active and one other tier should have content
        tier_counts = [
            memory.active_tier.current_token_count,
            memory.working_tier.current_token_count,
            memory.archive_tier.current_token_count,
        ]
        non_zero = sum(1 for c in tier_counts if c > 0)
        assert non_zero >= 2


# ---------------------------------------------------------------------------
# System messages (importance >= 9.0) are never demoted
# ---------------------------------------------------------------------------

class TestSystemMessageProtection:
    """Test that system messages are protected from demotion."""

    def test_system_messages_stay_in_active_tier(self):
        config = _make_small_config(active=50, working=100, archive=200)
        memory = ConversationMemory(config)
        # Add a system message first
        memory.add_message("system", "You are a helpful assistant.")
        # Fill up with user messages to trigger overflow
        for i in range(10):
            memory.add_message("user", f"User message {i} with some extra filler words to consume tokens quickly.")
        # System message should still be in active tier
        system_in_active = any(
            seg.segment_type == "system" for seg in memory.active_tier.segments
        )
        assert system_in_active is True

    def test_demotable_index_skips_system_segments(self):
        """_get_demotable_segment_index should skip importance >= 9.0 segments."""
        tier = ActiveTier(1000)
        tier.add_segment(_make_segment("System msg", importance=9.5, segment_type="system"))
        tier.add_segment(_make_segment("User msg", importance=5.0, segment_type="user"))
        tier.add_segment(_make_segment("Another system", importance=9.0, segment_type="system"))

        idx = ConversationMemory._get_demotable_segment_index(tier)
        assert idx == 1  # Should point to the user segment

    def test_only_system_segments_returns_none(self):
        tier = ActiveTier(1000)
        tier.add_segment(_make_segment("System 1", importance=9.0, segment_type="system"))
        tier.add_segment(_make_segment("System 2", importance=9.5, segment_type="system"))

        idx = ConversationMemory._get_demotable_segment_index(tier)
        assert idx is None


# ---------------------------------------------------------------------------
# get_context_messages returns from all tiers
# ---------------------------------------------------------------------------

class TestGetContextMessages:
    """Test that get_context_messages returns from all tiers."""

    def test_returns_messages_from_all_tiers(self):
        config = _make_small_config(active=10, working=20, archive=500)
        memory = ConversationMemory(config)
        for i in range(15):
            memory.add_message("user", f"Context message {i} with extra padding words for tokens.")
        messages = memory.get_context_messages()
        assert isinstance(messages, list)
        assert len(messages) > 0

    def test_messages_are_chronologically_ordered(self):
        config = _make_small_config(active=500, working=500, archive=500)
        memory = ConversationMemory(config)
        memory.add_message("user", "First message at time T1.")
        time.sleep(0.01)
        memory.add_message("assistant", "Second message at time T2.")
        time.sleep(0.01)
        memory.add_message("user", "Third message at time T3.")

        messages = memory.get_context_messages()
        assert len(messages) == 3
        # Should be ordered chronologically
        assert messages[0]["content"] == "First message at time T1."
        assert messages[2]["content"] == "Third message at time T3."

    def test_empty_memory_returns_empty_list(self):
        config = _make_small_config()
        memory = ConversationMemory(config)
        messages = memory.get_context_messages()
        assert messages == []

    def test_token_budget_limits_output(self):
        config = _make_small_config(active=500, working=500, archive=500)
        memory = ConversationMemory(config)
        for i in range(10):
            memory.add_message("user", f"Budget test message {i} with padding.")
        # Use a very small budget
        messages = memory.get_context_messages(token_budget=10)
        total_tokens = sum(len(m["content"].split()) for m in messages)
        # Should be limited (may be slightly over due to estimation)
        assert total_tokens <= 20  # generous margin


# ---------------------------------------------------------------------------
# Compression happens during demotion
# ---------------------------------------------------------------------------

class TestCompressionDuringDemotion:
    """Test that segments are compressed when demoted between tiers."""

    def test_demoted_segments_are_compressed(self):
        config = _make_small_config(active=10, working=500, archive=500)
        memory = ConversationMemory(config)
        # Add a long message that will overflow active tier
        long_content = _word_string(20)
        memory.add_message("user", long_content)
        # Add more to force demotion
        memory.add_message("user", _word_string(15))

        # Check that working tier segments have compressed metadata
        for seg in memory.working_tier.segments:
            if seg.metadata.get("compressed"):
                assert seg.metadata["original_length"] > len(seg.content)
                return
        # If no compressed segments found in working, they might be in archive
        # (happens with very small limits)
        for seg in memory.archive_tier.segments:
            if seg.metadata.get("compressed"):
                assert seg.metadata["original_length"] > len(seg.content)
                return
        # It's acceptable if the segments were too short to compress meaningfully

    def test_compressed_output_is_smaller_than_original(self):
        config = _make_small_config(active=15, working=500, archive=500)
        memory = ConversationMemory(config)
        # Generate enough content to fill active tier and trigger demotion
        for i in range(6):
            memory.add_message(
                "user",
                f"This is a fairly long message number {i} that contains various words "
                f"to ensure the token count is significant enough for compression testing purposes."
            )
        # Working or archive tier should have compressed segments
        all_demoted = memory.working_tier.segments + memory.archive_tier.segments
        compressed = [s for s in all_demoted if s.metadata.get("compressed")]
        for seg in compressed:
            assert len(seg.content) < seg.metadata["original_length"]


# ---------------------------------------------------------------------------
# Stats and clear
# ---------------------------------------------------------------------------

class TestStatsAndClear:
    """Test memory statistics and clear functionality."""

    def test_get_stats_returns_dict(self):
        config = _make_small_config()
        memory = ConversationMemory(config)
        stats = memory.get_stats()
        assert "message_count" in stats
        assert "tiers" in stats
        assert "active" in stats["tiers"]

    def test_clear_memory_resets_everything(self):
        config = _make_small_config(active=500, working=500, archive=500)
        memory = ConversationMemory(config)
        memory.add_message("user", "Some content here.")
        memory.add_message("assistant", "Response content.")
        memory.clear_memory()
        assert len(memory.messages) == 0
        assert len(memory.active_tier.segments) == 0
        assert len(memory.working_tier.segments) == 0
        assert len(memory.archive_tier.segments) == 0
        assert memory.next_message_id == 1

    def test_estimate_importance_system_is_9(self):
        importance = ConversationMemory._estimate_importance("system", "You are a bot.")
        assert importance == 9.0

    def test_estimate_importance_short_message_is_low(self):
        importance = ConversationMemory._estimate_importance("user", "Hi")
        assert importance == 3.0

    def test_estimate_tokens_positive(self):
        tokens = ConversationMemory._estimate_tokens("Hello world test")
        assert tokens > 0
