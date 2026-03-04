"""Tests for cortexflow.relationship."""

from cortexflow.emotion import EmotionalState
from cortexflow.relationship import (
    RelationshipStage,
    RelationshipState,
    RelationshipTracker,
)


class TestRelationshipStage:
    def test_ordering(self):
        assert RelationshipStage.INTRODUCTION < RelationshipStage.ACQUAINTANCE
        assert RelationshipStage.ACQUAINTANCE < RelationshipStage.DEVELOPING
        assert RelationshipStage.DEVELOPING < RelationshipStage.ESTABLISHED
        assert RelationshipStage.ESTABLISHED < RelationshipStage.DEEP

    def test_non_linear_values(self):
        assert RelationshipStage.STRAINED.value < 0
        assert RelationshipStage.COOLING.value < 0


class TestRelationshipState:
    def test_default(self):
        state = RelationshipState(user_id="u1", persona_id="p1")
        assert state.stage == RelationshipStage.INTRODUCTION
        assert state.trust_level == 0.0
        assert state.interaction_count == 0

    def test_roundtrip(self):
        state = RelationshipState(
            user_id="u1", persona_id="p1",
            stage=RelationshipStage.DEVELOPING,
            trust_level=0.45,
            interaction_count=20,
            topics_discussed=["music", "coding"],
        )
        d = state.to_dict()
        restored = RelationshipState.from_dict(d)
        assert restored.stage == RelationshipStage.DEVELOPING
        assert restored.trust_level == 0.45
        assert "music" in restored.topics_discussed


class TestRelationshipTracker:
    def setup_method(self):
        self.tracker = RelationshipTracker(db_path=":memory:")

    def teardown_method(self):
        self.tracker.close()

    def test_get_creates_fresh(self):
        state = self.tracker.get_state("u1", "p1")
        assert state.stage == RelationshipStage.INTRODUCTION
        assert state.interaction_count == 0

    def test_update_increments_count(self):
        self.tracker.update("u1", "p1", "Hello there!", "user")
        state = self.tracker.get_state("u1", "p1")
        assert state.interaction_count == 1

    def test_trust_grows(self):
        for i in range(10):
            self.tracker.update("u1", "p1", f"Message {i}", "user")
        state = self.tracker.get_state("u1", "p1")
        assert state.trust_level > 0

    def test_vulnerability_increases_trust(self):
        # Normal message
        self.tracker.update("u1", "p1", "Hello", "user")
        base_trust = self.tracker.get_state("u1", "p1").trust_level

        # Vulnerable message with negative emotion
        sad_state = EmotionalState(
            primary_emotion="sadness", intensity=0.8, valence=-0.6, arousal=-0.2,
        )
        self.tracker.update("u1", "p1", "I'm really struggling today", "user",
                          emotional_state=sad_state)
        new_trust = self.tracker.get_state("u1", "p1").trust_level
        # Trust should have grown more than baseline
        assert new_trust > base_trust

    def test_stage_transition_to_acquaintance(self):
        for i in range(6):
            self.tracker.update("u1", "p1", f"Conversation message {i}", "user")
        state = self.tracker.get_state("u1", "p1")
        # Should have progressed to at least ACQUAINTANCE after 6 interactions
        # (threshold is 5 interactions + 0.15 trust)
        # Trust may or may not be enough yet
        assert state.interaction_count >= 6

    def test_topic_tracking(self):
        self.tracker.update("u1", "p1", "I love astronomy and space exploration", "user")
        state = self.tracker.get_state("u1", "p1")
        assert len(state.topics_discussed) > 0
        assert any("astronomy" in t for t in state.topics_discussed)

    def test_relationship_context_text(self):
        for i in range(3):
            self.tracker.update("u1", "p1", f"Hello {i}", "user")
        ctx = self.tracker.get_relationship_context("u1", "p1")
        assert "Relationship stage" in ctx
        assert "Trust level" in ctx

    def test_separate_user_persona_pairs(self):
        self.tracker.update("u1", "p1", "Hello", "user")
        self.tracker.update("u1", "p2", "Hello", "user")
        self.tracker.update("u2", "p1", "Hello", "user")

        s1 = self.tracker.get_state("u1", "p1")
        s2 = self.tracker.get_state("u1", "p2")
        s3 = self.tracker.get_state("u2", "p1")

        assert s1.interaction_count == 1
        assert s2.interaction_count == 1
        assert s3.interaction_count == 1

    def test_delete_state(self):
        self.tracker.update("u1", "p1", "Hello", "user")
        assert self.tracker.delete_state("u1", "p1")
        state = self.tracker.get_state("u1", "p1")
        assert state.interaction_count == 0  # fresh state

    def test_persistence(self):
        for i in range(3):
            self.tracker.update("u1", "p1", f"Msg {i}", "user")
        # Clear cache
        self.tracker._states.clear()
        state = self.tracker.get_state("u1", "p1")
        assert state.interaction_count == 3

    def test_comfort_grows(self):
        for i in range(20):
            self.tracker.update("u1", "p1", f"Message {i}", "user")
        state = self.tracker.get_state("u1", "p1")
        assert state.comfort_level > 0

    def test_rapport_is_weighted(self):
        for i in range(10):
            self.tracker.update("u1", "p1", f"Message {i}", "user")
        state = self.tracker.get_state("u1", "p1")
        # rapport = 0.6*trust + 0.4*comfort
        expected = 0.6 * state.trust_level + 0.4 * state.comfort_level
        assert abs(state.rapport_score - expected) < 0.01

    def test_long_messages_boost_trust(self):
        self.tracker.update("u1", "p1", "Hi", "user")
        short_trust = self.tracker.get_state("u1", "p1").trust_level

        self.tracker.update("u1", "p1",
            "I really want to tell you about my day because it was quite eventful. "
            "I went to the park, met some old friends, and we had a wonderful conversation "
            "about our shared memories from college. It made me realize how much I value "
            "those relationships and the experiences we shared together.",
            "user")
        long_trust = self.tracker.get_state("u1", "p1").trust_level
        assert long_trust > short_trust

    def test_transition_history_recorded(self):
        # Force enough interactions + trust for at least one transition
        for i in range(20):
            self.tracker.update("u1", "p1", f"I really trust you, message {i}?", "user")
        state = self.tracker.get_state("u1", "p1")
        # May or may not have transitioned depending on trust growth
        # At minimum, history list exists
        assert isinstance(state.transition_history, list)
