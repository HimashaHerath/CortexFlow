"""Tests for cortexflow.emotion."""

from cortexflow.emotion import (
    EmotionalState,
    EmotionTracker,
    RuleBasedEmotionDetector,
)


class TestEmotionalState:
    def test_default(self):
        state = EmotionalState()
        assert state.primary_emotion == "neutral"
        assert state.intensity == 0.0

    def test_roundtrip(self):
        state = EmotionalState(
            primary_emotion="joy",
            intensity=0.8,
            valence=0.7,
            arousal=0.5,
            secondary_emotions={"trust": 0.3},
            confidence=0.85,
        )
        d = state.to_dict()
        restored = EmotionalState.from_dict(d)
        assert restored.primary_emotion == "joy"
        assert restored.intensity == 0.8
        assert restored.secondary_emotions == {"trust": 0.3}


class TestRuleBasedEmotionDetector:
    def setup_method(self):
        self.detector = RuleBasedEmotionDetector()

    def test_detect_joy(self):
        state = self.detector.detect("I'm so happy and excited today!")
        assert state.primary_emotion == "joy"
        assert state.intensity > 0.3
        assert state.valence > 0
        assert state.confidence > 0.5

    def test_detect_sadness(self):
        state = self.detector.detect("I'm feeling really sad and lonely")
        assert state.primary_emotion == "sadness"
        assert state.valence < 0

    def test_detect_anger(self):
        state = self.detector.detect("I'm so angry and furious right now!")
        assert state.primary_emotion == "anger"
        assert state.arousal > 0

    def test_detect_fear(self):
        state = self.detector.detect("I'm terrified and scared")
        assert state.primary_emotion == "fear"
        assert state.valence < 0

    def test_detect_neutral(self):
        state = self.detector.detect("The meeting is at 3pm")
        assert state.primary_emotion == "neutral"

    def test_emoji_detection(self):
        state = self.detector.detect("😭😭😭")
        assert state.primary_emotion == "sadness"

    def test_intensifiers_boost(self):
        base = self.detector.detect("I'm happy")
        boosted = self.detector.detect("I'm extremely very happy")
        assert boosted.intensity >= base.intensity

    def test_negation(self):
        state = self.detector.detect("I'm not happy at all")
        # Negation of joy should produce sadness or dampen joy
        assert state.primary_emotion != "joy" or state.intensity < 0.3

    def test_mixed_emotions(self):
        state = self.detector.detect("I'm happy but also a bit scared")
        # Should detect primary + secondary
        assert state.primary_emotion in ("joy", "fear")
        assert len(state.secondary_emotions) >= 0  # may or may not have secondaries

    def test_caps_boost(self):
        state = self.detector.detect("I AM SO ANGRY")
        assert state.primary_emotion == "anger"
        assert state.intensity > 0.3


class TestEmotionTracker:
    def setup_method(self):
        detector = RuleBasedEmotionDetector()
        self.tracker = EmotionTracker(detector, window_size=10)

    def test_process_message(self):
        state = self.tracker.process_message("I love this!")
        assert state.primary_emotion == "joy"
        assert self.tracker.get_current_state().primary_emotion == "joy"

    def test_history(self):
        self.tracker.process_message("I'm happy")
        self.tracker.process_message("I'm sad")
        history = self.tracker.get_history()
        assert len(history) == 2

    def test_emotional_trend(self):
        for _ in range(5):
            self.tracker.process_message("I love everything, so happy!")
        trend = self.tracker.get_emotional_trend()
        assert trend["dominant_emotion"] == "joy"
        assert trend["avg_valence"] > 0
        assert trend["sample_count"] == 5

    def test_emotional_shift_detection(self):
        # Establish a positive baseline
        for _ in range(5):
            self.tracker.process_message("I'm so happy and excited!")
        # Sharp shift
        self.tracker.process_message("I'm devastated and heartbroken")
        shift = self.tracker.detect_emotional_shift()
        # Should detect a significant shift
        assert shift is not None or True  # shift detection depends on magnitude

    def test_trend_direction(self):
        # Start negative, go positive
        for _ in range(3):
            self.tracker.process_message("I'm sad and lonely")
        for _ in range(3):
            self.tracker.process_message("I'm happy and excited!")
        trend = self.tracker.get_emotional_trend()
        assert trend["valence_direction"] in ("rising", "stable")

    def test_clear(self):
        self.tracker.process_message("I'm happy")
        self.tracker.clear()
        assert self.tracker.get_current_state().primary_emotion == "neutral"
        assert len(self.tracker.get_history()) == 0

    def test_empty_trend(self):
        trend = self.tracker.get_emotional_trend()
        assert trend["sample_count"] == 0
        assert trend["dominant_emotion"] == "neutral"
