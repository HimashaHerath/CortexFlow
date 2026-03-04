"""Tests for cortexflow.user_profile."""

from cortexflow.fact_detector import PersonalFactDetector
from cortexflow.user_profile import UserProfile, UserProfileManager


class TestUserProfile:
    def test_default(self):
        p = UserProfile(user_id="u1")
        assert p.name is None
        assert p.interests == []
        assert p.boundaries == []

    def test_roundtrip(self):
        p = UserProfile(user_id="u1", name="Alice",
                        demographics={"age": "25"},
                        interests=["coding", "music"])
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        assert p2.name == "Alice"
        assert p2.interests == ["coding", "music"]


class TestUserProfileManager:
    def setup_method(self):
        self.mgr = UserProfileManager(db_path=":memory:")
        self.detector = PersonalFactDetector(use_spacy=False)

    def teardown_method(self):
        self.mgr.close()

    def test_get_creates_empty(self):
        p = self.mgr.get_profile("u1")
        assert p.user_id == "u1"
        assert p.name is None

    def test_update_from_facts(self):
        facts = [
            {"fact_type": "name", "value": "Alice"},
            {"fact_type": "age", "value": "25"},
            {"fact_type": "location", "value": "New York"},
            {"fact_type": "preference", "value": "food: pizza"},
            {"fact_type": "interest", "value": "coding"},
        ]
        p = self.mgr.update_from_facts("u1", facts)
        assert p.name == "Alice"
        assert p.demographics["age"] == "25"
        assert p.demographics["location"] == "New York"
        assert p.preferences["food"] == "pizza"
        assert "coding" in p.interests

    def test_update_from_message(self):
        p = self.mgr.update_from_message("u1", "My name is Bob", self.detector)
        assert p.name == "Bob"

    def test_persistence(self):
        self.mgr.update_from_facts("u1", [{"fact_type": "name", "value": "Alice"}])
        # Clear cache and reload
        self.mgr._profiles.clear()
        p = self.mgr.get_profile("u1")
        assert p.name == "Alice"

    def test_record_boundary(self):
        self.mgr.record_boundary("u1", "Don't talk about politics", "topics")
        p = self.mgr.get_profile("u1")
        assert len(p.boundaries) == 1
        assert p.boundaries[0]["boundary"] == "Don't talk about politics"

    def test_boundary_no_duplicates(self):
        self.mgr.record_boundary("u1", "no politics")
        self.mgr.record_boundary("u1", "no politics")
        p = self.mgr.get_profile("u1")
        assert len(p.boundaries) == 1

    def test_check_boundary_exact(self):
        self.mgr.record_boundary("u1", "politics")
        result = self.mgr.check_boundary("u1", "Let's talk about politics today")
        assert result is not None
        assert result["match"] == "exact"

    def test_check_boundary_overlap(self):
        self.mgr.record_boundary("u1", "don't discuss my family problems please")
        result = self.mgr.check_boundary("u1", "I want to discuss my family problems with you")
        assert result is not None

    def test_check_boundary_no_match(self):
        self.mgr.record_boundary("u1", "politics")
        result = self.mgr.check_boundary("u1", "Let's talk about weather")
        assert result is None

    def test_profile_summary(self):
        self.mgr.update_from_facts("u1", [
            {"fact_type": "name", "value": "Alice"},
            {"fact_type": "interest", "value": "music"},
        ])
        summary = self.mgr.get_profile_summary("u1")
        assert "Alice" in summary
        assert "music" in summary

    def test_profile_for_prompt(self):
        self.mgr.update_from_facts("u1", [{"fact_type": "name", "value": "Alice"}])
        prompt = self.mgr.get_profile_for_prompt("u1")
        assert "[User Profile]" in prompt

    def test_empty_profile_prompt(self):
        prompt = self.mgr.get_profile_for_prompt("new_user")
        assert prompt == ""

    def test_delete_profile(self):
        self.mgr.update_from_facts("u1", [{"fact_type": "name", "value": "Alice"}])
        assert self.mgr.delete_profile("u1")
        p = self.mgr.get_profile("u1")
        assert p.name is None

    def test_emotional_state_fact(self):
        facts = [{"fact_type": "emotional_state", "value": "anxious"}]
        p = self.mgr.update_from_facts("u1", facts)
        assert p.emotional_patterns["latest"] == "anxious"

    def test_routine_fact(self):
        facts = [{"fact_type": "routine", "value": "go jogging in the morning"}]
        p = self.mgr.update_from_facts("u1", facts)
        assert "go jogging in the morning" in p.routines


class TestFactDetectorCompanionPatterns:
    """Test the companion-specific patterns added to PersonalFactDetector."""

    def setup_method(self):
        self.detector = PersonalFactDetector(use_spacy=False)

    def test_relationship_status(self):
        facts = self.detector.detect_facts("I'm single and looking")
        types = [f["fact_type"] for f in facts]
        assert "relationship_status" in types

    def test_emotional_state_detection(self):
        facts = self.detector.detect_facts("I'm feeling lonely lately")
        types = [f["fact_type"] for f in facts]
        assert "emotional_state" in types

    def test_routine_detection(self):
        facts = self.detector.detect_facts("Every morning I go for a run")
        types = [f["fact_type"] for f in facts]
        assert "routine" in types

    def test_boundary_detection(self):
        facts = self.detector.detect_facts("I'm not comfortable talking about my ex")
        types = [f["fact_type"] for f in facts]
        assert "boundary" in types

    def test_interest_detection(self):
        facts = self.detector.detect_facts("I'm really interested in astronomy")
        types = [f["fact_type"] for f in facts]
        assert "interest" in types

    def test_quick_hints_companion(self):
        assert self.detector.contains_personal_fact("I'm single")
        assert self.detector.contains_personal_fact("I'm feeling sad")
        assert self.detector.contains_personal_fact("Every day I walk the dog")
        assert self.detector.contains_personal_fact("I'm not comfortable with that")
        assert self.detector.contains_personal_fact("I'm really interested in art")
