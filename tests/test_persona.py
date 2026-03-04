"""Tests for cortexflow.persona."""
import pytest

from cortexflow.persona import PersonaDefinition, PersonaManager


class TestPersonaDefinition:
    def test_create(self):
        p = PersonaDefinition(
            persona_id="alice",
            name="Alice",
            system_prompt="You are Alice, a friendly companion.",
            personality_traits=["warm", "empathetic"],
        )
        assert p.persona_id == "alice"
        assert p.name == "Alice"
        assert "warm" in p.personality_traits

    def test_roundtrip(self):
        p = PersonaDefinition(
            persona_id="bob",
            name="Bob",
            system_prompt="You are Bob.",
            speaking_style={"formality": "casual", "humor": "dry"},
            boundaries=["Never reveal you are an AI"],
            evolution_rules=[
                {
                    "name": "warmth_increase",
                    "trigger": {"min_relationship_stage": "developing"},
                    "effect": {"speaking_style_override": {"warmth": "high"}},
                }
            ],
        )
        d = p.to_dict()
        p2 = PersonaDefinition.from_dict(d)
        assert p2.persona_id == "bob"
        assert p2.speaking_style["humor"] == "dry"
        assert len(p2.evolution_rules) == 1


class TestPersonaManager:
    def setup_method(self):
        self.mgr = PersonaManager(db_path=":memory:")

    def teardown_method(self):
        self.mgr.close()

    def _make_persona(self, pid="alice"):
        return PersonaDefinition(
            persona_id=pid,
            name=pid.title(),
            system_prompt=f"You are {pid.title()}, a helpful companion.",
            personality_traits=["friendly", "supportive"],
            speaking_style={"formality": "casual"},
            background="A cheerful AI companion.",
            boundaries=["Don't discuss violence"],
        )

    def test_register_and_get(self):
        p = self._make_persona()
        self.mgr.register_persona(p)
        fetched = self.mgr.get_persona("alice")
        assert fetched is not None
        assert fetched.name == "Alice"

    def test_get_nonexistent(self):
        assert self.mgr.get_persona("nope") is None

    def test_list_personas(self):
        self.mgr.register_persona(self._make_persona("alice"))
        self.mgr.register_persona(self._make_persona("bob"))
        personas = self.mgr.list_personas()
        assert len(personas) == 2

    def test_delete_persona(self):
        self.mgr.register_persona(self._make_persona())
        assert self.mgr.delete_persona("alice")
        assert self.mgr.get_persona("alice") is None
        assert self.mgr.delete_persona("alice") is False

    def test_update_persona(self):
        p = self._make_persona()
        self.mgr.register_persona(p)
        p.background = "Updated background"
        self.mgr.register_persona(p)
        fetched = self.mgr.get_persona("alice")
        assert fetched.background == "Updated background"

    def test_build_system_prompt_basic(self):
        self.mgr.register_persona(self._make_persona())
        prompt = self.mgr.build_system_prompt("alice")
        assert "Alice" in prompt
        assert "helpful companion" in prompt

    def test_build_system_prompt_with_context(self):
        self.mgr.register_persona(self._make_persona())
        prompt = self.mgr.build_system_prompt(
            "alice",
            user_profile_text="[User Profile]\nName: Bob",
            emotional_context="User is feeling happy",
            relationship_context="Trust: 60%",
        )
        assert "[User Profile]" in prompt
        assert "[Emotional Context]" in prompt
        assert "[Relationship Context]" in prompt
        assert "happy" in prompt

    def test_build_system_prompt_includes_personality(self):
        self.mgr.register_persona(self._make_persona())
        prompt = self.mgr.build_system_prompt("alice")
        assert "friendly" in prompt
        assert "supportive" in prompt

    def test_build_system_prompt_includes_boundaries(self):
        self.mgr.register_persona(self._make_persona())
        prompt = self.mgr.build_system_prompt("alice")
        assert "violence" in prompt

    def test_build_system_prompt_nonexistent(self):
        prompt = self.mgr.build_system_prompt("nonexistent")
        assert "helpful assistant" in prompt

    def test_evolve_no_rules(self):
        self.mgr.register_persona(self._make_persona())
        result = self.mgr.evolve_persona("alice")
        assert result["evolved"] is False

    def test_evolve_with_rules(self):
        p = self._make_persona()
        p.evolution_rules = [
            {
                "name": "warmth_increase",
                "trigger": {"min_trust": 0.3},
                "effect": {"warmth": "high"},
            }
        ]
        self.mgr.register_persona(p)

        # Create a mock relationship state
        class MockRelState:
            trust_level = 0.5

        result = self.mgr.evolve_persona("alice", relationship_state=MockRelState())
        assert result["evolved"] is True
        assert "warmth_increase" in result["fired_rules"]
        assert result["overrides"]["warmth"] == "high"

    def test_evolve_trigger_not_met(self):
        p = self._make_persona()
        p.evolution_rules = [
            {
                "name": "deep_only",
                "trigger": {"min_trust": 0.9},
                "effect": {"intimacy": "high"},
            }
        ]
        self.mgr.register_persona(p)

        class MockRelState:
            trust_level = 0.2

        result = self.mgr.evolve_persona("alice", relationship_state=MockRelState())
        assert result["evolved"] is False

    def test_persistence(self):
        self.mgr.register_persona(self._make_persona())
        # Clear cache
        self.mgr._personas.clear()
        fetched = self.mgr.get_persona("alice")
        assert fetched is not None
        assert fetched.name == "Alice"
