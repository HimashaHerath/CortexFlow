"""
Persona management for CortexFlow.

Defines AI persona identities with personality traits, speaking styles,
emotional baselines, and dynamic evolution rules.  Personas adapt
based on relationship state and conversation history.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cortexflow")


@dataclass
class PersonaDefinition:
    """Immutable definition of an AI persona."""

    persona_id: str
    name: str
    system_prompt: str
    personality_traits: list[str] = field(default_factory=list)
    speaking_style: dict[str, Any] = field(default_factory=dict)
    background: str = ""
    boundaries: list[str] = field(default_factory=list)
    emotional_baseline: dict[str, float] = field(default_factory=dict)
    evolution_rules: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "personality_traits": self.personality_traits,
            "speaking_style": self.speaking_style,
            "background": self.background,
            "boundaries": self.boundaries,
            "emotional_baseline": self.emotional_baseline,
            "evolution_rules": self.evolution_rules,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaDefinition:
        return cls(
            persona_id=data["persona_id"],
            name=data["name"],
            system_prompt=data["system_prompt"],
            personality_traits=data.get("personality_traits", []),
            speaking_style=data.get("speaking_style", {}),
            background=data.get("background", ""),
            boundaries=data.get("boundaries", []),
            emotional_baseline=data.get("emotional_baseline", {}),
            evolution_rules=data.get("evolution_rules", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
        )


class PersonaManager:
    """Registry and runtime manager for AI personas.

    Persists persona definitions to SQLite.  At runtime, dynamically
    composes system prompts by combining persona definition + user profile
    + emotional context + relationship state.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        # In-memory cache
        self._personas: dict[str, PersonaDefinition] = {}

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS personas (
                    persona_id TEXT PRIMARY KEY,
                    persona_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register_persona(self, persona: PersonaDefinition) -> None:
        """Register (or update) a persona definition."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO personas (persona_id, persona_json, created_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(persona_id) DO UPDATE SET
                       persona_json = excluded.persona_json""",
                (persona.persona_id, json.dumps(persona.to_dict()), persona.created_at),
            )
            self._conn.commit()
            self._personas[persona.persona_id] = persona
            logger.info("Registered persona: %s", persona.name)

    def get_persona(self, persona_id: str) -> PersonaDefinition | None:
        with self._lock:
            if persona_id in self._personas:
                return self._personas[persona_id]
            row = self._conn.execute(
                "SELECT persona_json FROM personas WHERE persona_id = ?",
                (persona_id,),
            ).fetchone()
            if row is None:
                return None
            persona = PersonaDefinition.from_dict(json.loads(row["persona_json"]))
            self._personas[persona_id] = persona
            return persona

    def list_personas(self) -> list[PersonaDefinition]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT persona_json FROM personas ORDER BY created_at"
            ).fetchall()
        return [PersonaDefinition.from_dict(json.loads(r["persona_json"])) for r in rows]

    def delete_persona(self, persona_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM personas WHERE persona_id = ?", (persona_id,)
            )
            self._conn.commit()
            self._personas.pop(persona_id, None)
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Dynamic prompt composition
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        persona_id: str,
        *,
        user_profile_text: str = "",
        emotional_context: str = "",
        relationship_context: str = "",
        extra_instructions: str = "",
    ) -> str:
        """Dynamically compose a system prompt from persona + context layers.

        Returns the persona's base ``system_prompt`` enriched with optional
        user profile, emotional state, and relationship context sections.
        """
        persona = self.get_persona(persona_id)
        if persona is None:
            return extra_instructions or "You are a helpful assistant."

        parts: list[str] = []

        # 1. Core persona identity
        parts.append(persona.system_prompt)

        # 2. Personality & style
        if persona.personality_traits:
            parts.append(
                f"\n[Personality]\nYour personality traits: {', '.join(persona.personality_traits)}."
            )
        if persona.speaking_style:
            style_desc = ", ".join(f"{k}: {v}" for k, v in persona.speaking_style.items())
            parts.append(f"\n[Speaking Style]\n{style_desc}")
        if persona.background:
            parts.append(f"\n[Background]\n{persona.background}")

        # 3. Persona boundaries
        if persona.boundaries:
            parts.append(
                "\n[Persona Boundaries]\n"
                + "\n".join(f"- {b}" for b in persona.boundaries)
            )

        # 4. Contextual layers (injected at runtime)
        if user_profile_text:
            parts.append(f"\n{user_profile_text}")
        if emotional_context:
            parts.append(f"\n[Emotional Context]\n{emotional_context}")
        if relationship_context:
            parts.append(f"\n[Relationship Context]\n{relationship_context}")
        if extra_instructions:
            parts.append(f"\n[Additional Instructions]\n{extra_instructions}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Persona evolution
    # ------------------------------------------------------------------

    def evolve_persona(self, persona_id: str, relationship_state=None,
                       emotional_trend: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply evolution rules to a persona based on relationship and emotion.

        Returns a dict describing which rules fired and what changed.
        This does NOT mutate the stored ``PersonaDefinition`` — it returns
        *runtime overrides* the caller can inject into prompt composition.
        """
        persona = self.get_persona(persona_id)
        if persona is None:
            return {"evolved": False, "reason": "persona_not_found"}

        overrides: dict[str, Any] = {}
        fired_rules: list[str] = []

        for rule in persona.evolution_rules:
            trigger = rule.get("trigger", {})
            effect = rule.get("effect", {})
            rule_name = rule.get("name", "unnamed")

            if self._check_trigger(trigger, relationship_state, emotional_trend):
                fired_rules.append(rule_name)
                overrides.update(effect)

        return {
            "evolved": bool(fired_rules),
            "fired_rules": fired_rules,
            "overrides": overrides,
        }

    def _check_trigger(self, trigger: dict[str, Any], relationship_state,
                       emotional_trend: dict[str, Any] | None) -> bool:
        """Check whether an evolution-rule trigger is satisfied."""
        # Relationship stage trigger
        req_stage = trigger.get("min_relationship_stage")
        if req_stage and relationship_state is not None:
            from cortexflow.relationship import RelationshipStage
            try:
                current = relationship_state.stage if hasattr(relationship_state, "stage") else relationship_state
                if isinstance(current, str):
                    current = RelationshipStage[current.upper()]
                required = RelationshipStage[req_stage.upper()]
                if current.value < required.value:
                    return False
            except (KeyError, AttributeError):
                return False

        # Trust threshold trigger
        min_trust = trigger.get("min_trust")
        if min_trust is not None and relationship_state is not None:
            trust = getattr(relationship_state, "trust_level", 0)
            if trust < min_trust:
                return False

        # Emotional valence trigger
        valence_req = trigger.get("min_avg_valence")
        if valence_req is not None and emotional_trend:
            if emotional_trend.get("avg_valence", 0) < valence_req:
                return False

        return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self._personas.clear()
