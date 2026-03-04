"""
Relationship state machine for CortexFlow.

Models the evolving relationship between user and AI persona across
conversation sessions.  Tracks trust, comfort, rapport, and manages
stage transitions based on interaction signals.
"""

from __future__ import annotations

import enum
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cortexflow")


class RelationshipStage(enum.IntEnum):
    """Ordered stages of a user–persona relationship."""

    INTRODUCTION = 0
    ACQUAINTANCE = 1
    DEVELOPING = 2
    ESTABLISHED = 3
    DEEP = 4
    # Non-linear states (can occur at any point)
    STRAINED = -1
    COOLING = -2
    RECONNECTING = -3


# Minimum interaction counts for forward transitions
_STAGE_THRESHOLDS: dict[RelationshipStage, int] = {
    RelationshipStage.ACQUAINTANCE: 5,
    RelationshipStage.DEVELOPING: 15,
    RelationshipStage.ESTABLISHED: 40,
    RelationshipStage.DEEP: 100,
}

# Minimum trust for forward transitions
_TRUST_THRESHOLDS: dict[RelationshipStage, float] = {
    RelationshipStage.ACQUAINTANCE: 0.15,
    RelationshipStage.DEVELOPING: 0.30,
    RelationshipStage.ESTABLISHED: 0.55,
    RelationshipStage.DEEP: 0.80,
}


@dataclass
class RelationshipState:
    """Mutable snapshot of a user–persona relationship."""

    user_id: str
    persona_id: str
    stage: RelationshipStage = RelationshipStage.INTRODUCTION
    trust_level: float = 0.0  # 0–1
    comfort_level: float = 0.0  # 0–1
    rapport_score: float = 0.0  # 0–1
    interaction_count: int = 0
    topics_discussed: list[str] = field(default_factory=list)
    shared_experiences: list[str] = field(default_factory=list)
    transition_history: list[dict[str, Any]] = field(default_factory=list)
    last_interaction_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "persona_id": self.persona_id,
            "stage": self.stage.name,
            "trust_level": round(self.trust_level, 4),
            "comfort_level": round(self.comfort_level, 4),
            "rapport_score": round(self.rapport_score, 4),
            "interaction_count": self.interaction_count,
            "topics_discussed": self.topics_discussed,
            "shared_experiences": self.shared_experiences,
            "transition_history": self.transition_history,
            "last_interaction_at": self.last_interaction_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipState:
        stage_val = data.get("stage", "INTRODUCTION")
        if isinstance(stage_val, str):
            stage = RelationshipStage[stage_val.upper()]
        else:
            stage = RelationshipStage(stage_val)

        return cls(
            user_id=data["user_id"],
            persona_id=data["persona_id"],
            stage=stage,
            trust_level=data.get("trust_level", 0.0),
            comfort_level=data.get("comfort_level", 0.0),
            rapport_score=data.get("rapport_score", 0.0),
            interaction_count=data.get("interaction_count", 0),
            topics_discussed=data.get("topics_discussed", []),
            shared_experiences=data.get("shared_experiences", []),
            transition_history=data.get("transition_history", []),
            last_interaction_at=data.get("last_interaction_at", time.time()),
            created_at=data.get("created_at", time.time()),
        )


class RelationshipTracker:
    """Manages relationship state for all user–persona pairs.

    Persists state to SQLite.  On every message call ``update()`` to
    adjust metrics and potentially trigger stage transitions.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        # In-memory cache: (user_id, persona_id) → RelationshipState
        self._states: dict[tuple[str, str], RelationshipState] = {}

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS relationships (
                    user_id TEXT NOT NULL,
                    persona_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, persona_id)
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self, user_id: str, persona_id: str) -> RelationshipState:
        """Return the current relationship state, creating a fresh one if needed."""
        key = (user_id, persona_id)
        with self._lock:
            if key in self._states:
                return self._states[key]
            return self._load_or_create(user_id, persona_id)

    def update(
        self,
        user_id: str,
        persona_id: str,
        message: str,
        role: str,
        emotional_state=None,
    ) -> RelationshipState:
        """Call on every message to update relationship metrics.

        Args:
            user_id: The user.
            persona_id: The AI persona.
            message: The message text.
            role: ``"user"`` or ``"assistant"``.
            emotional_state: Optional ``EmotionalState`` for the message.

        Returns:
            The updated ``RelationshipState``.
        """
        state = self.get_state(user_id, persona_id)

        # --- Increment interaction count ---
        state.interaction_count += 1
        state.last_interaction_at = time.time()

        # --- Trust signals ---
        trust_delta = 0.0

        if role == "user":
            # Vulnerability signals increase trust
            if emotional_state and hasattr(emotional_state, "valence"):
                if emotional_state.valence < -0.3 and emotional_state.intensity > 0.5:
                    trust_delta += 0.02  # sharing negative emotions = vulnerability
                if emotional_state.primary_emotion in ("trust", "joy"):
                    trust_delta += 0.01

            # Message length signals engagement
            word_count = len(message.split())
            if word_count > 50:
                trust_delta += 0.005
            elif word_count > 20:
                trust_delta += 0.002

            # Questions indicate curiosity/engagement
            if "?" in message:
                trust_delta += 0.003

        # Baseline trust growth per interaction
        trust_delta += 0.005

        state.trust_level = min(1.0, max(0.0, state.trust_level + trust_delta))

        # --- Comfort ---
        # Comfort grows slowly with interaction count
        target_comfort = min(1.0, state.interaction_count / 200.0)
        state.comfort_level += (target_comfort - state.comfort_level) * 0.05

        # --- Rapport ---
        # Weighted average of trust and comfort
        state.rapport_score = 0.6 * state.trust_level + 0.4 * state.comfort_level

        # --- Topic tracking (keep last 50) ---
        # Simple heuristic: extract first 3 non-stopword tokens
        tokens = [
            w.lower().strip(".,!?;:")
            for w in message.split()
            if len(w) > 3 and w.lower() not in _STOPWORDS
        ]
        for t in tokens[:3]:
            if t and t not in state.topics_discussed:
                state.topics_discussed.append(t)
        state.topics_discussed = state.topics_discussed[-50:]

        # --- Stage transitions ---
        self._evaluate_transitions(state)

        # --- Persist ---
        self._save(state)
        return state

    def get_relationship_context(self, user_id: str, persona_id: str) -> str:
        """Return a text summary suitable for system-prompt injection."""
        state = self.get_state(user_id, persona_id)
        lines = [
            f"Relationship stage: {state.stage.name.replace('_', ' ').title()}",
            f"Trust level: {state.trust_level:.0%}",
            f"Comfort level: {state.comfort_level:.0%}",
            f"Rapport: {state.rapport_score:.0%}",
            f"Interactions: {state.interaction_count}",
        ]
        if state.topics_discussed:
            lines.append(f"Recent topics: {', '.join(state.topics_discussed[-10:])}")
        return "\n".join(lines)

    def delete_state(self, user_id: str, persona_id: str) -> bool:
        key = (user_id, persona_id)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM relationships WHERE user_id = ? AND persona_id = ?",
                (user_id, persona_id),
            )
            self._conn.commit()
            self._states.pop(key, None)
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self._states.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_transitions(self, state: RelationshipState) -> None:
        """Check if the relationship should move to the next stage."""
        current = state.stage

        # Handle non-linear states first
        if current in (
            RelationshipStage.STRAINED,
            RelationshipStage.COOLING,
            RelationshipStage.RECONNECTING,
        ):
            # Recovery: if trust has recovered above 0.2 and several interactions
            if state.trust_level > 0.2 and state.interaction_count > 5:
                # Return to the highest forward stage we qualify for
                for target in (
                    RelationshipStage.DEEP,
                    RelationshipStage.ESTABLISHED,
                    RelationshipStage.DEVELOPING,
                    RelationshipStage.ACQUAINTANCE,
                ):
                    if state.interaction_count >= _STAGE_THRESHOLDS.get(
                        target, 0
                    ) and state.trust_level >= _TRUST_THRESHOLDS.get(target, 0):
                        self._transition(state, target)
                        return
                self._transition(state, RelationshipStage.INTRODUCTION)
            return

        # Forward transitions
        for target in (
            RelationshipStage.DEEP,
            RelationshipStage.ESTABLISHED,
            RelationshipStage.DEVELOPING,
            RelationshipStage.ACQUAINTANCE,
        ):
            if target <= current:
                continue
            needed_count = _STAGE_THRESHOLDS.get(target, 0)
            needed_trust = _TRUST_THRESHOLDS.get(target, 0.0)
            if (
                state.interaction_count >= needed_count
                and state.trust_level >= needed_trust
            ):
                self._transition(state, target)
                return

    def _transition(
        self, state: RelationshipState, new_stage: RelationshipStage
    ) -> None:
        old_stage = state.stage
        if old_stage == new_stage:
            return
        state.stage = new_stage
        entry = {
            "from": old_stage.name,
            "to": new_stage.name,
            "timestamp": time.time(),
            "interaction_count": state.interaction_count,
            "trust_level": round(state.trust_level, 4),
        }
        state.transition_history.append(entry)
        logger.info(
            "Relationship %s/%s transitioned: %s → %s",
            state.user_id,
            state.persona_id,
            old_stage.name,
            new_stage.name,
        )

    def _load_or_create(self, user_id: str, persona_id: str) -> RelationshipState:
        row = self._conn.execute(
            "SELECT state_json FROM relationships WHERE user_id = ? AND persona_id = ?",
            (user_id, persona_id),
        ).fetchone()
        if row:
            state = RelationshipState.from_dict(json.loads(row["state_json"]))
        else:
            state = RelationshipState(user_id=user_id, persona_id=persona_id)
        self._states[(user_id, persona_id)] = state
        return state

    def _save(self, state: RelationshipState) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO relationships (user_id, persona_id, state_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, persona_id) DO UPDATE SET
                       state_json = excluded.state_json,
                       updated_at = excluded.updated_at""",
                (
                    state.user_id,
                    state.persona_id,
                    json.dumps(state.to_dict()),
                    time.time(),
                ),
            )
            self._conn.commit()
            self._states[(state.user_id, state.persona_id)] = state


# Minimal stopword set for topic extraction
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "were",
        "they",
        "their",
        "what",
        "when",
        "where",
        "which",
        "about",
        "would",
        "could",
        "should",
        "there",
        "these",
        "those",
        "then",
        "than",
        "them",
        "some",
        "into",
        "just",
        "also",
        "like",
        "very",
        "really",
        "your",
        "will",
        "more",
        "much",
        "only",
        "even",
        "because",
        "does",
        "doing",
        "being",
        "each",
        "other",
    }
)
