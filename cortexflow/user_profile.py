"""
User profile model for CortexFlow.

Consolidates extracted personal facts into a structured user profile,
including preferences, communication style, boundaries, and relationship
context for companion AI applications.
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
class UserProfile:
    """Structured representation of everything known about a user."""

    user_id: str
    name: str | None = None
    demographics: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    personality_traits: list[str] = field(default_factory=list)
    communication_style: dict[str, Any] = field(default_factory=dict)
    interests: list[str] = field(default_factory=list)
    boundaries: list[dict[str, Any]] = field(default_factory=list)
    relationship_context: dict[str, Any] = field(default_factory=dict)
    emotional_patterns: dict[str, Any] = field(default_factory=dict)
    routines: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "demographics": self.demographics,
            "preferences": self.preferences,
            "personality_traits": self.personality_traits,
            "communication_style": self.communication_style,
            "interests": self.interests,
            "boundaries": self.boundaries,
            "relationship_context": self.relationship_context,
            "emotional_patterns": self.emotional_patterns,
            "routines": self.routines,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        return cls(
            user_id=data["user_id"],
            name=data.get("name"),
            demographics=data.get("demographics", {}),
            preferences=data.get("preferences", {}),
            personality_traits=data.get("personality_traits", []),
            communication_style=data.get("communication_style", {}),
            interests=data.get("interests", []),
            boundaries=data.get("boundaries", []),
            relationship_context=data.get("relationship_context", {}),
            emotional_patterns=data.get("emotional_patterns", {}),
            routines=data.get("routines", []),
            updated_at=data.get("updated_at", time.time()),
        )


class UserProfileManager:
    """Builds and maintains ``UserProfile`` instances from extracted facts.

    Persists profiles to SQLite so they survive across sessions.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        # In-memory cache
        self._profiles: dict[str, UserProfile] = {}

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> UserProfile:
        """Return the profile for *user_id*, creating an empty one if needed."""
        with self._lock:
            if user_id in self._profiles:
                return self._profiles[user_id]
            return self._load_or_create(user_id)

    def update_from_facts(self, user_id: str, facts: list[dict[str, Any]]) -> UserProfile:
        """Merge a list of extracted facts into the user's profile.

        Each fact dict should have at least ``fact_type`` and ``value``.
        """
        profile = self.get_profile(user_id)
        for fact in facts:
            self._apply_fact(profile, fact)
        profile.updated_at = time.time()
        self._save(profile)
        return profile

    def update_from_message(self, user_id: str, content: str,
                            fact_detector=None) -> UserProfile:
        """Extract facts from *content* via *fact_detector* and merge them."""
        if fact_detector is None:
            return self.get_profile(user_id)
        try:
            facts = fact_detector.detect_facts(content)
            if facts:
                return self.update_from_facts(user_id, facts)
        except Exception as exc:
            logger.debug("Profile update from message failed: %s", exc)
        return self.get_profile(user_id)

    def record_boundary(self, user_id: str, boundary: str,
                        category: str = "general") -> None:
        """Record a user-stated boundary (critical for safety)."""
        profile = self.get_profile(user_id)
        entry = {
            "boundary": boundary,
            "category": category,
            "recorded_at": time.time(),
        }
        # Avoid duplicates
        if not any(b["boundary"] == boundary for b in profile.boundaries):
            profile.boundaries.append(entry)
            profile.updated_at = time.time()
            self._save(profile)

    def check_boundary(self, user_id: str, text: str) -> dict[str, Any] | None:
        """Check whether *text* might violate any recorded boundary.

        Returns a dict with ``boundary`` and ``match`` keys if a potential
        violation is found, else ``None``.
        """
        profile = self.get_profile(user_id)
        text_lower = text.lower()
        for entry in profile.boundaries:
            boundary_lower = entry["boundary"].lower()
            # Simple keyword overlap check
            boundary_words = set(boundary_lower.split())
            if len(boundary_words) <= 2:
                # Short boundary — require exact substring
                if boundary_lower in text_lower:
                    return {"boundary": entry["boundary"], "match": "exact"}
            else:
                # Longer boundary — require >50% word overlap
                text_words = set(text_lower.split())
                overlap = boundary_words & text_words
                if len(overlap) / len(boundary_words) > 0.5:
                    return {"boundary": entry["boundary"], "match": "keyword_overlap"}
        return None

    def get_profile_summary(self, user_id: str) -> str:
        """Return a concise natural-language summary of the user's profile."""
        profile = self.get_profile(user_id)
        parts = []
        if profile.name:
            parts.append(f"Name: {profile.name}")
        if profile.demographics:
            demo_parts = [f"{k}: {v}" for k, v in profile.demographics.items()]
            parts.append("Demographics: " + ", ".join(demo_parts))
        if profile.interests:
            parts.append("Interests: " + ", ".join(profile.interests[:10]))
        if profile.preferences:
            pref_parts = [f"{k}: {v}" for k, v in list(profile.preferences.items())[:5]]
            parts.append("Preferences: " + ", ".join(pref_parts))
        if profile.personality_traits:
            parts.append("Personality: " + ", ".join(profile.personality_traits[:5]))
        if profile.routines:
            parts.append("Routines: " + ", ".join(profile.routines[:5]))
        if profile.boundaries:
            parts.append(f"Boundaries: {len(profile.boundaries)} recorded")
        return "\n".join(parts) if parts else "No profile information yet."

    def get_profile_for_prompt(self, user_id: str) -> str:
        """Return profile text suitable for injection into a system prompt."""
        summary = self.get_profile_summary(user_id)
        if summary == "No profile information yet.":
            return ""
        return f"[User Profile]\n{summary}"

    def delete_profile(self, user_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
            )
            self._conn.commit()
            self._profiles.pop(user_id, None)
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self._profiles.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_or_create(self, user_id: str) -> UserProfile:
        row = self._conn.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            profile = UserProfile.from_dict(json.loads(row["profile_json"]))
        else:
            profile = UserProfile(user_id=user_id)
        self._profiles[user_id] = profile
        return profile

    def _save(self, profile: UserProfile) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO user_profiles (user_id, profile_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       profile_json = excluded.profile_json,
                       updated_at = excluded.updated_at""",
                (profile.user_id, json.dumps(profile.to_dict()), profile.updated_at),
            )
            self._conn.commit()
            self._profiles[profile.user_id] = profile

    def _apply_fact(self, profile: UserProfile, fact: dict[str, Any]) -> None:
        """Apply a single extracted fact to the profile."""
        fact_type = fact.get("fact_type", "")
        value = fact.get("value", "")
        if not value:
            return

        if fact_type == "name":
            profile.name = value
        elif fact_type == "age":
            profile.demographics["age"] = value
        elif fact_type == "location":
            profile.demographics["location"] = value
        elif fact_type == "occupation":
            profile.demographics["occupation"] = value
        elif fact_type == "preference":
            # value may be "food: pizza" or just "pizza"
            if ":" in value:
                k, v = value.split(":", 1)
                profile.preferences[k.strip()] = v.strip()
            else:
                profile.preferences[value] = True
        elif fact_type == "interest":
            if value not in profile.interests:
                profile.interests.append(value)
        elif fact_type == "possession":
            if value not in profile.interests:
                profile.interests.append(value)
        elif fact_type == "routine":
            if value not in profile.routines:
                profile.routines.append(value)
        elif fact_type == "relationship_status":
            profile.relationship_context["status"] = value
        elif fact_type == "emotional_state":
            profile.emotional_patterns["latest"] = value
            profile.emotional_patterns["last_updated"] = time.time()
        elif fact_type == "boundary":
            self.record_boundary(profile.user_id, value)
        else:
            # Generic — store in preferences
            profile.preferences[fact_type] = value
