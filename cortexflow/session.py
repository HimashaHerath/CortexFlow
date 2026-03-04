"""
Session management for CortexFlow.

Provides per-user session isolation so that memory, knowledge, and
conversation state can be scoped to individual users and sessions.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cortexflow")


@dataclass
class SessionContext:
    """Represents an active conversation session."""

    session_id: str
    user_id: str
    persona_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def touch(self) -> None:
        """Update the last-active timestamp."""
        self.last_active_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "persona_id": self.persona_id,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "metadata": self.metadata,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionContext:
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            persona_id=data.get("persona_id"),
            created_at=data.get("created_at", time.time()),
            last_active_at=data.get("last_active_at", time.time()),
            metadata=data.get("metadata", {}),
            is_active=data.get("is_active", True),
        )


class SessionManager:
    """Manages user sessions with SQLite-backed persistence.

    Thread-safe: all public methods acquire ``_lock`` before touching
    the database or in-memory session cache.
    """

    def __init__(self, db_path: str = ":memory:", session_ttl: int = 86400,
                 max_sessions_per_user: int = 10):
        self._db_path = db_path
        self._session_ttl = session_ttl
        self._max_sessions_per_user = max_sessions_per_user
        self._lock = threading.RLock()

        # In-memory cache of active sessions keyed by session_id
        self._sessions: dict[str, SessionContext] = {}

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    persona_id TEXT,
                    created_at REAL NOT NULL,
                    last_active_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    is_active INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions (user_id, is_active);
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, user_id: str, persona_id: str | None = None,
                       metadata: dict[str, Any] | None = None) -> SessionContext:
        """Create a new session for *user_id*.

        If the user already has ``max_sessions_per_user`` active sessions the
        oldest one is automatically closed.
        """
        import json

        with self._lock:
            # Enforce per-user limit
            self._enforce_session_limit(user_id)

            session = SessionContext(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                persona_id=persona_id,
                metadata=metadata or {},
            )

            self._conn.execute(
                """INSERT INTO sessions
                   (session_id, user_id, persona_id, created_at, last_active_at,
                    metadata, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, 1)""",
                (session.session_id, session.user_id, session.persona_id,
                 session.created_at, session.last_active_at,
                 json.dumps(session.metadata)),
            )
            self._conn.commit()

            self._sessions[session.session_id] = session
            logger.info("Created session %s for user %s", session.session_id, user_id)
            return session

    def get_session(self, session_id: str) -> SessionContext | None:
        """Return the session with *session_id*, or ``None``."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            return self._load_session(session_id)

    def list_sessions(self, user_id: str, *, active_only: bool = True) -> list[SessionContext]:
        """List sessions belonging to *user_id*."""
        import json

        with self._lock:
            if active_only:
                rows = self._conn.execute(
                    "SELECT * FROM sessions WHERE user_id = ? AND is_active = 1 ORDER BY last_active_at DESC",
                    (user_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM sessions WHERE user_id = ? ORDER BY last_active_at DESC",
                    (user_id,),
                ).fetchall()

            sessions = []
            for row in rows:
                ctx = SessionContext(
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                    persona_id=row["persona_id"],
                    created_at=row["created_at"],
                    last_active_at=row["last_active_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    is_active=bool(row["is_active"]),
                )
                sessions.append(ctx)
            return sessions

    def close_session(self, session_id: str) -> bool:
        """Mark a session as inactive. Returns ``True`` if it existed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_id = ? AND is_active = 1",
                (session_id,),
            )
            self._conn.commit()
            self._sessions.pop(session_id, None)
            return cur.rowcount > 0

    def touch_session(self, session_id: str) -> None:
        """Update the last-active timestamp for a session."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            self._conn.commit()
            if session_id in self._sessions:
                self._sessions[session_id].last_active_at = now

    def cleanup_expired(self) -> int:
        """Close sessions that have exceeded ``session_ttl``. Returns count closed."""
        cutoff = time.time() - self._session_ttl
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE is_active = 1 AND last_active_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            # Evict from cache
            expired = [sid for sid, s in self._sessions.items()
                       if s.last_active_at < cutoff]
            for sid in expired:
                del self._sessions[sid]
            return cur.rowcount

    def close(self) -> None:
        """Close the backing database connection."""
        with self._lock:
            self._conn.close()
            self._sessions.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_session(self, session_id: str) -> SessionContext | None:
        import json

        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        ctx = SessionContext(
            session_id=row["session_id"],
            user_id=row["user_id"],
            persona_id=row["persona_id"],
            created_at=row["created_at"],
            last_active_at=row["last_active_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            is_active=bool(row["is_active"]),
        )
        if ctx.is_active:
            self._sessions[session_id] = ctx
        return ctx

    def _enforce_session_limit(self, user_id: str) -> None:
        """If the user already has ``max_sessions_per_user`` active sessions,
        close the oldest one(s)."""
        rows = self._conn.execute(
            "SELECT session_id FROM sessions WHERE user_id = ? AND is_active = 1 "
            "ORDER BY last_active_at ASC",
            (user_id,),
        ).fetchall()

        excess = len(rows) - self._max_sessions_per_user + 1  # +1 for the new one
        if excess > 0:
            for row in rows[:excess]:
                sid = row["session_id"]
                self._conn.execute(
                    "UPDATE sessions SET is_active = 0 WHERE session_id = ?",
                    (sid,),
                )
                self._sessions.pop(sid, None)
                logger.debug("Auto-closed oldest session %s for user %s", sid, user_id)
            self._conn.commit()
