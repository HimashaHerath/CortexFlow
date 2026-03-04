"""
Persistent user metadata storage for CortexFlow.

Provides a simple SQLite-backed store for per-user metadata so that
the framework can remember user attributes across sessions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger("cortexflow")


class UserStore:
    """SQLite-backed key/value store for per-user metadata.

    Thread-safe: every public method acquires ``_lock``.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_user(
        self,
        user_id: str,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or update a user record. Returns the stored dict."""
        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self._lock:
            self._conn.execute(
                """INSERT INTO users (user_id, display_name, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       display_name = COALESCE(excluded.display_name, display_name),
                       metadata = excluded.metadata,
                       updated_at = excluded.updated_at""",
                (user_id, display_name, meta_json, now, now),
            )
            self._conn.commit()
        return self.get_user(user_id)  # type: ignore[return-value]

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Return the user record as a dict, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "display_name": row["display_name"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_user_metadata(self, user_id: str, metadata: dict[str, Any]) -> bool:
        """Merge *metadata* into the user's existing metadata. Returns ``True`` if the user existed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is None:
                return False
            existing = json.loads(row["metadata"]) if row["metadata"] else {}
            existing.update(metadata)
            self._conn.execute(
                "UPDATE users SET metadata = ?, updated_at = ? WHERE user_id = ?",
                (json.dumps(existing), time.time(), user_id),
            )
            self._conn.commit()
            return True

    def delete_user(self, user_id: str) -> bool:
        """Delete the user record. Returns ``True`` if it existed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_users(self) -> list[dict[str, Any]]:
        """Return all user records."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM users ORDER BY created_at"
            ).fetchall()
        return [
            {
                "user_id": r["user_id"],
                "display_name": r["display_name"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
