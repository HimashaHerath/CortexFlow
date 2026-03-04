"""Episodic memory for CortexFlow -- stores and retrieves conversation episodes."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("cortexflow")


@dataclass
class Episode:
    """A discrete conversational episode."""

    id: int | None = None
    session_id: str | None = None
    user_id: str | None = None
    title: str = ""
    summary: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    end_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5


class EpisodicMemoryStore:
    """SQLite-backed episodic memory with FTS5 for full-text search."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                title TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                messages TEXT DEFAULT '[]',
                emotions TEXT DEFAULT '[]',
                topics TEXT DEFAULT '[]',
                start_time TEXT NOT NULL,
                end_time TEXT,
                metadata TEXT DEFAULT '{}',
                importance REAL DEFAULT 0.5
            )
        """)
        # FTS5 virtual table for full-text search on summary and title
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                title, summary, topics, content=episodes, content_rowid=id
            )
        """)
        # Triggers to keep FTS in sync
        self._conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, title, summary, topics)
                VALUES (new.id, new.title, new.summary, new.topics);
            END;
            CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, title, summary, topics)
                VALUES ('delete', old.id, old.title, old.summary, old.topics);
                INSERT INTO episodes_fts(rowid, title, summary, topics)
                VALUES (new.id, new.title, new.summary, new.topics);
            END;
            CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, title, summary, topics)
                VALUES ('delete', old.id, old.title, old.summary, old.topics);
            END;
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)"
        )
        self._conn.commit()

    def save_episode(self, episode: Episode) -> int:
        """Save an episode. Returns the episode ID."""
        import json

        cursor = self._conn.execute(
            """INSERT INTO episodes
               (session_id, user_id, title, summary, messages, emotions, topics, start_time, end_time, metadata, importance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.session_id,
                episode.user_id,
                episode.title,
                episode.summary,
                json.dumps(episode.messages),
                json.dumps(episode.emotions),
                json.dumps(episode.topics)
                if isinstance(episode.topics, list)
                else episode.topics,
                episode.start_time,
                episode.end_time,
                json.dumps(episode.metadata),
                episode.importance,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def recall_episodes(self, query: str, max_results: int = 5) -> list[Episode]:
        """Search episodes using FTS5 full-text search."""
        try:
            rows = self._conn.execute(
                """SELECT e.* FROM episodes e
                   JOIN episodes_fts f ON e.id = f.rowid
                   WHERE episodes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, max_results),
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS fails
            rows = self._conn.execute(
                """SELECT * FROM episodes
                   WHERE summary LIKE ? OR title LIKE ? OR topics LIKE ?
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", max_results),
            ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def get_recent_episodes(
        self, user_id: str | None = None, limit: int = 10
    ) -> list[Episode]:
        """Get recent episodes, optionally filtered by user."""
        query = "SELECT * FROM episodes"
        params: list = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def summarize_session(self, session_id: str) -> Episode | None:
        """Get or create a summary episode for a session."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE session_id = ? ORDER BY start_time ASC",
            (session_id,),
        ).fetchall()
        if not rows:
            return None

        episodes = [self._row_to_episode(row) for row in rows]
        # Merge into summary
        all_messages = []
        all_emotions = set()
        all_topics = set()
        for ep in episodes:
            all_messages.extend(ep.messages)
            all_emotions.update(ep.emotions)
            all_topics.update(ep.topics)

        return Episode(
            session_id=session_id,
            user_id=episodes[0].user_id,
            title=f"Session {session_id} summary",
            summary=f"Session with {len(all_messages)} messages across {len(episodes)} episodes",
            messages=all_messages,
            emotions=list(all_emotions),
            topics=list(all_topics),
            start_time=episodes[0].start_time,
            end_time=episodes[-1].end_time,
            importance=max(ep.importance for ep in episodes),
        )

    def _row_to_episode(self, row) -> Episode:
        import json

        topics_raw = row["topics"]
        if isinstance(topics_raw, str):
            try:
                topics = json.loads(topics_raw)
            except (json.JSONDecodeError, TypeError):
                topics = [topics_raw] if topics_raw else []
        else:
            topics = topics_raw or []

        return Episode(
            id=row["id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            title=row["title"],
            summary=row["summary"],
            messages=json.loads(row["messages"]) if row["messages"] else [],
            emotions=json.loads(row["emotions"]) if row["emotions"] else [],
            topics=topics if isinstance(topics, list) else [topics],
            start_time=row["start_time"],
            end_time=row["end_time"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            importance=row["importance"],
        )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
