"""Temporal fact management for CortexFlow."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("cortexflow")


@dataclass
class TemporalFact:
    """A fact with temporal validity."""
    id: int | None = None
    subject: str = ""
    predicate: str = ""
    object: str = ""
    valid_from: str | None = None  # ISO format
    valid_until: str | None = None  # ISO format, None = still valid
    confidence: float = 1.0
    source: str | None = None
    superseded_by: int | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class TemporalManager:
    """Manages temporal facts with time-based validity."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS temporal_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_until TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                superseded_by INTEGER,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (superseded_by) REFERENCES temporal_facts(id)
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_temporal_subject ON temporal_facts(subject)
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_temporal_validity ON temporal_facts(valid_from, valid_until)
        ''')
        self._conn.commit()

    def add_temporal_fact(self, fact: TemporalFact) -> int:
        """Add a temporal fact. Returns the fact ID."""
        import json
        cursor = self._conn.execute(
            '''INSERT INTO temporal_facts
               (subject, predicate, object, valid_from, valid_until, confidence, source, superseded_by, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (fact.subject, fact.predicate, fact.object, fact.valid_from, fact.valid_until,
             fact.confidence, fact.source, fact.superseded_by, fact.created_at,
             json.dumps(fact.metadata))
        )
        self._conn.commit()
        return cursor.lastrowid

    def update_fact_validity(self, fact_id: int, valid_until: str) -> bool:
        """Update the end validity of a fact."""
        cursor = self._conn.execute(
            'UPDATE temporal_facts SET valid_until = ? WHERE id = ?',
            (valid_until, fact_id)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def supersede_fact(self, old_fact_id: int, new_fact: TemporalFact) -> int:
        """Supersede an existing fact with a new one. Closes old fact's validity and links."""
        now = datetime.utcnow().isoformat()
        # Close old fact
        self._conn.execute(
            'UPDATE temporal_facts SET valid_until = ? WHERE id = ? AND valid_until IS NULL',
            (now, old_fact_id)
        )
        # Add new fact
        new_id = self.add_temporal_fact(new_fact)
        # Link old to new
        self._conn.execute(
            'UPDATE temporal_facts SET superseded_by = ? WHERE id = ?',
            (new_id, old_fact_id)
        )
        self._conn.commit()
        return new_id

    def get_facts_at_time(self, timestamp: str, subject: str | None = None) -> list[TemporalFact]:
        """Get all facts valid at a specific time."""
        query = '''SELECT * FROM temporal_facts
                   WHERE (valid_from IS NULL OR valid_from <= ?)
                   AND (valid_until IS NULL OR valid_until > ?)
                   AND superseded_by IS NULL'''
        params = [timestamp, timestamp]
        if subject:
            query += ' AND subject = ?'
            params.append(subject)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_fact_timeline(self, subject: str, predicate: str | None = None) -> list[TemporalFact]:
        """Get the timeline of facts for a subject, ordered by creation."""
        query = 'SELECT * FROM temporal_facts WHERE subject = ?'
        params: list[str] = [subject]
        if predicate:
            query += ' AND predicate = ?'
            params.append(predicate)
        query += ' ORDER BY created_at ASC'

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def detect_temporal_conflicts(self, subject: str | None = None) -> list[tuple[TemporalFact, TemporalFact]]:
        """Detect overlapping facts that may conflict (same subject+predicate, overlapping validity)."""
        query = '''SELECT * FROM temporal_facts WHERE superseded_by IS NULL'''
        params: list[str] = []
        if subject:
            query += ' AND subject = ?'
            params.append(subject)
        query += ' ORDER BY subject, predicate, valid_from'

        rows = self._conn.execute(query, params).fetchall()
        facts = [self._row_to_fact(row) for row in rows]

        conflicts: list[tuple[TemporalFact, TemporalFact]] = []
        for i, f1 in enumerate(facts):
            for f2 in facts[i+1:]:
                if f1.subject == f2.subject and f1.predicate == f2.predicate:
                    if self._overlaps(f1, f2):
                        conflicts.append((f1, f2))
        return conflicts

    def _overlaps(self, f1: TemporalFact, f2: TemporalFact) -> bool:
        """Check if two temporal facts have overlapping validity periods."""
        # If either has no start, assume infinite past
        start1 = f1.valid_from or "0000-01-01"
        start2 = f2.valid_from or "0000-01-01"
        end1 = f1.valid_until or "9999-12-31"
        end2 = f2.valid_until or "9999-12-31"
        return start1 < end2 and start2 < end1

    def _row_to_fact(self, row) -> TemporalFact:
        import json
        return TemporalFact(
            id=row['id'],
            subject=row['subject'],
            predicate=row['predicate'],
            object=row['object'],
            valid_from=row['valid_from'],
            valid_until=row['valid_until'],
            confidence=row['confidence'],
            source=row['source'],
            superseded_by=row['superseded_by'],
            created_at=row['created_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
        )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
