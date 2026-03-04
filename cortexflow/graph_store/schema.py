"""
Database schema management for the graph store.

Provides DDL validation, schema creation, and migration logic
used by :class:`GraphStore` during initialization.
"""
from __future__ import annotations

import logging
import re
import sqlite3

# ---------------------------------------------------------------------------
# DDL allowlists and validation
# ---------------------------------------------------------------------------

# Tables that may be referenced in ALTER TABLE statements
VALID_TABLE_NAMES = {"graph_relationships", "graph_entities"}

# Pattern for acceptable column type declarations
VALID_COL_TYPE_PATTERN = re.compile(r'^[A-Z]+(\s+DEFAULT\s+[\w.]+)?$')


def validate_ddl_identifier(value: str, allowed: set) -> None:
    """Validate a DDL identifier against an allowlist.

    Args:
        value: The identifier to validate (table name or column name).
        allowed: Set of permitted values.

    Raises:
        ValueError: If *value* is not in *allowed*.
    """
    if value not in allowed:
        raise ValueError(
            f"DDL identifier {value!r} is not in the allowlist: {allowed}"
        )


# ---------------------------------------------------------------------------
# Schema initialisation helpers
# ---------------------------------------------------------------------------

def ensure_schema(cursor: sqlite3.Cursor) -> None:
    """Create all required tables and insert base relation types.

    This is called once during :meth:`GraphStore.__init__` and is safe to
    call repeatedly (all statements use ``CREATE TABLE IF NOT EXISTS``).
    """
    # Create entities table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS graph_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL,
        entity_type TEXT,
        metadata TEXT,
        embedding BLOB,
        timestamp REAL,
        provenance TEXT,
        confidence REAL DEFAULT 0.8,
        temporal_start TEXT,
        temporal_end TEXT,
        extraction_method TEXT,
        version INTEGER DEFAULT 1,
        last_updated REAL,
        UNIQUE(entity)
    )
    ''')

    # Create relation_types table for proper relation type ontology
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS relation_types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        parent_type TEXT,
        description TEXT,
        symmetric BOOLEAN DEFAULT 0,
        transitive BOOLEAN DEFAULT 0,
        inverse_relation TEXT,
        taxonomy_level INTEGER DEFAULT 0,
        metadata TEXT,
        UNIQUE(name)
    )
    ''')

    # Create relationships table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS graph_relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        relation_type TEXT NOT NULL,
        weight REAL,
        metadata TEXT,
        timestamp REAL,
        provenance TEXT,
        confidence REAL DEFAULT 0.5,
        temporal_start TEXT,
        temporal_end TEXT,
        extraction_method TEXT,
        version INTEGER DEFAULT 1,
        last_updated REAL,
        FOREIGN KEY (source_id) REFERENCES graph_entities (id),
        FOREIGN KEY (target_id) REFERENCES graph_entities (id),
        FOREIGN KEY (relation_type) REFERENCES relation_types (name),
        UNIQUE(source_id, target_id, relation_type)
    )
    ''')

    # Create entity_versions table for tracking entity changes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entity_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_id INTEGER NOT NULL,
        entity TEXT NOT NULL,
        entity_type TEXT,
        metadata TEXT,
        provenance TEXT,
        confidence REAL,
        temporal_start TEXT,
        temporal_end TEXT,
        extraction_method TEXT,
        version INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        change_type TEXT NOT NULL,
        changed_by TEXT,
        FOREIGN KEY (entity_id) REFERENCES graph_entities (id)
    )
    ''')

    # Create relationship_versions table for tracking relationship changes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS relationship_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        relationship_id INTEGER NOT NULL,
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        relation_type TEXT NOT NULL,
        weight REAL,
        metadata TEXT,
        provenance TEXT,
        confidence REAL,
        temporal_start TEXT,
        temporal_end TEXT,
        extraction_method TEXT,
        version INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        change_type TEXT NOT NULL,
        changed_by TEXT,
        FOREIGN KEY (relationship_id) REFERENCES graph_relationships (id),
        FOREIGN KEY (source_id) REFERENCES graph_entities (id),
        FOREIGN KEY (target_id) REFERENCES graph_entities (id)
    )
    ''')

    # Create table for n-ary relationships
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nary_relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        relation_type TEXT NOT NULL,
        metadata TEXT,
        provenance TEXT,
        confidence REAL,
        extraction_method TEXT,
        version INTEGER DEFAULT 1,
        timestamp REAL,
        last_updated REAL,
        FOREIGN KEY (relation_type) REFERENCES relation_types (name)
    )
    ''')

    # Create table for n-ary relationship participants
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nary_participants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        relationship_id INTEGER NOT NULL,
        entity_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        metadata TEXT,
        timestamp REAL,
        FOREIGN KEY (relationship_id) REFERENCES nary_relationships (id),
        FOREIGN KEY (entity_id) REFERENCES graph_entities (id),
        UNIQUE(relationship_id, entity_id, role)
    )
    ''')

    # Insert basic relation types if not exist
    basic_relation_types = [
        ('is_a', None, 'Taxonomic relationship', 0, 0, None, 1),
        ('part_of', None, 'Meronymic relationship', 0, 1, 'contains', 1),
        ('located_in', None, 'Spatial relationship', 0, 1, 'contains', 1),
        ('has_property', None, 'Attributional relationship', 0, 0, 'is_property_of', 1),
        ('causes', None, 'Causal relationship', 0, 0, 'caused_by', 1),
        ('related_to', None, 'Generic relationship', 1, 0, 'related_to', 0),
        ('same_as', None, 'Identity relationship', 1, 1, 'same_as', 1),
        ('temporal_before', None, 'Temporal relationship', 0, 1, 'temporal_after', 1),
        ('temporal_after', None, 'Temporal relationship', 0, 1, 'temporal_before', 1),
        ('contains', None, 'Containment relationship', 0, 0, 'part_of', 1),
        ('created_by', None, 'Creative relationship', 0, 0, 'created', 1),
        ('instance_of', 'is_a', 'Instance relationship', 0, 0, 'has_instance', 2),
        ('subclass_of', 'is_a', 'Subclass relationship', 0, 1, 'has_subclass', 2)
    ]

    for relation in basic_relation_types:
        cursor.execute('''
            INSERT OR IGNORE INTO relation_types
            (name, parent_type, description, symmetric, transitive, inverse_relation, taxonomy_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', relation)


def add_metadata_columns(cursor: sqlite3.Cursor) -> None:
    """Migrate existing tables to add columns that may be missing.

    This MUST run before index creation so indexes on new columns succeed.
    It is safe to call repeatedly -- missing columns are detected via
    ``PRAGMA table_info``.
    """
    try:
        for table, new_columns in [
            ("graph_relationships", [
                ("extraction_method", "TEXT"),
                ("version", "INTEGER DEFAULT 1"),
                ("last_updated", "REAL"),
                ("provenance", "TEXT"),
                ("confidence", "REAL DEFAULT 0.5"),
                ("temporal_start", "TEXT"),
                ("temporal_end", "TEXT"),
            ]),
            ("graph_entities", [
                ("extraction_method", "TEXT"),
                ("version", "INTEGER DEFAULT 1"),
                ("last_updated", "REAL"),
                ("embedding", "BLOB"),
                ("provenance", "TEXT"),
                ("confidence", "REAL DEFAULT 0.8"),
                ("temporal_start", "TEXT"),
                ("temporal_end", "TEXT"),
            ]),
        ]:
            # Validate table name against allowlist
            validate_ddl_identifier(table, VALID_TABLE_NAMES)

            cursor.execute(f"PRAGMA table_info({table})")
            existing = {info[1] for info in cursor.fetchall()}
            for col_name, col_type in new_columns:
                # Validate column name: only alphanumeric and underscores
                if not re.fullmatch(r'[A-Za-z_]\w*', col_name):
                    raise ValueError(
                        f"Invalid DDL column name: {col_name!r}"
                    )
                # Validate column type against allowed pattern
                if not VALID_COL_TYPE_PATTERN.match(col_type):
                    raise ValueError(
                        f"Invalid DDL column type: {col_type!r}"
                    )
                if col_name not in existing:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
    except sqlite3.OperationalError as e:
        logging.error(f"Error adding metadata columns: {e}")


def create_indexes(cursor: sqlite3.Cursor) -> None:
    """Create all indexes for faster lookups.

    Must be called *after* :func:`add_metadata_columns` so that the
    columns referenced by the indexes already exist.
    """
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON graph_entities(entity)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON graph_entities(entity_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version ON graph_entities(version)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_provenance ON graph_entities(provenance)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_extraction ON graph_entities(extraction_method)')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_type_name ON relation_types(name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_type_parent ON relation_types(parent_type)')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON graph_relationships(source_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON graph_relationships(target_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation ON graph_relationships(relation_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_version ON graph_relationships(version)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_provenance ON graph_relationships(provenance)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_extraction ON graph_relationships(extraction_method)')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version_entity ON entity_versions(entity_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version_number ON entity_versions(version)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_version_type ON entity_versions(change_type)')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_version_rel ON relationship_versions(relationship_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_version_number ON relationship_versions(version)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_version_type ON relationship_versions(change_type)')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_type ON nary_relationships(relation_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_extraction ON nary_relationships(extraction_method)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_version ON nary_relationships(version)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_participant ON nary_participants(relationship_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nary_entity ON nary_participants(entity_id)')
