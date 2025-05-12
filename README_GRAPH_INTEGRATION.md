# CortexFlow Knowledge Graph Integration

This document describes the enhanced knowledge graph integration capabilities implemented in CortexFlow.

## Overview

The CortexFlow knowledge graph now supports advanced integration mechanisms that enable:

1. Intelligent merging of information from multiple sources
2. Extended schema with relation types and confidence scores
3. Comprehensive metadata tracking for provenance and versioning
4. Automatic taxonomic relationship extraction
5. Conflict detection and resolution strategies
6. Relationship inference based on logical properties

## Enhanced Schema

The database schema has been extended with the following enhancements:

### Entity and Relationship Versioning

- `entity_versions` table tracks changes to entities over time
- `relationship_versions` table tracks changes to relationships
- Version history includes provenance information and change types

### Relation Type Ontology

- `relation_types` table defines properties of different relationship types
- Support for symmetric, transitive, and inverse relationships
- Hierarchical taxonomy levels for relation types

### Metadata Fields

- Provenance tracking (source, timestamp, extraction method)
- Confidence scores for entities and relationships
- Temporal validity periods (start/end dates)

## Knowledge Integration

The `GraphMerger` component provides intelligent integration of new knowledge:

### Entity Merging

```python
# Example: Merge an entity with potential duplicates
merger = GraphMerger(graph_store)
entity_id = merger.merge_entity(
    entity="Artificial Intelligence",
    entity_type="TECHNOLOGY",
    metadata={"aliases": ["AI", "Machine Intelligence"]},
    provenance="wikipedia",
    confidence=0.9
)
```

- Duplicate detection using exact and fuzzy matching
- Metadata consolidation from multiple sources
- Alias management for entity linking

### Relation Integration

```python
# Example: Merge a relation with conflict resolution
merger.merge_relation(
    source_entity="Machine Learning",
    relation_type="is_a",
    target_entity="Artificial Intelligence",
    provenance="textbook",
    confidence=0.95
)
```

- Conflict detection for contradictory relationships
- Confidence-based conflict resolution
- Integration with the relation type ontology

### Text-to-Graph Integration

```python
# Example: Process text and merge extracted information
results = merger.merge_from_text(
    text="Machine Learning is a subfield of Artificial Intelligence that uses statistical methods.",
    source="textbook"
)
print(f"Merged {results['entities']} entities and {results['relations']} relations")
```

- Extracts entities and relations from text
- Automatically merges with existing knowledge
- Tracks provenance of extracted information

## Automatic Relationship Inference

The system can automatically infer new relationships based on existing knowledge:

### Taxonomic Relationship Discovery

```python
# Example: Discover hierarchical relationships
discovered = merger.discover_taxonomic_relationships()
print(f"Discovered {discovered} taxonomic relationships")
```

- Detects hierarchical relationships from entity names and types
- Creates missing type entities when needed
- Applies proper taxonomic relationship types (is_a, instance_of, subclass_of)

### Logical Inference

```python
# Logical inference happens automatically during relation addition
graph_store.add_relation(
    source_entity="Physics",
    relation_type="part_of",
    target_entity="Science"
)
# Will automatically infer: Science contains Physics
```

- Transitive relationship inference
- Symmetric relationship handling
- Inverse relationship generation

## Conflict Resolution

The system can detect and resolve conflicts in the knowledge graph:

```python
# Example: Detect conflicts
conflicts = merger.detect_conflicts()
print(f"Detected {len(conflicts)} conflicts")

# Example: Resolve conflicts
resolved = merger.resolve_conflicts(conflict_resolution='confidence')
print(f"Resolved {resolved} conflicts")
```

- Cycle detection in hierarchical relationships
- Attribute value conflicts
- Resolution strategies:
  - Confidence-based: Keep information with higher confidence
  - Recency-based: Keep more recent information
  - Provenance-based: Prioritize by source reliability

## Performance Considerations

- Entity merging uses optimized fuzzy matching algorithms
- Relationship inference is selective to avoid explosion of inferred facts
- Conflict detection uses targeted queries rather than graph-wide analysis

## Future Enhancements

- Support for probabilistic knowledge representation
- Integration with external knowledge bases via API
- Enhanced domain-specific integration rules
- Collaborative knowledge curation workflows 