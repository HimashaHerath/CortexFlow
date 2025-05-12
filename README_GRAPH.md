# CortexFlow Enhanced Knowledge Graph

This package provides advanced entity recognition and relation extraction capabilities for building knowledge graphs in CortexFlow.

## Enhanced Entity Recognition

The updated entity recognition system now uses multiple models and techniques:

- **SpaCy** for basic NER functionality
- **Flair NER** for improved recognition with state-of-the-art models
- **SpanBERT** for contextualized entity recognition
- **Entity Linking** to connect entity mentions to canonical entities
- **Fuzzy Matching** for resolving different forms of the same entity

### Features

- Multi-model ensemble approach combining different NER techniques
- Cross-model entity deduplication and overlap resolution
- Entity linking to canonical representations
- Support for entity aliases and fuzzy matching
- Metadata tracking for entity provenance and confidence

## Advanced Relation Extraction

The new `RelationExtractor` class provides comprehensive relation extraction:

- **Dependency Parsing** for accurate subject-verb-object extraction
- **Semantic Role Labeling** for proper argument identification
- **Relation Classification** to categorize relationships into semantic types
- **Pattern-based Rules** for common relationship patterns

### Features

- Improved extraction of subject-verb-object relationships
- Better handling of complex sentences and nested clauses
- Semantic categorization of relationships
- Rule templates for domain-specific relationships
- Deduplication and confidence scoring

## Usage

```python
from cortexflow.graph_store import GraphStore
from cortexflow.config import CortexFlowConfig

# Initialize graph store
config = CortexFlowConfig()
graph = GraphStore(config)

# Extract entities from text
text = "Albert Einstein developed the theory of relativity while working in Berlin."
entities = graph.extract_entities(text)

# Extract relations
relations = graph.extract_relations(text)

# Add an entity with aliases
entity_id = graph.add_entity(
    entity="Albert Einstein",
    entity_type="PERSON",
    metadata={
        "aliases": ["Einstein", "A. Einstein"],
        "birth_year": 1879,
        "death_year": 1955
    }
)

# Add an additional alias
graph.add_entity_alias(entity_id, "Father of Relativity")

# Process text to automatically extract entities and relations
graph.process_text_to_graph(
    text="Marie Curie discovered radium and polonium. She won the Nobel Prize in Physics in 1903.",
    source="Wikipedia"
)
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For optimal performance, install the larger models:

```bash
python -m spacy download en_core_web_lg
```

## Advanced Graph Queries

The enhanced knowledge graph supports advanced query capabilities:

- **Path Queries**: Find connections between entities
- **Weighted Path Queries**: Consider confidence and importance
- **Bidirectional Search**: More efficient path discovery
- **Constrained Paths**: Filter by relationship types

```python
# Find paths between entities
paths = graph.path_query("Albert Einstein", "Theory of Relativity")

# Find weighted paths considering confidence
paths = graph.weighted_path_query(
    "Albert Einstein", 
    "Theory of Relativity",
    importance_weight=0.7,
    confidence_weight=0.3
)
``` 