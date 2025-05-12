# CortexFlow

![CortexFlow Logo](docs/source/_static/logo.png)

CortexFlow is a multi-tier memory optimization system for LLMs that implements a cognitive-inspired architecture to maximize context utilization and enable complex reasoning.

[![PyPI version](https://badge.fury.io/py/cortexflow.svg)](https://badge.fury.io/py/cortexflow)
[![Documentation Status](https://readthedocs.org/projects/cortexflow/badge/?version=latest)](https://cortexflow.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CortexFlow dynamically manages context information, retaining important elements while discarding less relevant ones to maximize effective context window utilization without increasing memory requirements. The system uses a multi-tier memory approach:

- **Active Tier**: Holds the most recent and important context
- **Working Tier**: Medium-term storage with moderate compression
- **Archive Tier**: Long-term storage with higher compression
- **Knowledge Store**: Persistent storage for important facts beyond the conversation

## Features

- Multi-tier memory management optimizes token usage
- Importance-based information retention using rule-based, ML, and LLM classification
- Progressive context compression with extractive and abstractive summarization
- Knowledge store for long-term information persistence
- **Dynamic memory tier weighting** for adaptive token allocation
- **Vector-based knowledge retrieval** for semantic search
- **Advanced retrieval techniques** including GraphRAG for complex multi-hop queries
- **Chain of Agents** for collaborative multi-agent reasoning over complex queries
- **Self-Reflection** for verifying knowledge relevance and response consistency
- **Enhanced entity and relation extraction** with semantic role labeling and coreference resolution

## Installation

```bash
pip install cortexflow
```

For the graph-based knowledge features:
```bash
pip install "cortexflow[graph]"
```

For the full package with all dependencies (including enhanced NLP capabilities):
```bash
pip install "cortexflow[all]"
```

## Quick Start

```python
from cortexflow import CortexFlowManager, CortexFlowConfig

# Configure with custom settings
config = CortexFlowConfig(
    active_token_limit=2000,
    working_token_limit=4000,
    archive_token_limit=6000,
    default_model="llama3"  # Use your preferred Ollama model
)

# Create the context manager
context_manager = CortexFlowManager(config)

# Add messages to the context
context_manager.add_message("system", "You are a helpful AI assistant.")
context_manager.add_message("user", "What is the capital of France?")
response = context_manager.generate_response()
print(f"Assistant: {response}")

# Explicitly store important information
context_manager.remember_knowledge("The user's name is Alice and she lives in Boston.")

# Use the advanced knowledge graph features
config = CortexFlowConfig(
    use_graph_rag=True,
    knowledge_store_path="knowledge.db"
)
manager = CortexFlowManager(config)

# Automatically extract entities and relations from text
document = """
Albert Einstein developed the theory of relativity in the early 20th century.
He was born in Germany but later moved to the United States where he worked at Princeton University.
His famous equation E=mc² relates energy and mass.
"""
manager.knowledge_store.graph_store.process_text_to_graph(document)

# Query the knowledge graph
paths = manager.knowledge_store.graph_store.path_query(
    start_entity="Albert Einstein", 
    end_entity="Princeton University", 
    max_hops=2
)

# Clean up when done
context_manager.close()
```

## Advanced NLP Features

CortexFlow provides enhanced entity and relation extraction capabilities:

- **Semantic Role Labeling**: Extracts high-quality relationships based on semantic roles
- **Coreference Resolution**: Resolves pronouns and other references to their entities
- **Domain-Specific Entity Recognition**: Identifies entities in specialized domains
- **Advanced Dependency Parsing**: Extracts complex relationships between entities
- **Improved Triple Extraction**: Creates more accurate subject-predicate-object triples

These features enable building more robust knowledge graphs for improved reasoning.

## Advanced Graph Traversal

CortexFlow now includes sophisticated knowledge graph traversal capabilities:

- **Weighted Path Algorithms**: Find paths that prioritize relation importance and confidence scores
- **Bidirectional Search**: Efficiently find connections between entities in large graphs
- **Constrained Path Finding**: Search for paths with specific relation type requirements
- **Graph Contraction**: Handle large knowledge graphs efficiently through smart compression techniques
- **Hierarchical Graph Abstraction**: Create simplified representations of complex graphs for faster traversal

Example usage:

```python
from cortexflow import CortexFlowManager, CortexFlowConfig

# Configure with graph features enabled
config = CortexFlowConfig(
    use_graph_rag=True,
    knowledge_store_path="knowledge.db"
)
manager = CortexFlowManager(config)

# Add some knowledge to the graph
manager.knowledge_store.remember_knowledge(
    "Marie Curie discovered radium and polonium. She worked at the University of Paris."
)

# Find weighted paths considering relation importance and confidence
paths = manager.knowledge_store.graph_store.weighted_path_query(
    start_entity="Marie Curie", 
    end_entity="University of Paris",
    importance_weight=0.7,
    confidence_weight=0.3
)

# Find paths through specific relation types
paths = manager.knowledge_store.graph_store.constrained_path_search(
    start_entity="Marie Curie",
    end_entity="radium",
    allowed_relations=["discovered", "researched", "studied"]
)

# Use bidirectional search for efficiency in large graphs
paths = manager.knowledge_store.graph_store.bidirectional_search(
    start_entity="Marie Curie",
    end_entity="polonium"
)

# Contract the graph to handle large knowledge graphs efficiently
stats = manager.knowledge_store.graph_store.contract_graph(
    min_edge_weight=0.3,
    min_confidence=0.4,
    combine_parallel_edges=True
)

# Create hierarchical abstraction for large graph traversal
abstraction = manager.knowledge_store.graph_store.create_graph_abstraction()

# Use the abstracted graph for efficient path finding
paths = manager.knowledge_store.graph_store.path_query_with_abstraction(
    start_entity="Marie Curie",
    end_entity="University of Paris"
)
```

## Documentation

For full documentation, visit [cortexflow.readthedocs.io](https://cortexflow.readthedocs.io).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Note for AdaptiveContext Users

CortexFlow is the new name for the project previously known as AdaptiveContext. See the [migration guide](https://cortexflow.readthedocs.io/en/latest/guides/migration.html) for details on transitioning.