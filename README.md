# CortexFlow

CortexFlow is a multi-tier memory optimization system for LLMs that implements a cognitive-inspired architecture to maximize context utilization and enable complex reasoning.

[![PyPI version](https://badge.fury.io/py/cortexflow-llm.svg)](https://pypi.org/project/cortexflow-llm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/cortexflow/badge/?version=latest)](https://cortexflow.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

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
- **Personal fact extraction** with regex-based detection, entity-aware importance scoring, and dual-write to knowledge store for deep memory recall
- **Dynamic memory tier weighting** for adaptive token allocation
- **Vector-based knowledge retrieval** for semantic search
- **Advanced retrieval techniques** including GraphRAG for complex multi-hop queries
- **Chain of Agents** for collaborative multi-agent reasoning over complex queries
- **Self-Reflection** for verifying knowledge relevance and response consistency
- **Enhanced entity and relation extraction** with semantic role labeling and dependency parsing
- **Logical reasoning capabilities** with forward chaining, backward chaining, and abductive reasoning
- **Uncertainty handling and belief revision** for managing contradictions and incomplete information
- **Performance optimization features** for scalable knowledge graph operations:
  - Graph partitioning for efficient storage and retrieval
  - Multi-hop indexing for faster traversal
  - Query planning for optimizing reasoning paths
  - Caching for common reasoning patterns
- **MCP Server** -- expose CortexFlow as a Model Context Protocol tool server with 9 tools over stdio transport
- **Framework integrations** -- first-class adapters for LangChain (ChatMessageHistory, Retriever, Memory) and CrewAI (StorageBackend)
- **Pluggable vector store backends** -- swap between SQLite (default), ChromaDB, and Qdrant
- **Temporal facts** -- time-aware fact management with validity windows, superseding, and conflict detection
- **Episodic memory** -- session-based episode storage with FTS5 full-text search
- **Event system** -- thread-safe pub/sub event bus with 12 event types for middleware and hooks
- **Safety pipeline** -- content filtering with PII detection, boundary enforcement, and custom rules

## Requirements

- Python 3.10 or higher

## Installation

```bash
pip install cortexflow-llm
```

For the graph-based knowledge features:
```bash
pip install "cortexflow-llm[graph]"
```

For LiteLLM multi-provider support (OpenAI, Anthropic, Cohere, Azure, Bedrock, etc.):
```bash
pip install "cortexflow-llm[litellm]"
```

For the MCP server (Model Context Protocol):
```bash
pip install "cortexflow-llm[mcp]"
```

For LangChain integration:
```bash
pip install "cortexflow-llm[langchain]"
```

For CrewAI integration:
```bash
pip install "cortexflow-llm[crewai]"
```

For ChromaDB vector backend:
```bash
pip install "cortexflow-llm[chromadb]"
```

For Qdrant vector backend:
```bash
pip install "cortexflow-llm[qdrant]"
```

For the full package with all dependencies:
```bash
pip install "cortexflow-llm[all]"
```

## Quick Start

```python
from cortexflow import CortexFlowManager, CortexFlowConfig, ConfigBuilder
from cortexflow import MemoryConfig, LLMConfig, KnowledgeStoreConfig, GraphRagConfig

# Configure with nested dataclasses
config = CortexFlowConfig(
    memory=MemoryConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000,
    ),
    llm=LLMConfig(default_model="llama3"),
)

# Or use the builder pattern
config = (ConfigBuilder()
    .with_memory(active_token_limit=2000, working_token_limit=4000)
    .with_llm(default_model="llama3")
    .build())

# Create the context manager
context_manager = CortexFlowManager(config)

# Add messages to the context
context_manager.add_message("system", "You are a helpful AI assistant.")
context_manager.add_message("user", "What is the capital of France?")
response = context_manager.generate_response()
print(f"Assistant: {response}")

# Explicitly store important information
context_manager.add_knowledge("The user's name is Alice and she lives in Boston.")

# Use the advanced knowledge graph features
config = CortexFlowConfig(
    graph_rag=GraphRagConfig(use_graph_rag=True),
    knowledge_store=KnowledgeStoreConfig(knowledge_store_path="knowledge.db"),
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
    graph_rag=GraphRagConfig(use_graph_rag=True),
    knowledge_store=KnowledgeStoreConfig(knowledge_store_path="knowledge.db"),
)
manager = CortexFlowManager(config)

# Add some knowledge to the graph
manager.knowledge_store.add_knowledge(
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

## Performance Optimization for Knowledge Graphs

CortexFlow provides advanced performance optimization capabilities for scalable knowledge graph operations:

- **Graph Partitioning**: Divides large graphs into manageable subgraphs based on connectivity patterns
- **Multi-hop Indexing**: Pre-computes and stores paths between entities for faster traversal
- **Query Planning**: Generates optimized execution plans for knowledge graph operations
- **Reasoning Pattern Caching**: Stores and reuses common reasoning patterns to avoid redundant computation

Example usage:

```python
from cortexflow import CortexFlowManager, CortexFlowConfig

# Configure with performance optimization enabled
config = (ConfigBuilder()
    .with_graph_rag(use_graph_rag=True, use_graph_partitioning=True,
                    use_multihop_indexing=True, graph_partition_method="louvain",
                    max_indexed_hops=2)
    .with_performance(use_performance_optimization=True)
    .with_knowledge_store(knowledge_store_path="knowledge.db")
    .build())
manager = CortexFlowManager(config)

# Partition the knowledge graph for efficient operations
partition_result = manager.partition_graph(method="louvain")
print(f"Created {partition_result['partitions']} partitions")

# Create indexes for multi-hop paths (up to 2 hops)
index_result = manager.create_hop_indexes(max_hops=2)
print(f"Created {index_result['indexes_created']} indexes")

# Use the query planning system for optimized path queries
query_plan = manager.optimize_path_query(
    start_entity="Albert Einstein",
    end_entity="Princeton University",
    max_hops=3,
    relation_constraints=["affiliated_with", "worked_at"]
)
print(f"Query plan using {query_plan['plan']['strategy']} strategy")

# Cache common reasoning patterns for reuse
manager.cache_reasoning_pattern(
    pattern_key="scientist_institution|Albert Einstein",
    pattern_result={"institution": "Princeton University", "role": "Professor"}
)

# Get performance statistics
stats = manager.get_performance_stats()
print(f"Cache hit rate: {stats['caching']['hit_rate']}%")
```

## Logical Reasoning Mechanisms

CortexFlow now includes formal reasoning capabilities over the knowledge graph:

- **Inference Engine**: Apply logical rules to derive new knowledge and answer complex queries
- **Backward Chaining**: Answer "why" questions by tracing logical dependencies between facts
- **Forward Chaining**: Discover novel implications by applying rules to existing knowledge
- **Abductive Reasoning**: Generate hypotheses to explain observations when information is incomplete

Example usage:

```python
from cortexflow import CortexFlowManager, CortexFlowConfig

# Configure with inference engine enabled
config = CortexFlowConfig(
    graph_rag=GraphRagConfig(use_graph_rag=True),
    inference=InferenceConfig(use_inference_engine=True),
    knowledge_store=KnowledgeStoreConfig(knowledge_store_path="knowledge.db"),
)
manager = CortexFlowManager(config)

# Add facts to the knowledge graph
knowledge = manager.knowledge_store
graph = knowledge.graph_store

# Add some entities and facts
graph.add_entity("bird", "category")
graph.add_entity("eagle", "animal_species")
graph.add_relation("eagle", "is_a", "bird")

# Add a logical rule
manager.add_logical_rule(
    name="birds_can_fly",
    premise_patterns=[
        {"source": "?X", "relation": "is_a", "target": "bird"}
    ],
    conclusion_pattern={"source": "?X", "relation": "can_fly", "target": "true"},
    confidence=0.8
)

# Use backward chaining to answer why questions
explanation = manager.answer_why_question("Why can an eagle fly?")
for step in explanation:
    print(f"[{step['type']}] {step['message']}")

# Use forward chaining to discover novel implications
new_facts = manager.generate_novel_implications()
for fact in new_facts:
    print(f"{fact['source']} {fact['relation']} {fact['target']}")

# Use abductive reasoning to generate hypotheses
hypotheses = manager.generate_hypotheses("Eagles have hollow bones")
for hypothesis in hypotheses:
    print(f"{hypothesis['text']} (Confidence: {hypothesis['confidence']})")
```

## Uncertainty Handling and Belief Revision

CortexFlow now provides mechanisms to handle uncertainty and contradictions:

- **Belief Revision**: Update beliefs when new contradictory information arrives
- **Explicit Uncertainty Representation**: Track confidence scores and probability distributions
- **Conflict Resolution**: Resolve conflicts based on source reliability and recency
- **Reasoning with Incomplete Information**: Provide reasonable answers even with partial knowledge

Example usage:

```python
from cortexflow import CortexFlowManager, CortexFlowConfig

# Configure with uncertainty handling enabled
config = CortexFlowConfig(
    uncertainty=UncertaintyConfig(
        use_uncertainty_handling=True,
        auto_detect_contradictions=True,
        default_contradiction_strategy="weighted",
    ),
    knowledge_store=KnowledgeStoreConfig(knowledge_store_path="knowledge.db"),
)
manager = CortexFlowManager(config)

# Add knowledge with confidence scores
manager.add_knowledge(
    "Mount Everest is 8,848 meters tall.",
    source="geography_textbook",
    confidence=0.9
)

# Add contradictory information
manager.add_knowledge(
    "Mount Everest is 8,849 meters tall.",
    source="recent_survey",
    confidence=0.85
)

# Contradictions are automatically detected and resolved based on the strategy

# Set source reliability for conflict resolution
manager.update_source_reliability("scientific_journal", 0.95)
manager.update_source_reliability("social_media", 0.3)

# Get the belief revision history
revisions = manager.get_belief_revision_history()

# Reason with incomplete information
query = {
    "question": "What is the capital of France and its population?",
    "required_fields": ["capital", "population"]
}
result = manager.reason_with_incomplete_information(query, available_knowledge)
```

For more details, see [Uncertainty Handling Documentation](docs/uncertainty_handling.md).

## Personal Fact Extraction (Deep Memory Recall)

CortexFlow can automatically detect and preserve personal facts shared during conversation, ensuring they survive memory tier compression and remain retrievable even after hundreds of filler messages.

Three defense layers work together:

1. **Importance scoring** — Personal facts ("My name is Alice") score 8.0, above all regular content (3.0-7.0) and below system messages (9.0). This keeps facts in the active tier longer and triggers the compression preservation guard.
2. **Entity-preserving compression** — During extractive summarization, sentences containing personal facts get the maximum score (10.0, same as code blocks) and are never dropped.
3. **Dual-write to knowledge store** — Detected facts are automatically stored in the knowledge store on ingestion. Even after tier compression destroys the original message, `generate_response()` retrieves the fact via BM25 search.

**Detected patterns:** names, occupations, locations, preferences, possessions, age statements.

```python
from cortexflow import ConfigBuilder, CortexFlowManager

config = (ConfigBuilder()
    .with_memory(active_token_limit=1024, working_token_limit=4096)
    .with_fact_extraction(enabled=True)
    .with_llm(default_model="llama3")
    .build())

manager = CortexFlowManager(config)

# Personal facts are automatically detected and preserved
manager.add_message("user", "My name is Alice.")
manager.add_message("user", "I work at NASA as a flight engineer.")

# ... hundreds of filler messages later ...

manager.add_message("user", "What is my name?")
response = manager.generate_response()
# Response will correctly recall "Alice"
```

Enable via config: `use_fact_extraction=True` in `MemoryConfig`, or `.with_fact_extraction()` on the builder. Disabled by default for backward compatibility.

## New in v1.0

### MCP Server (Model Context Protocol)

CortexFlow exposes itself as an MCP tool server with 9 tools, allowing any MCP-compatible client (Claude Desktop, IDEs, custom agents) to use CortexFlow memory, knowledge, and companion AI features over stdio transport.

**Available tools:** `add_memory`, `search_memory`, `add_knowledge`, `get_conversation_context`, `get_user_profile`, `get_emotional_state`, `get_relationship_state`, `manage_persona`, `create_session`.

Run the server directly:

```bash
# Via console-script entry point
cortexflow-mcp

# Or as a Python module
python -m cortexflow.mcp_server
```

Configure in Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cortexflow": {
      "command": "cortexflow-mcp",
      "env": {
        "CORTEXFLOW_DB_PATH": "/path/to/knowledge.db",
        "CORTEXFLOW_SESSIONS": "true",
        "CORTEXFLOW_EMOTIONS": "true",
        "CORTEXFLOW_FACT_EXTRACTION": "true"
      }
    }
  }
}
```

The server is configured entirely through environment variables (`CORTEXFLOW_DB_PATH`, `CORTEXFLOW_SESSIONS`, `CORTEXFLOW_EMOTIONS`, `CORTEXFLOW_PERSONAS`, `CORTEXFLOW_RELATIONSHIPS`, `CORTEXFLOW_FACT_EXTRACTION`, `CORTEXFLOW_VERBOSE`). See `examples/mcp_quickstart.py` for a full walkthrough.

### Framework Integrations

CortexFlow provides first-class adapters for popular LLM frameworks. Import only the integration you need -- dependencies are optional and lazy-loaded.

#### LangChain

```python
from cortexflow import CortexFlowManager, ConfigBuilder
from cortexflow.integrations.langchain import (
    CortexFlowChatMessageHistory,
    CortexFlowRetriever,
    CortexFlowMemory,
)

manager = CortexFlowManager(ConfigBuilder().with_memory().build())

# Use as LangChain chat history
history = CortexFlowChatMessageHistory(manager=manager)
history.add_user_message("My name is Alice.")
history.add_ai_message("Nice to meet you, Alice!")

# Use as a LangChain retriever in RAG chains
retriever = CortexFlowRetriever(manager=manager, search_kwargs={"max_results": 5})
docs = retriever.invoke("What is the user's name?")

# Or bundle both into a single memory object
memory = CortexFlowMemory(manager=manager)
```

See `examples/langchain_integration.py` for a full example.

#### CrewAI

```python
from cortexflow import CortexFlowManager, ConfigBuilder
from cortexflow.integrations.crewai import CortexFlowCrewStorage

manager = CortexFlowManager(ConfigBuilder().with_memory().build())
storage = CortexFlowCrewStorage(manager=manager)

# CrewAI storage protocol
storage.save(key="user_pref", value="Prefers Python over JavaScript")
results = storage.search(query="programming language preference")
```

See `examples/crewai_integration.py` for a full example.

### Pluggable Vector Store Backends

Swap the default SQLite vector store for ChromaDB or Qdrant with a single builder call. All backends implement the same `VectorStoreBackend` interface, so the rest of CortexFlow works identically regardless of choice.

```python
from cortexflow import ConfigBuilder, CortexFlowManager

# ChromaDB (local or client/server)
config = (ConfigBuilder()
    .with_vector_store(backend="chromadb", chromadb_path="/tmp/cortexflow_chroma")
    .with_memory(active_token_limit=2048)
    .build())

# Qdrant (local or cloud)
config = (ConfigBuilder()
    .with_vector_store(backend="qdrant", qdrant_url="http://localhost:6333")
    .with_memory(active_token_limit=2048)
    .build())

# SQLite (default -- no extra config needed)
config = (ConfigBuilder()
    .with_vector_store(backend="sqlite")
    .build())

manager = CortexFlowManager(config)
```

See `examples/vector_backends.py` for a full example.

### Temporal Facts

Time-aware fact management with validity windows. Facts can be superseded when new information arrives, and conflicts are automatically detected.

```python
from cortexflow import ConfigBuilder, CortexFlowManager
from cortexflow.temporal import TemporalFact, TemporalManager

config = (ConfigBuilder()
    .with_temporal(temporal_db_path="temporal.db")
    .build())

manager = CortexFlowManager(config)
temporal = TemporalManager(db_path="temporal.db")

# Add a fact with a validity window
fact_id = temporal.add_temporal_fact(TemporalFact(
    subject="Alice",
    predicate="lives_in",
    object="Boston",
    valid_from="2020-01-01",
    confidence=0.95,
    source="user_statement",
))

# Supersede when information changes
new_id = temporal.add_temporal_fact(TemporalFact(
    subject="Alice",
    predicate="lives_in",
    object="San Francisco",
    valid_from="2025-06-01",
    confidence=0.9,
    source="user_statement",
))
temporal.update_fact_validity(fact_id, valid_until="2025-05-31")
```

See `examples/temporal_facts.py` for a full example.

### Episodic Memory

Session-based episode storage with FTS5 full-text search. Automatically captures conversation episodes with associated emotions, topics, and importance scores.

```python
from cortexflow.episodic_memory import EpisodicMemoryStore, Episode

store = EpisodicMemoryStore(db_path="episodes.db")

# Store an episode
episode_id = store.save_episode(Episode(
    session_id="session-001",
    user_id="alice",
    title="Planning a trip to Japan",
    summary="Discussed travel plans, budget, and itinerary for a 2-week trip.",
    topics=["travel", "japan", "budget"],
    emotions=["excited", "curious"],
    importance=0.8,
))

# Full-text search across episodes
results = store.search_episodes("japan travel plans")
```

### Event System

A thread-safe pub/sub event bus with 12 event types for building middleware, logging, analytics, and custom hooks.

```python
from cortexflow import ConfigBuilder, CortexFlowManager
from cortexflow.events import EventBus, EventType

config = (ConfigBuilder()
    .with_events()
    .with_memory(active_token_limit=2048)
    .build())

bus = EventBus()

# Subscribe to specific events
@bus.on(EventType.MESSAGE_ADDED)
def on_message(event):
    print(f"New message from {event.data.get('role')}: {event.data.get('content')[:50]}")

@bus.on(EventType.KNOWLEDGE_ADDED)
def on_knowledge(event):
    print(f"Knowledge stored: {event.data.get('text')[:50]}")

# Or subscribe to all events for logging
@bus.on_all
def log_all(event):
    print(f"[{event.type.value}] {event.timestamp}")
```

**Event types:** `MESSAGE_ADDED`, `MEMORY_CLEARED`, `KNOWLEDGE_ADDED`, `KNOWLEDGE_RETRIEVED`, `SESSION_CREATED`, `SESSION_CLOSED`, `RESPONSE_GENERATED`, `EMOTION_DETECTED`, `RELATIONSHIP_STAGE_CHANGED`, `TEMPORAL_FACT_ADDED`, `MANAGER_INITIALIZED`, `MANAGER_CLOSING`.

### Safety Pipeline

Content filtering with PII detection, boundary enforcement, and custom blocked-pattern rules. Disabled by default; enable via the builder.

```python
from cortexflow import ConfigBuilder, CortexFlowManager

config = (ConfigBuilder()
    .with_safety(
        enable_pii_detection=True,
        enable_boundary_enforcement=True,
        custom_blocked_patterns=[r"\b(SSN|social security)\b"],
        block_on_safety_violation=False,  # True = block message entirely
    )
    .with_memory(active_token_limit=2048)
    .build())

manager = CortexFlowManager(config)
```

When `block_on_safety_violation` is `False` (default), flagged content is annotated but still processed. Set to `True` to reject messages that trigger safety rules.

## LLM Backends

CortexFlow supports three LLM backends: **Ollama** (default, local), **Vertex AI / Gemini**, and **LiteLLM** (100+ providers).

### LiteLLM (Multi-Provider)

Use any LLM provider (OpenAI, Anthropic, Cohere, Azure, Bedrock, etc.) through a unified interface:

```bash
pip install "cortexflow-llm[litellm]"
```

```python
from cortexflow.llm_client import LiteLLMClient

# OpenAI
client = LiteLLMClient(default_model="gpt-4o-mini")

# Anthropic
client = LiteLLMClient(default_model="anthropic/claude-sonnet-4-20250514")

# Async support
response = await client.agenerate([{"role": "user", "content": "Hello"}])
```

Set `backend = "litellm"` in your LLM config or `CORTEXFLOW_LLM_BACKEND=litellm` env var.

### Vertex AI / Gemini

```bash
pip install "cortexflow-llm[vertex]"
```

```python
from cortexflow import ConfigBuilder, CortexFlowManager

config = (ConfigBuilder()
    .with_vertex_ai(
        project_id="your-gcp-project",
        location="us-central1",
        default_model="gemini-2.0-flash",
        credentials_path="/path/to/service-account.json",
    )
    .with_memory(active_token_limit=4096)
    .build())

manager = CortexFlowManager(config)
response = manager.generate_response()
```

Vertex AI requires a GCP project with the Vertex AI API enabled and service account credentials with the `cloud-platform` scope.

## Benchmarks

CortexFlow includes two benchmark suites that validate its memory management against live LLM calls (Gemini 2.0 Flash via Vertex AI).

### Comparative Evidence Benchmark — CortexFlow vs. Naive Baseline

An A/B benchmark that runs the **same queries** through both CortexFlow (full tiered-memory pipeline) and a naive last-N-messages baseline using the **same LLM and credentials**. The only difference is memory management.

| # | Scenario | CortexFlow | Naive | Winner |
|---|----------|:----------:|:-----:|:------:|
| 1 | **Deep Memory Recall** — 5 personal facts + 40 filler pairs, recall through tier cascade | **5/5** | 0/5 | **CortexFlow** |
| 2 | **Knowledge-Augmented** — 10 invented facts the LLM can't know from training | 9/10 | 0/10 | **CortexFlow** |
| 3 | **System Instruction Preservation** — sentinel phrase after 60 filler pairs | 3/3 | 1/3 | **CortexFlow** |
| 4 | **Conversational Coherence** — framework/language recall across scattered context | 3/3 | 3/3 | Tie |
| 5 | **Token Efficiency** — quality per token (quality / 1k tokens) | varies | varies | varies |

**Overall: CortexFlow 3 wins, Naive 1, Tie 1. LLM-as-Judge mean score: CortexFlow 9.6/10 vs. Naive 5.7/10.**

Key findings:
- **Deep memory recall** — With personal fact extraction enabled, CortexFlow recalls **5/5 personal facts** (perfect score) after 40 filler message pairs, while the naive baseline recalls 0/5. Three defense layers (importance scoring, entity-preserving compression, dual-write to knowledge store) ensure facts survive tier cascade compression.
- **Knowledge retrieval** is the strongest differentiator — CortexFlow retrieves 9/10 invented facts that no LLM could answer from training data alone, while the naive baseline scores 0/10.
- **System message preservation** works because CortexFlow assigns importance=9.0 to system messages, keeping them pinned through tier cascades.

### Chatbot Scenario Benchmark

End-to-end benchmark covering 6 real-world chatbot scenarios:

| # | Scenario | Result |
|---|----------|--------|
| 1 | Basic Conversation (5 turns) | 5/5 |
| 2 | Long Context Stress (50+ message pairs) | Tier cascade verified |
| 3 | Knowledge-Augmented Retrieval (20 facts, 8 queries) | 50%+ hit rate |
| 4 | Memory Persistence & Recall (30 filler pairs) | 1+ of 3 facts recalled |
| 5 | Context Window Overflow (100 message pairs) | No crashes, tiers within limits |
| 6 | Concurrent Knowledge + Conversation | 50%+ hit rate |

**Overall pass rate: 96.6% (28/29 checks).**

### Running the Benchmarks

Both suites require Vertex AI credentials (via environment variables) and make live LLM calls:

```bash
# Set credentials (service account JSON + project ID)
export VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
export VERTEX_PROJECT_ID=your-gcp-project

# Comparative evidence benchmark (~2-5 min):
python -m pytest tests/test_evidence_benchmark.py -v -s --timeout=900

# Chatbot scenario benchmark (~3-5 min):
python -m pytest tests/test_chatbot_benchmark.py -v -s --timeout=600

# Quick smoke test (single scenario):
python -m pytest tests/test_evidence_benchmark.py::TestEvidenceBenchmark::test_01_deep_memory_recall -v -s
```

## Documentation

For full documentation, visit [cortexflow-llm.readthedocs.io](https://cortexflow-llm.readthedocs.io).

## Development

```bash
pip install -e ".[dev]"

# Lint and format
ruff check cortexflow       # lint
ruff format cortexflow      # format

# Run tests with coverage
pytest --cov=cortexflow --cov-fail-under=80 tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Migrating from AdaptiveContext

CortexFlow was previously known as AdaptiveContext. To migrate:

1. Replace `from adaptive_context` with `from cortexflow` in all imports
2. Update flat config dicts to nested `CortexFlowConfig` dataclasses (see [migration guide](https://cortexflow-llm.readthedocs.io/en/latest/guides/migration.html))
