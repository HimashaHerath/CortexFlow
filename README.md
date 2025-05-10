# AdaptiveContext

A memory optimization system for local LLMs running via Ollama that implements a multi-tier memory architecture inspired by human cognition and recent AI research (TITANS and Transformer²).

## Overview

AdaptiveContext dynamically manages context information, retaining important elements while discarding less relevant ones to maximize effective context window utilization without increasing memory requirements. The system uses a multi-tier memory approach:

- **Active Tier**: Holds the most recent and important context
- **Working Tier**: Medium-term storage with moderate compression
- **Archive Tier**: Long-term storage with higher compression
- **Knowledge Store**: Persistent storage for important facts beyond the conversation

## Features

- Multi-tier memory management optimizes token usage
- Importance-based information retention using rule-based, ML, and LLM classification
- Progressive context compression with extractive and abstractive summarization
- Knowledge store for long-term information persistence
- **Dynamic memory tier weighting** for adaptive token allocation based on query complexity and content type
- **Vector-based knowledge retrieval** for semantic search of stored knowledge
- **Advanced retrieval techniques**:
  - **Hybrid search** combining vector similarity with BM25 keyword scoring
  - **Sparse-dense fusion** merging results from different retrieval methods
  - **Re-ranking** for improved retrieval precision
  - **GraphRAG** using knowledge graphs for complex multi-hop queries
  - **Chain of Agents** for collaborative multi-agent reasoning over complex queries
  - **Self-Reflection** for verifying knowledge relevance and response consistency
- Compatible with all Ollama models (tested with Llama, Mistral, Phi, Gemma variants)
- Task-aware adaptation for different conversation types
- Surprise-based retention prioritizes unexpected or important information
- Time-based decay gradually deprioritizes older information

## Requirements

- Python 3.7+
- [Ollama](https://ollama.com/) with at least one model installed
- For GraphRAG functionality: 
  - `networkx` for graph operations
  - `spacy` with `en_core_web_sm` model for entity extraction

## Installation

```bash
# Clone the repository
git clone https://github.com/himashaherath/adaptivecontext.git
cd adaptivecontext

# Install dependencies
pip install -r requirements.txt

# TODO: Install as a package
# pip install -e .
```

## Quick Start

```python
from adaptive_context import AdaptiveContextManager, AdaptiveContextConfig

# Configure with custom settings
config = AdaptiveContextConfig(
    active_tier_tokens=2000,
    working_tier_tokens=4000,
    archive_tier_tokens=6000,
    default_model="llama3"  # Use your preferred Ollama model
)

# Create the context manager
context_manager = AdaptiveContextManager(config)

# Add messages to the context
context_manager.add_message("You are a helpful AI assistant.", segment_type="system")
context_manager.add_message("What is the capital of France?", segment_type="user")
context_manager.add_message("The capital of France is Paris.", segment_type="assistant")

# Get full context for sending to LLM
full_context = context_manager.get_full_context()

# Explicitly store important information
context_manager.explicitly_remember("The user's name is Alice and she lives in Boston.")

# Get stats about memory usage
stats = context_manager.get_stats()
print(stats)

# Clear context memory (but keep knowledge store)
context_manager.flush()

# Clean up when done
context_manager.close()
```

## Command-line Tools

### Interactive Example

Run the interactive example to chat with your Ollama model using adaptive context:

```bash
python example.py
```

Or specify a different model:

```bash
python example.py --model mistral
```

Available options:
- `--model`: Ollama model to use (default: llama3)
- `--host`: Ollama API host (default: http://localhost:11434)
- `--active-tokens`: Active tier token limit (default: 2000)
- `--working-tokens`: Working tier token limit (default: 4000)
- `--archive-tokens`: Archive tier token limit (default: 6000)
- `--db`: Knowledge store database path (default: in-memory)

### Testing

Run the basic functionality test:

```bash
python test.py
```

Run the Ollama integration test:

```bash
python ollama_test.py
```

Test the vector-based knowledge retrieval:

```bash
python vector_test.py
```

Test advanced retrieval techniques:

```bash
python advanced_retrieval_test.py
```

Test the Chain of Agents functionality:

```bash
python coa_test.py --model gemma3:1b
```

Test the Self-Reflection functionality:

```bash
python self_reflection_test.py --model llama3.2
```

Test the Dynamic Memory Weighting functionality:

```bash
python dynamic_weighting_test.py --model llama3.2 --plot
```

### Unit Testing

The project includes comprehensive unit tests using pytest:

```bash
# Install pytest and pytest-cov if not already installed
pip install pytest pytest-cov

# Run all unit tests
pytest tests/

# Generate test coverage report
pytest --cov=adaptive_context tests/

# Generate HTML coverage report
pytest --cov=adaptive_context --cov-report=html tests/
```

For more information on running tests, see [tests/README.md](tests/README.md).

## Documentation

Comprehensive API documentation is available in the `docs/api/` directory:

- [API Reference](docs/api/index.md) - Complete API documentation
- [Dynamic Weighting](docs/dynamic_weighting.md) - Documentation for the dynamic weighting feature
- [Chain of Agents](docs/chain_of_agents.md) - Documentation for the Chain of Agents framework
- [Self-Reflection](docs/self_reflection.md) - Documentation for the Self-Reflection feature
- [Research Improvements](docs/research_improvements.md) - Research-based improvements implemented and planned

## Core Components

### Memory Manager

Orchestrates context flow between the three memory tiers, managing token limits and tier transitions. The manager intelligently moves context segments based on importance scores, age, and tier fullness, ensuring essential information remains accessible while less important content is compressed or archived.

### Importance Classifier

Determines which information is worth retaining using three classification methods:
1. **Rule-based**: Uses patterns and heuristics to identify important content
2. **ML-based**: Uses lightweight features for fast classification (simplified implementation)
3. **LLM-based**: Uses the Ollama model itself for high-precision classification

### Context Compressor

Reduces token count while preserving semantic meaning using multiple compression strategies:
1. **Truncation**: Simple truncation for code blocks and very short content
2. **Extractive**: Keyword-based sentence extraction for medium-importance content
3. **Abstractive**: LLM-based summarization for high-importance content

### Knowledge Store

Persists important facts and retrievable context beyond the immediate conversation:
1. Stores structured fact triples (subject-predicate-object)
2. Maintains conversation summaries with keyword indexing
3. Supports explicit "remember" commands
4. Implements time-based forgetting for outdated information
5. **Uses vector embeddings** for semantic search and retrieval
6. **Advanced retrieval techniques**:
   - **Hybrid search**: Combines dense vector similarity with sparse BM25 keyword scoring
   - **Result fusion**: Merges results from different retrieval methods with normalized scores
   - **Re-ranking**: Applies a secondary scoring pass to refine retrieval precision
   - **GraphRAG**: Builds and queries a knowledge graph for complex multi-hop queries

### Graph Store

Manages a knowledge graph for complex relational queries:
1. Extracts entities and relationships from text using NER and dependency parsing
2. Builds a graph database with entities as nodes and relationships as edges
3. Enables complex multi-hop queries across documents
4. Provides path finding between entities to answer complex questions
5. Integrates with the knowledge store for comprehensive information retrieval

### Chain of Agents Framework

Enables collaborative multi-agent reasoning over complex queries:
1. Explorer Agent searches broadly for relevant information
2. Analyzer Agent examines relationships between facts
3. Synthesizer Agent generates comprehensive answers
4. Sequential processing allows for complex reasoning chains
5. Specialized prompt templates guide each agent's focus

Each agent in the chain serves a specific role:
- **Explorer Agent**: Concentrates on broad exploration without answering the query directly, retrieving diverse relevant knowledge using both vector and graph-based approaches
- **Analyzer Agent**: Identifies relationships between discovered facts, finding connections and patterns relevant to the query
- **Synthesizer Agent**: Combines insights from previous agents to generate a comprehensive final answer that integrates all discovered information

This multi-agent approach follows research from Google's "Chain of Agents: Large Language Models Collaborating on Long Context Tasks" (2025) and is particularly effective for:
- Complex queries requiring multi-hop reasoning
- Questions that benefit from structured exploration before answering
- Queries where connecting multiple pieces of information is necessary

To use the Chain of Agents framework:
```python
# Enable Chain of Agents in your configuration
config = AdaptiveContextConfig(
    use_chain_of_agents=True,
    chain_complexity_threshold=5,  # Only use for reasonably complex queries
    # ... other config options
)

# Create the manager with this configuration
manager = AdaptiveContextManager(config)

# Regular usage - CoA is automatically engaged for complex queries
manager.add_message("user", "What connection exists between X and Y?")
response = manager.generate_response()
```

### Self-Reflection and Self-Correction

Implements verification and correction mechanisms to improve response accuracy:
1. **Knowledge Relevance Verification**: Evaluates retrieved knowledge items for relevance to the query
2. **Response Consistency Checking**: Analyzes responses for factual inconsistencies with the knowledge base
3. **Automated Response Revision**: Revises responses when inconsistencies are detected
4. **Confidence Calibration**: Expresses appropriate uncertainty when knowledge is limited or contradictory

The Self-Reflection mechanism is particularly effective for:
- Handling contradictory information in the knowledge base
- Preventing hallucinations and unsupported claims
- Ensuring factual accuracy in responses
- Resolving ambiguities in retrieved knowledge

To use the Self-Reflection functionality:
```python
# Enable Self-Reflection in your configuration
config = AdaptiveContextConfig(
    use_self_reflection=True,
    reflection_relevance_threshold=0.6,  # Minimum score for knowledge relevance
    reflection_confidence_threshold=0.7,  # Minimum confidence for consistency
    # ... other config options
)

# Create the manager with this configuration
manager = AdaptiveContextManager(config)

# Regular usage - Self-Reflection is automatically applied
manager.add_message("user", "What is X?")
response = manager.generate_response()  # Response is verified and corrected if needed
```

For more details on Self-Reflection, see [docs/self_reflection.md](docs/self_reflection.md).

### Dynamic Memory Tier Weighting

Implements adaptive token allocation between memory tiers based on conversation characteristics:
1. **Query Complexity Analysis**: Evaluates the complexity of user queries using linguistic features
2. **Document Type Detection**: Identifies the type of content in the conversation (code, text, data, mixed)
3. **Historical Pattern Analysis**: Monitors query complexity patterns over time
4. **Adaptive Token Allocation**: Dynamically adjusts token limits for each memory tier

The Dynamic Weighting mechanism is particularly effective for:
- Optimizing token usage across different conversation types
- Adapting to changing complexity levels during a conversation
- Providing more memory resources to tiers that need them most
- Automatically adjusting to user interaction patterns

To use the Dynamic Weighting functionality:
```python
# Enable Dynamic Weighting in your configuration
config = AdaptiveContextConfig(
    use_dynamic_weighting=True,
    dynamic_weighting_learning_rate=0.1,  # Learning rate for weight adjustments
    dynamic_weighting_min_tier_size=1000,  # Minimum token size for any tier
    # ... other config options
)

# Create the manager with this configuration
manager = AdaptiveContextManager(config)

# Regular usage - Dynamic Weighting is automatically applied
manager.add_message("user", "What is X?")
response = manager.generate_response()  # Memory tiers are dynamically adjusted

# Get statistics about dynamic weighting
stats = manager.get_dynamic_weighting_stats()
print(stats)

# Reset to default weights if needed
manager.reset_dynamic_weighting()
```

For more details on Dynamic Weighting, see [docs/dynamic_weighting.md](docs/dynamic_weighting.md).

## Configuration Options

AdaptiveContext is highly configurable. Key configuration options include:

```python
config = AdaptiveContextConfig(
    # Memory tier settings
    active_tier_tokens=4096,     # Maximum tokens in active memory tier
    working_tier_tokens=8192,    # Maximum tokens in working memory tier
    archive_tier_tokens=16384,   # Maximum tokens in archive memory tier
    
    # Importance thresholds
    working_importance_threshold=0.3,  # Minimum importance for working tier
    archive_importance_threshold=0.1,  # Minimum importance for archive tier
    
    # Token counting method
    token_counting_method="basic",  # "basic", "tiktoken", "ollama"
    
    # Memory compression
    enable_compression=True,      # Whether to enable compression
    compression_threshold=0.8,    # Tier fullness threshold to trigger compression
    compression_target=0.6,       # Target fullness after compression
    
    # Classification parameters
    classifier_model="gpt-3.5-turbo",  # Model for importance classification
    classifier_temperature=0.1,        # Temperature for classification
    
    # Vector embeddings
    vector_embedding_model="all-MiniLM-L6-v2",  # Model for vector embeddings
    
    # Ollama settings
    ollama_host="http://localhost:11434",
    default_model="gemma3:1b",    # Default Ollama model to use
    
    # Persistence paths
    storage_path="~/.adaptive_context",  # Base storage path
    knowledge_store_path="~/.adaptive_context/knowledge.db",  # Knowledge DB path
    
    # Retrieval settings
    use_vector_embeddings=True,   # Whether to use vector embeddings
    use_reranking=True,           # Whether to use result re-ranking
    
    # GraphRAG configuration
    use_graph_rag=True,           # Whether to use knowledge graph for retrieval
    graph_weight=0.3,             # Weight for graph-based results in ranking
    enable_multi_hop_queries=True,  # Enable complex multi-hop queries
    max_graph_hops=3,             # Maximum path length for graph traversal
    
    # Chain of Agents settings
    use_chain_of_agents=True,      # Enable Chain of Agents framework
    chain_complexity_threshold=5,  # Minimum query complexity for CoA
    chain_agent_count=3,           # Number of agents in the chain
    
    # Self-Reflection settings
    use_self_reflection=True,      # Enable Self-Reflection functionality
    reflection_relevance_threshold=0.6,  # Minimum score for knowledge relevance
    reflection_confidence_threshold=0.7,  # Minimum confidence for consistency
    
    # Dynamic Weighting settings
    use_dynamic_weighting=True,    # Enable Dynamic Memory Tier Weighting
    dynamic_weighting_learning_rate=0.1,  # Learning rate for weight adjustments
    dynamic_weighting_min_tier_size=1000,  # Minimum token size for any tier
    dynamic_weighting_default_ratios={     # Default tier ratios
        "active": 0.25,
        "working": 0.35,
        "archive": 0.40
    }
)
```

## How It Works

1. When new messages are added, they're classified for importance (0-1 scale)
2. Messages are added to the active tier initially
3. As tiers fill up, less important and older messages are moved to lower tiers
4. Messages moved to lower tiers undergo progressive compression
5. Very important facts are extracted to the knowledge store for permanent retention
6. When relevant, knowledge is retrieved using advanced hybrid search techniques and added to the context
7. For complex queries, GraphRAG builds and traverses a knowledge graph to find relationships between entities
8. Memory tier token limits are dynamically adjusted based on query complexity and content type

This approach optimizes token usage while maintaining contextual understanding across long conversations.

## Test GraphRAG Functionality

Test the GraphRAG knowledge graph functionality:

```bash
python graph_rag_benchmark.py
```

This benchmark evaluates the GraphRAG functionality with various types of queries:
- Single-hop queries: Direct entity-relation lookups
- Multi-hop queries: Queries requiring traversal across multiple relationships
- Counterfactual queries: Queries that test negative examples

Command line options:
```bash
python graph_rag_benchmark.py --help

# Options:
# --model MODEL           Ollama model to use (default: gemma3:1b)
# --graph-weight WEIGHT   Weight for graph-based results (default: 0.3)
# --max-hops HOPS         Maximum path length for graph traversal (default: 3)
# --db-path PATH          Database path (default: benchmark_graph_rag.db)
# --keep-db               Keep existing database
# --reload-knowledge      Force reload knowledge even if DB exists
# --output FILE           Output file for benchmark results (JSON)
# --verbose               Verbose output
# --plot                  Generate plots
# --evaluate-llm          Evaluate LLM answer accuracy
```

Example: Run the benchmark with verbose output and keep the database:
```bash
python graph_rag_benchmark.py --verbose --keep-db
```

## Project Structure

```
adaptivecontext/
├── adaptive_context/      # Main package
│   ├── __init__.py        # Package initialization
│   ├── manager.py         # AdaptiveContextManager
│   ├── memory.py          # Memory tiers implementation
│   ├── classifier.py      # Importance classification
│   ├── compressor.py      # Context compression
│   ├── knowledge.py       # Knowledge store
│   ├── config.py          # Configuration
│   ├── graph_store.py     # Knowledge graph store
│   ├── agent_chain.py     # Chain of Agents implementation
│   ├── reflection.py      # Self-Reflection implementation
│   └── dynamic_weighting.py # Dynamic Memory Weighting implementation
├── examples/              # Example scripts
│   └── chat_example.py    # Interactive chat example
├── tests/                 # Test scripts
│   ├── test.py            # Basic functionality tests
│   └── ollama_test.py     # Ollama integration tests
├── graph_rag_benchmark.py # GraphRAG evaluation benchmark
├── self_reflection_test.py # Self-Reflection test script
├── dynamic_weighting_test.py # Dynamic Weighting test script
├── metrics_utils.py       # Utilities for benchmark metrics
├── requirements.txt       # Dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## GraphRAG Architecture

The GraphRAG system combines traditional retrieval-augmented generation with knowledge graph capabilities:

1. **Entity Extraction**: Identifies entities in text using SpaCy NER, noun phrase extraction, and proper noun detection
2. **Relation Extraction**: Extracts relationships between entities using dependency parsing
3. **Knowledge Graph**: Builds a graph database using SQLite and NetworkX
4. **Graph Queries**:
   - Entity-based retrieval: Finds facts about specific entities
   - Path queries: Finds connections between entities
   - Subgraph extraction: Builds knowledge subgraphs for complex queries
5. **Integration with Vector Retrieval**: Combines graph-based and vector-based retrieval results

The GraphRAG approach is particularly effective for:
- Questions about relationships between entities
- Multi-hop reasoning (e.g., "What connects X and Y?")
- Complex queries that traditional vector retrieval struggles with

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Benchmarking Framework

AdaptiveContext includes a comprehensive benchmarking framework to evaluate and compare its performance against other RAG and context management systems.

### Running Benchmarks

To run the benchmarks, use the following commands:

```bash
# Install benchmark requirements
pip install -r benchmark/requirements.txt

# Run a benchmark against a specific dataset
python -m benchmark.run_benchmark --systems adaptivecontext --dataset hotpotqa

# Run benchmarks against multiple systems
python -m benchmark.run_benchmark --systems adaptivecontext,llamaindex,langchain --dataset complex

# Run all benchmarks
python -m benchmark.run_benchmark --all-systems --visualize
```

### Benchmark Features

The benchmark framework evaluates systems across multiple dimensions:

1. **Retrieval Quality** - How accurately the system can retrieve and use relevant information
2. **Memory Efficiency** - How efficiently the system manages token usage and context
3. **Query Complexity** - How well the system handles multi-hop and relationship questions

### Adding New Systems

To add a new system for comparison:

1. Create a new adapter in `benchmark/adapters/`
2. Implement the `BenchmarkSystemAdapter` interface
3. Register the adapter in `benchmark/registry.py`