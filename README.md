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
- **Vector-based knowledge retrieval** for semantic search of stored knowledge
- **Advanced retrieval techniques**:
  - **Hybrid search** combining vector similarity with BM25 keyword scoring
  - **Sparse-dense fusion** merging results from different retrieval methods
  - **Re-ranking** for improved retrieval precision
- Compatible with all Ollama models (tested with Llama, Mistral, Phi, Gemma variants)
- Task-aware adaptation for different conversation types
- Surprise-based retention prioritizes unexpected or important information
- Time-based decay gradually deprioritizes older information

## Requirements

- Python 3.7+
- [Ollama](https://ollama.com/) with at least one model installed

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

## Configuration Options

AdaptiveContext is highly configurable. Key configuration options include:

```python
config = AdaptiveContextConfig(
    # Memory tier settings
    active_tier_tokens=2000,     # Maximum tokens in active memory tier
    working_tier_tokens=4000,    # Maximum tokens in working memory tier
    archive_tier_tokens=6000,    # Maximum tokens in archive memory tier
    
    # Classification settings
    use_ml=False,                # Whether to use ML-based importance classifier
    use_llm_classification=True, # Whether to use LLM for classification
    rule_weight=0.5,             # Weight for rule-based classifier
    ml_weight=0.3,               # Weight for ML-based classifier
    llm_weight=0.7,              # Weight for LLM-based classifier
    
    # Compression settings
    compression_threshold=0.8,   # Tier fullness threshold to trigger compression
    
    # Ollama settings
    ollama_host='http://localhost:11434',
    default_model='llama3',      # Default Ollama model to use
    
    # Knowledge retrieval settings
    vector_embedding_model='all-MiniLM-L6-v2',  # Model for vector embeddings
    use_vector_search=True,      # Whether to use vector-based semantic search
    use_bm25_search=True,        # Whether to use BM25 keyword search
    hybrid_search_alpha=0.7,     # Weight for vector search in hybrid search (0-1)
    use_reranking=True,          # Whether to use result re-ranking
    rerank_top_k=20              # Number of candidates to consider for re-ranking
)
```

## How It Works

1. When new messages are added, they're classified for importance (0-10 scale)
2. Messages are added to the active tier initially
3. As tiers fill up, less important and older messages are moved to lower tiers
4. Messages moved to lower tiers undergo progressive compression
5. Very important facts are extracted to the knowledge store for permanent retention
6. When relevant, knowledge is retrieved using advanced hybrid search techniques and added to the context

This approach optimizes token usage while maintaining contextual understanding across long conversations.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 