"""Pluggable vector store backends quickstart.

Shows how to use different vector backends with CortexFlow.
"""
import numpy as np

from cortexflow import ConfigBuilder
from cortexflow.vector_stores import create_vector_store

# --- Default SQLite backend (zero config) ---
config = ConfigBuilder().build()
store = create_vector_store(config)
print(f"Backend type: {type(store).__name__}")

# Add embeddings
rng = np.random.default_rng(42)
for i in range(5):
    embedding = rng.random(384).astype(np.float32)
    store.add_embedding(
        id=f"doc_{i}",
        text=f"Document {i} about topic {['AI', 'ML', 'NLP', 'CV', 'RL'][i]}",
        embedding=embedding,
        metadata={"source": "example"},
    )

print(f"Stored {store.count()} embeddings")

# Search
query = rng.random(384).astype(np.float32)
results = store.search(query, max_results=3)
print("\nTop 3 results:")
for r in results:
    print(f"  - [{r.score:.4f}] {r.text}")

store.close()

# --- ChromaDB backend (requires: pip install chromadb) ---
try:
    config = ConfigBuilder().with_vector_store(backend="chromadb").build()
    chroma_store = create_vector_store(config)
    print(f"\nChromaDB backend: {type(chroma_store).__name__}")
    chroma_store.close()
except ImportError:
    print("\nChromaDB not installed — skipping (pip install chromadb)")

# --- Qdrant backend (requires: pip install qdrant-client) ---
try:
    config = ConfigBuilder().with_vector_store(backend="qdrant").build()
    qdrant_store = create_vector_store(config)
    print(f"Qdrant backend: {type(qdrant_store).__name__}")
    qdrant_store.close()
except ImportError:
    print("Qdrant not installed — skipping (pip install qdrant-client)")

print("\nDone!")
