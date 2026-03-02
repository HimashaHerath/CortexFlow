#!/usr/bin/env python3
"""
Test script for vector-based knowledge retrieval in CortexFlow.
This demonstrates the improved semantic search capabilities.
"""

import sys
import time
from cortexflow import CortexFlowManager, CortexFlowConfig, MemoryConfig, KnowledgeStoreConfig

def print_divider():
    print("\n" + "="*70 + "\n")

def main():
    print("Testing CortexFlow Vector-Based Knowledge Retrieval")
    print_divider()

    # Initialize with vector search enabled using nested config
    config = CortexFlowConfig(
        memory=MemoryConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000,
        ),
        knowledge_store=KnowledgeStoreConfig(
            vector_model='all-MiniLM-L6-v2',
            retrieval_type='hybrid',
        ),
    )

    manager = CortexFlowManager(config)

    # Store some facts using the knowledge store's add_knowledge method
    print("Storing facts...")
    manager.knowledge_store.add_knowledge("Alice lives in Boston.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Bob has a dog named Max.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("The capital of France is Paris.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Python is a programming language.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("The Eiffel Tower is in Paris.", source="demo", confidence=0.95)

    # Wait for storage to complete
    time.sleep(1)
    print("Facts stored successfully.")
    print_divider()

    # Test exact matches
    print("Testing exact match queries:")

    queries = [
        "Where does Alice live?",
        "What is Bob's dog's name?",
        "What is the capital of France?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = manager.knowledge_store.get_relevant_knowledge(query)
        if results:
            print("Results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['content']}")
                if 'similarity' in result:
                    print(f"   Similarity score: {result['similarity']:.4f}")
        else:
            print("No results found")

    print_divider()

    # Test semantic matches (should work well with vector search)
    print("Testing semantic match queries:")

    semantic_queries = [
        "Where is Alice's residence?",  # Semantic match for "Alice lives in Boston"
        "What pet does Bob own?",       # Semantic match for "Bob has a dog named Max"
        "What famous monument is located in France's capital?",  # Should connect Eiffel Tower and Paris
        "Is Python a coding language?"  # Semantic match for "Python is a programming language"
    ]

    for query in semantic_queries:
        print(f"\nQuery: {query}")
        results = manager.knowledge_store.get_relevant_knowledge(query)
        if results:
            print("Results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['content']}")
                if 'similarity' in result:
                    print(f"   Similarity score: {result['similarity']:.4f}")
        else:
            print("No results found")

    print_divider()

    # Clean up
    manager.close()
    print("Test completed successfully")

if __name__ == "__main__":
    main()
