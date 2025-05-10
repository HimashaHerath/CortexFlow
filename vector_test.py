#!/usr/bin/env python3
"""
Test script for vector-based knowledge retrieval in AdaptiveContext.
This demonstrates the improved semantic search capabilities.
"""

import sys
import time
from adaptive_context import AdaptiveContextManager, AdaptiveContextConfig

def print_divider():
    print("\n" + "="*70 + "\n")

def main():
    print("Testing AdaptiveContext Vector-Based Knowledge Retrieval")
    print_divider()
    
    # Initialize with vector search enabled
    config = AdaptiveContextConfig(
        active_tier_tokens=1000,
        working_tier_tokens=2000,
        archive_tier_tokens=3000,
        vector_embedding_model='all-MiniLM-L6-v2',
        use_vector_search=True
    )
    
    manager = AdaptiveContextManager(config)
    
    # Store some facts explicitly
    print("Storing facts...")
    manager.explicitly_remember("Alice lives in Boston.")
    manager.explicitly_remember("Bob has a dog named Max.")
    manager.explicitly_remember("The capital of France is Paris.")
    manager.explicitly_remember("Python is a programming language.")
    manager.explicitly_remember("The Eiffel Tower is in Paris.")
    
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