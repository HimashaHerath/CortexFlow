#!/usr/bin/env python3
"""
Test script for advanced vector retrieval techniques in AdaptiveContext.
This demonstrates hybrid search (BM25 + vector), sparse-dense fusion, and re-ranking.
"""

import sys
import time
import json
from adaptive_context import AdaptiveContextManager, AdaptiveContextConfig

def print_divider():
    print("\n" + "="*70 + "\n")

def compare_search_methods(manager, query):
    """Compare different search methods for the same query."""
    # Test pure vector search
    config = manager.knowledge_store.config
    
    # Save original config settings
    original_bm25 = config.use_bm25_search
    original_reranking = config.use_reranking
    
    # 1. Pure vector search (no BM25, no reranking)
    config.use_bm25_search = False
    config.use_reranking = False
    vector_results = manager.knowledge_store.get_relevant_knowledge(query)
    
    # 2. Pure BM25 search (no vectors, no reranking)
    config.use_bm25_search = True
    config.use_vector_search = False
    bm25_results = manager.knowledge_store.get_relevant_knowledge(query)
    
    # 3. Hybrid search (both vectors and BM25, no reranking)
    config.use_vector_search = True
    hybrid_results = manager.knowledge_store.get_relevant_knowledge(query)
    
    # 4. Full advanced retrieval (hybrid search with reranking)
    config.use_reranking = True
    reranked_results = manager.knowledge_store.get_relevant_knowledge(query)
    
    # Restore original settings
    config.use_bm25_search = original_bm25
    config.use_reranking = original_reranking
    
    return {
        "vector": vector_results,
        "bm25": bm25_results,
        "hybrid": hybrid_results,
        "reranked": reranked_results
    }

def print_results(results, method_name):
    """Print search results with method name."""
    print(f"\n{method_name} Results:")
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['content']}")
            if 'similarity' in result:
                print(f"   Vector Score: {result['similarity']:.4f}")
            if 'bm25_score' in result:
                print(f"   BM25 Score: {result['bm25_score']:.4f}")
            if 'combined_score' in result:
                print(f"   Combined Score: {result['combined_score']:.4f}")
            if 'rerank_score' in result:
                print(f"   Rerank Score: {result['rerank_score']:.4f}")
    else:
        print("No results found")

def main():
    print("Testing AdaptiveContext Advanced Retrieval Techniques")
    print_divider()
    
    # Initialize with all advanced retrieval features enabled
    config = AdaptiveContextConfig(
        active_tier_tokens=1000,
        working_tier_tokens=2000,
        archive_tier_tokens=3000,
        vector_embedding_model='all-MiniLM-L6-v2',
        use_vector_search=True,
        use_bm25_search=True,
        hybrid_search_alpha=0.7,
        use_reranking=True,
        rerank_top_k=20
    )
    
    manager = AdaptiveContextManager(config)
    
    # Store some test facts
    print("Storing sample facts...")
    manager.explicitly_remember("Alice lives in Boston, Massachusetts.")
    manager.explicitly_remember("Boston is the capital of Massachusetts.")
    manager.explicitly_remember("Boston is known for its prestigious universities.")
    manager.explicitly_remember("Harvard University is located in Cambridge near Boston.")
    manager.explicitly_remember("MIT is a world-renowned technical institute in Cambridge.")
    manager.explicitly_remember("The Boston Red Sox play at Fenway Park.")
    manager.explicitly_remember("Bob works at a technology company in Silicon Valley.")
    manager.explicitly_remember("Python is a popular programming language used in data science.")
    manager.explicitly_remember("Machine learning models require training data.")
    manager.explicitly_remember("The Eiffel Tower is a famous landmark in Paris, France.")
    
    # Wait for storage to complete
    time.sleep(1)
    print("Facts stored successfully.")
    print_divider()
    
    # Test with various queries
    test_queries = [
        "Where does Alice live?",
        "Tell me about universities near Boston",
        "What is Boston known for?",
        "machine learning and AI",
        "famous landmarks in Europe"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        
        # Compare different search methods
        results_by_method = compare_search_methods(manager, query)
        
        # Print results from each method
        print_results(results_by_method["vector"], "Pure Vector Search")
        print_results(results_by_method["bm25"], "Pure BM25 Search")
        print_results(results_by_method["hybrid"], "Hybrid Search (Vector + BM25)")
        print_results(results_by_method["reranked"], "Reranked Hybrid Search")
        
        print_divider()
    
    # Clean up
    manager.close()
    print("Test completed successfully.")

if __name__ == "__main__":
    main() 