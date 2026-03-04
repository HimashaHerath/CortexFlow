#!/usr/bin/env python3
"""
Test script for advanced vector retrieval techniques in CortexFlow.
This demonstrates hybrid search (BM25 + vector), sparse-dense fusion, and re-ranking.
"""

import time

from cortexflow import (
    CortexFlowConfig,
    CortexFlowManager,
    KnowledgeStoreConfig,
    MemoryConfig,
)


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
    print("Testing CortexFlow Advanced Retrieval Techniques")
    print_divider()

    # Initialize with all advanced retrieval features enabled using nested config
    config = CortexFlowConfig(
        memory=MemoryConfig(
            active_token_limit=1000,
            working_token_limit=2000,
            archive_token_limit=3000,
        ),
        knowledge_store=KnowledgeStoreConfig(
            vector_model='all-MiniLM-L6-v2',
            retrieval_type='hybrid',
            use_reranking=True,
            rerank_top_k=20,
        ),
    )

    manager = CortexFlowManager(config)

    # Store some test facts using the knowledge store's add_knowledge method
    print("Storing sample facts...")
    manager.knowledge_store.add_knowledge("Alice lives in Boston, Massachusetts.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Boston is the capital of Massachusetts.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Boston is known for its prestigious universities.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Harvard University is located in Cambridge near Boston.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("MIT is a world-renowned technical institute in Cambridge.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("The Boston Red Sox play at Fenway Park.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Bob works at a technology company in Silicon Valley.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Python is a popular programming language used in data science.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("Machine learning models require training data.", source="demo", confidence=0.95)
    manager.knowledge_store.add_knowledge("The Eiffel Tower is a famous landmark in Paris, France.", source="demo", confidence=0.95)

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
