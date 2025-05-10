#!/usr/bin/env python3
"""
Test script to demonstrate GraphRAG functionality in AdaptiveContext.
"""

import os
import sys
import time
import logging
import requests
import json
from typing import Dict, Any, List

from adaptive_context import AdaptiveContextManager
from adaptive_context.config import AdaptiveContextConfig
from adaptive_context.graph_store import GraphStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test data with connected facts for knowledge graph
TEST_KNOWLEDGE = [
    # Technology domain - explicit relationships
    "Python is a programming language created by Guido van Rossum.",
    "Guido van Rossum developed Python in the early 1990s.",
    "Guido van Rossum worked at Google from 2005 to 2012.",
    "Guido van Rossum joined Dropbox after leaving Google.",
    "Python supports multiple programming paradigms including object-oriented programming.",
    "Google uses Python extensively for many internal systems.",
    "Google was founded by Larry Page and Sergey Brin in 1998.",
    "TensorFlow is a machine learning framework created by Google.",
    "TensorFlow supports Python as its main programming interface.",
    "PyTorch is a machine learning framework developed by Facebook.",
    "Facebook was founded by Mark Zuckerberg in 2004.",
    "Facebook uses Python for many of its services and tools.",
    
    # Geography domain - explicit relationships
    "San Francisco is located in California.",
    "California is a state in the western United States.",
    "Los Angeles is the largest city in California.",
    "Hollywood is a neighborhood in Los Angeles known for movie production.",
    "The Golden Gate Bridge is a famous landmark in San Francisco.",
    "Silicon Valley is a region in California's San Francisco Bay Area.",
    "Silicon Valley is home to many technology companies and startups.",
    "Google's headquarters are located in Mountain View, California.",
    "Mountain View is a city in Silicon Valley.",
    "Facebook's headquarters are located in Menlo Park, California.",
    "Menlo Park is a city in Silicon Valley.",
    
    # Connections between domains - explicit relationships
    "Many technology companies like Google and Facebook have offices in California.",
    "Python is widely used by Silicon Valley companies for development.",
    "Google develops TensorFlow, which is popular in Silicon Valley for AI research.",
    "Facebook develops PyTorch, which competes with Google's TensorFlow.",
    "Guido van Rossum worked at Google's Silicon Valley campus.",
    "Python is the most popular programming language in Silicon Valley.",
]

def test_graph_rag():
    """Test GraphRAG functionality."""
    
    # Create a test DB path
    test_db_path = "test_graph_rag.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # Initialize config with GraphRAG enabled
    config = AdaptiveContextConfig(
        knowledge_store_path=test_db_path,
        use_graph_rag=True,
        graph_weight=0.3,
        enable_multi_hop_queries=True,
        max_graph_hops=3,
        default_model="gemma3:1b"
    )
    
    # Initialize manager
    manager = AdaptiveContextManager(config)
    
    print("\n" + "="*80)
    print("Testing GraphRAG Functionality")
    print("="*80)
    
    # Add knowledge to the system
    print("\nAdding knowledge to the system...")
    for i, knowledge in enumerate(TEST_KNOWLEDGE):
        print(f"[{i+1}/{len(TEST_KNOWLEDGE)}] Adding: {knowledge}")
        manager.knowledge_store.remember_explicit(
            text=knowledge,
            source="test_data",
            confidence=0.9
        )
    
    # Print graph statistics
    print("\nGraph Statistics:")
    graph_store = manager.knowledge_store.graph_store
    if graph_store.graph:
        print(f"Number of entities (nodes): {graph_store.graph.number_of_nodes()}")
        print(f"Number of relationships (edges): {graph_store.graph.number_of_edges()}")
    
    # Test single-hop queries
    print("\nTesting single-hop queries:")
    single_hop_queries = [
        "What is Python?",
        "Who created Python?",
        "Where is Google's headquarters?",
        "What is in California?"
    ]
    
    for query in single_hop_queries:
        print(f"\nQUERY: {query}")
        results = manager.knowledge_store.get_relevant_knowledge(query, max_results=3)
        print("Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['text']}")
    
    # Test multi-hop queries
    print("\nTesting multi-hop queries:")
    multi_hop_queries = [
        "What is the connection between Python and Google?",
        "How are Silicon Valley and Python related?",
        "What connects Facebook and machine learning?",
        "Is there a connection between California and Guido van Rossum?"
    ]
    
    for query in multi_hop_queries:
        print(f"\nQUERY: {query}")
        results = manager.knowledge_store.get_relevant_knowledge(query, max_results=3)
        
        # Print the actual graph search details
        print("Graph search details:")
        graph_results = manager.knowledge_store._graph_search(query, max_results=5)
        for i, result in enumerate(graph_results):
            print(f"  {i+1}. {result.get('text', '')} [Score: {result.get('score', 0):.2f}, Type: {result.get('type', 'unknown')}]")
        
        print("Final results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['text']} [Type: {result.get('type', 'unknown')}]")
    
    # Test with Ollama integration if available
    print("\nTesting with Ollama integration:")
    try:
        # Check if Ollama is running
        response = requests.get(f"{config.ollama_host}/api/tags")
        if response.status_code == 200:
            ollama_available = True
            print("Ollama detected - proceeding with LLM test")
        else:
            ollama_available = False
            print("Ollama not available - skipping LLM test")
    except:
        ollama_available = False
        print("Ollama not available - skipping LLM test")
    
    if ollama_available:
        test_questions = [
            "Who created Python and where did they work?",
            "What machine learning frameworks were developed by big tech companies?",
            "Tell me about the connection between California and technology companies"
        ]
        
        for question in test_questions:
            print(f"\nQUESTION: {question}")
            
            # Get relevant knowledge using GraphRAG
            context = manager.knowledge_store.get_relevant_knowledge(question, max_results=5)
            context_text = "\n".join([item["text"] for item in context])
            
            # Build prompt for Ollama
            prompt = f"""
            Based on the following context, please answer the question. 
            If you cannot answer from the provided context, say "I don't have enough information."
            
            Context:
            {context_text}
            
            Question: {question}
            Answer:
            """
            
            # Call Ollama
            try:
                response = requests.post(
                    f"{config.ollama_host}/api/generate",
                    json={"model": config.default_model, "prompt": prompt, "stream": False}
                )
                
                if response.status_code == 200:
                    answer = response.json().get("response", "Error: No response")
                    print(f"ANSWER: {answer}\n")
                else:
                    print(f"Error: {response.status_code} - {response.text}\n")
            except Exception as e:
                print(f"Error calling Ollama: {e}\n")
    
    # Test graph traversal
    if graph_store.graph:
        print("\nTesting graph traversal:")
        
        # Find entities
        print("\nEntities by type:")
        person_entities = graph_store.query_entities(entity_type="PERSON", limit=5)
        print("People:")
        for entity in person_entities:
            print(f"  - {entity['entity']} (ID: {entity['id']})")
            
        org_entities = graph_store.query_entities(entity_type="ORG", limit=5)
        print("Organizations:")
        for entity in org_entities:
            print(f"  - {entity['entity']} (ID: {entity['id']})")
        
        # Test path queries
        print("\nPath queries:")
        if person_entities and len(person_entities) >= 2:
            person1 = person_entities[0]['entity']
            person2 = person_entities[-1]['entity']
            print(f"Finding paths between {person1} and {person2}:")
            
            paths = graph_store.path_query(
                start_entity=person1,
                end_entity=person2,
                max_hops=3
            )
            
            if paths:
                for i, path in enumerate(paths):
                    path_text = manager.knowledge_store._format_path_as_text(path)
                    print(f"  Path {i+1}: {path_text}")
            else:
                print("  No paths found")
    
    print("\nCleaning up...")
    manager.close()
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_graph_rag() 