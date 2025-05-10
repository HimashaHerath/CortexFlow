#!/usr/bin/env python3
"""
AdaptiveContext Chain of Agents Test

This script tests the Chain of Agents functionality in AdaptiveContext.
"""

import argparse
import json
import time
from adaptive_context import AdaptiveContextManager, AdaptiveContextConfig

def test_chain_of_agents():
    """Test the Chain of Agents functionality."""
    parser = argparse.ArgumentParser(description="Test AdaptiveContext Chain of Agents")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--active-tokens", type=int, default=1000, help="Active tier token limit")
    parser.add_argument("--working-tokens", type=int, default=2000, help="Working tier token limit")
    parser.add_argument("--archive-tokens", type=int, default=3000, help="Archive tier token limit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    print(f"Testing AdaptiveContext Chain of Agents with model {args.model}...")
    
    # Initialize with in-memory storage and Chain of Agents enabled
    config = AdaptiveContextConfig(
        active_token_limit=args.active_tokens,
        working_token_limit=args.working_tokens,
        archive_token_limit=args.archive_tokens,
        knowledge_store_path=":memory:",
        ollama_host=args.host,
        default_model=args.model,
        use_graph_rag=True,  # Enable graph RAG
        use_chain_of_agents=True,  # Enable Chain of Agents
        verbose_logging=args.verbose
    )
    
    # Check if required packages are installed
    try:
        import networkx
        import spacy
        graph_packages_available = True
    except ImportError:
        print("Note: networkx or spacy not installed. Some graph functionality will be limited.")
        graph_packages_available = False
    
    context_manager = AdaptiveContextManager(config)
    
    try:
        # Set up system message
        context_manager.add_message(
            "system",
            "You are a helpful AI assistant with advanced reasoning capabilities."
        )
        
        # Add some knowledge to the system
        print("\nAdding knowledge to the system...")
        facts = [
            "Tokyo is the capital of Japan and has a population of approximately 14 million.",
            "Mount Fuji is the tallest mountain in Japan, standing at 3,776 meters.",
            "Kyoto was the imperial capital of Japan for over a thousand years before Tokyo.",
            "Sushi is a traditional Japanese dish consisting of vinegared rice with various toppings.",
            "The Shinkansen is Japan's high-speed railway network, with trains reaching speeds of 320 km/h.",
            "Cherry blossoms (sakura) are an important cultural symbol in Japan.",
            "Japan consists of four main islands: Honshu, Hokkaido, Kyushu, and Shikoku.",
            "The current emperor of Japan is Emperor Naruhito, who ascended to the throne in 2019.",
            "Baseball is one of the most popular sports in Japan.",
            "Mario, Pokemon, and Nintendo are all Japanese creations."
        ]
        
        for fact in facts:
            if graph_packages_available:
                context_manager.remember_knowledge(fact)
                print(f"Added fact: {fact}")
        
        # Test with simple query
        print("\n[Test] Simple Query")
        query = "What is the capital of Japan?"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Test with complex query requiring multi-hop reasoning
        print("\n[Test] Complex Query with Multi-hop Reasoning")
        query = "Is the tallest mountain in Japan located on the same island as the capital city?"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Test with query requiring reasoning over multiple pieces of information
        print("\n[Test] Integration Query")
        query = "What traditional Japanese cultural elements are associated with the city that was the imperial capital before Tokyo?"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
    finally:
        # Clean up
        context_manager.close()
        print("\nChain of Agents test completed")

if __name__ == "__main__":
    test_chain_of_agents() 