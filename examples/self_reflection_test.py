#!/usr/bin/env python3
"""
AdaptiveContext Self-Reflection Test

This script tests the Self-Reflection functionality in AdaptiveContext.
"""

import argparse
import json
import time
from adaptive_context import CortexFlowManager, CortexFlowConfig

def test_self_reflection():
    """Test the Self-Reflection functionality."""
    parser = argparse.ArgumentParser(description="Test AdaptiveContext Self-Reflection")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--active-tokens", type=int, default=1000, help="Active tier token limit")
    parser.add_argument("--working-tokens", type=int, default=2000, help="Working tier token limit")
    parser.add_argument("--archive-tokens", type=int, default=3000, help="Archive tier token limit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    print(f"Testing AdaptiveContext Self-Reflection with model {args.model}...")
    
    # Initialize with in-memory storage and Self-Reflection enabled
    config = CortexFlowConfig(
        active_token_limit=args.active_tokens,
        working_token_limit=args.working_tokens,
        archive_token_limit=args.archive_tokens,
        knowledge_store_path=":memory:",
        ollama_host=args.host,
        default_model=args.model,
        use_self_reflection=True,  # Enable Self-Reflection
        reflection_relevance_threshold=0.6,  # Minimum relevance score
        reflection_confidence_threshold=0.7,  # Minimum confidence for consistency
        verbose_logging=args.verbose
    )
    
    context_manager = CortexFlowManager(config)
    
    try:
        # Set up system message
        context_manager.add_message(
            "system",
            "You are a helpful AI assistant with advanced reasoning capabilities."
        )
        
        # Add some knowledge to the system
        print("\nAdding knowledge to the system...")
        facts = [
            "The Eiffel Tower is 330 meters tall and was completed in 1889.",
            "The Great Wall of China is over 21,000 kilometers long.",
            "Mount Everest is the tallest mountain on Earth at 8,848 meters.",
            "The Pacific Ocean is the largest and deepest ocean on Earth.",
            "The human brain contains approximately 86 billion neurons.",
            "The Amazon rainforest produces about 20% of Earth's oxygen.",
            "The speed of light in vacuum is exactly 299,792,458 meters per second.",
            "The Sahara Desert covers an area of over 9 million square kilometers.",
            "Jupiter is the largest planet in our solar system.",
            "Water covers approximately 71% of the Earth's surface."
        ]
        
        # Add facts with deliberate contradictions
        contradictory_facts = [
            "The Eiffel Tower is 300 meters tall and was completed in 1899.",  # Contradicts first fact
            "Mount Everest is the tallest mountain on Earth at 8,849 meters.",  # Slight contradiction
            "The human brain contains approximately 100 billion neurons.",  # Contradicts fifth fact
        ]
        
        # Add original facts
        for fact in facts:
            context_manager.remember_knowledge(fact)
            print(f"Added fact: {fact}")
            
        # Add contradictory facts
        print("\nAdding contradictory facts...")
        for fact in contradictory_facts:
            context_manager.remember_knowledge(fact)
            print(f"Added contradictory fact: {fact}")
        
        # Test with queries that should trigger self-reflection
        
        # Test 1: Knowledge relevance verification
        print("\n[Test 1] Knowledge Relevance Verification")
        query = "What is the height of the Eiffel Tower?"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Test 2: Consistency checking and correction
        print("\n[Test 2] Consistency Check and Correction")
        query = "How many neurons are in the human brain?"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Test 3: Handling slight contradictions
        print("\n[Test 3] Handling Slight Contradictions")
        query = "What is the height of Mount Everest?"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Test 4: Unrelated query with no contradictions
        print("\n[Test 4] Query Without Contradictions")
        query = "What is the largest ocean on Earth?"
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
        print("\nSelf-Reflection test completed")

if __name__ == "__main__":
    test_self_reflection() 