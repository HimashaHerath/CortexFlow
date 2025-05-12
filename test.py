#!/usr/bin/env python3
"""
AdaptiveContext Test Script

This script performs basic testing of the AdaptiveContext functionality.
"""

import time
import json
from adaptive_context import CortexFlowManager, CortexFlowConfig

def test_basic_functionality():
    print("Testing basic AdaptiveContext functionality...")
    
    # Initialize with in-memory storage
    config = CortexFlowConfig(
        active_tier_tokens=500,
        working_tier_tokens=1000,
        archive_tier_tokens=1500,
        knowledge_store_path="test_memory.db"
    )
    
    context_manager = CortexFlowManager(config)
    
    # Test adding messages
    print("\nAdding messages to context...")
    context_manager.add_message("This is the first message", segment_type="user")
    context_manager.add_message("This is the response to the first message", segment_type="assistant")
    
    # Add more messages to test tier management
    print("\nAdding multiple messages to test tier management...")
    for i in range(10):
        context_manager.add_message(f"User message {i}: This is a test message with some content to take up space in the context window", segment_type="user")
        context_manager.add_message(f"Assistant response {i}: This is a response with enough content to trigger memory management between tiers", segment_type="assistant")
    
    # Get and display stats
    stats = context_manager.get_stats()
    print("\nMemory stats after adding messages:")
    print(json.dumps(stats, indent=2))
    
    # Test importance classification
    print("\nAdding high importance message...")
    context_manager.add_message("IMPORTANT: Remember that my name is John and I live in New York!", segment_type="user")
    context_manager.add_message("I'll remember that your name is John and you live in New York.", segment_type="assistant")
    
    # Test explicit memory storage
    print("\nTesting explicit memory storage...")
    context_manager.explicitly_remember("John's favorite color is blue")
    
    # Get full context
    full_context = context_manager.get_full_context()
    print(f"\nFull context ({len(full_context)} chars):")
    print(f"First 500 chars: {full_context[:500]}...")
    
    # Test memory flushing
    print("\nTesting memory flush...")
    context_manager.flush()
    
    stats_after_flush = context_manager.get_stats()
    print("\nMemory stats after flush:")
    print(json.dumps(stats_after_flush, indent=2))
    
    # Test knowledge retrieval after flush
    print("\nTesting knowledge retrieval after flush...")
    context_manager.add_message("What's my name and where do I live?", segment_type="user")
    full_context_after_flush = context_manager.get_full_context()
    print(f"\nContext after flush ({len(full_context_after_flush)} chars):")
    print(full_context_after_flush)
    
    # Clean up
    context_manager.close()
    print("\nTest completed successfully")

if __name__ == "__main__":
    test_basic_functionality() 