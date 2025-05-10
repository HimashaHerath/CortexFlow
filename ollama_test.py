#!/usr/bin/env python3
"""
AdaptiveContext Ollama Integration Test

This script tests the integration between AdaptiveContext and Ollama.
"""

import time
import json
import requests
import argparse
from adaptive_context import AdaptiveContextManager, AdaptiveContextConfig

def test_ollama_available(host="http://localhost:11434", model="llama3"):
    """Check if Ollama is available with the specified model."""
    try:
        response = requests.get(f"{host}/api/tags")
        if response.status_code != 200:
            print(f"Error connecting to Ollama: {response.status_code}")
            return False
            
        models = [tag["name"] for tag in response.json().get("models", [])]
        if model not in models:
            print(f"Model {model} not found. Available models: {', '.join(models)}")
            return False
            
        return True
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return False

def test_with_ollama():
    parser = argparse.ArgumentParser(description="Test AdaptiveContext with Ollama")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--skip-check", action="store_true", help="Skip checking for Ollama availability")
    parser.add_argument("--active-tokens", type=int, default=1000, help="Active tier token limit")
    parser.add_argument("--working-tokens", type=int, default=2000, help="Working tier token limit")
    parser.add_argument("--archive-tokens", type=int, default=3000, help="Archive tier token limit")
    args = parser.parse_args()
    
    if not args.skip_check and not test_ollama_available(args.host, args.model):
        print("Ollama test skipped - Ollama not available or model not found")
        return
    
    print(f"Testing AdaptiveContext with Ollama using model {args.model}...")
    
    # Initialize with in-memory storage
    config = AdaptiveContextConfig(
        active_token_limit=args.active_tokens,
        working_token_limit=args.working_tokens,
        archive_token_limit=args.archive_tokens,
        knowledge_store_path=":memory:",
        ollama_host=args.host,
        default_model=args.model,
        use_graph_rag=True  # Enable graph RAG
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
        # Add system message
        context_manager.add_message(
            "system",
            "You are a helpful AI assistant with advanced memory capabilities."
        )
        
        # Test conversation with memory
        test_messages = [
            "My name is Alice and I live in Boston.",
            "I have a dog named Max.",
            "My favorite color is purple.",
            "Can you tell me some facts about Boston?",
            "Tell me another fact about my city.",
            "What was my dog's name again?",
            "What's my favorite color?"
        ]
        
        for i, message in enumerate(test_messages):
            print(f"\n[Test] User: {message}")
            
            # Add message to context
            context_manager.add_message("user", message)
            
            # Generate response using the manager
            start_time = time.time()
            try:
                assistant_response = context_manager.generate_response()
                
                # Print response
                print(f"[Test] Assistant: {assistant_response}")
                
                # Print timing
                end_time = time.time()
                print(f"Response time: {end_time - start_time:.2f}s")
                
                # Print memory stats every few messages
                if i % 3 == 0:
                    print("\nMemory stats not available in current implementation.")
                
            except Exception as e:
                print(f"Error communicating with Ollama: {e}")
        
        # Test forgetting and remembering
        print("\n\nTesting memory management...")
        
        # Flush memory
        print("\n[Test] Flushing memory...")
        context_manager.clear_memory()
        
        # Try to remember previously stated info
        print("\n[Test] User: What's my name and where do I live?")
        context_manager.add_message("user", "What's my name and where do I live?")
        
        try:
            assistant_response = context_manager.generate_response()
            print(f"[Test] Assistant: {assistant_response}")
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
        
        # Explicitly remember a fact
        print("\n[Test] Explicitly remembering fact...")
        try:
            if graph_packages_available:
                context_manager.remember_knowledge("Alice lives in Boston and has a dog named Max")
            else:
                print("Skipping graph-based knowledge storage due to missing packages")
        except AttributeError as e:
            print(f"Note: Graph functionality error - {e}")
            print("Continuing with test...")
        
        # Ask again
        print("\n[Test] User: What's my name and where do I live?")
        context_manager.add_message("user", "What's my name and where do I live?")
        
        try:
            assistant_response = context_manager.generate_response()
            print(f"[Test] Assistant: {assistant_response}")
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
    
    finally:
        # Clean up
        context_manager.close()
        print("\nOllama integration test completed")

if __name__ == "__main__":
    test_with_ollama() 