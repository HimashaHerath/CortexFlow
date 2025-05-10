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
    args = parser.parse_args()
    
    if not args.skip_check and not test_ollama_available(args.host, args.model):
        print("Ollama test skipped - Ollama not available or model not found")
        return
    
    print(f"Testing AdaptiveContext with Ollama using model {args.model}...")
    
    # Initialize with in-memory storage
    config = AdaptiveContextConfig(
        active_tier_tokens=1000,
        working_tier_tokens=2000,
        archive_tier_tokens=3000,
        knowledge_store_path=":memory:",
        ollama_host=args.host,
        default_model=args.model
    )
    
    context_manager = AdaptiveContextManager(config)
    
    try:
        # Add system message
        context_manager.add_message(
            "You are a helpful AI assistant with advanced memory capabilities.", 
            segment_type="system"
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
            context_manager.add_message(message, segment_type="user")
            
            # Get full context
            full_context = context_manager.get_full_context()
            
            # Send to Ollama
            start_time = time.time()
            try:
                response = requests.post(
                    f"{args.host}/api/generate",
                    json={
                        "model": args.model,
                        "prompt": full_context,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result.get("response", "")
                    
                    # Print response
                    print(f"[Test] Assistant: {assistant_response}")
                    
                    # Add to context
                    context_manager.add_message(assistant_response, segment_type="assistant")
                    
                    # Print timing
                    end_time = time.time()
                    print(f"Response time: {end_time - start_time:.2f}s")
                    
                    # Print memory stats every few messages
                    if i % 3 == 0:
                        stats = context_manager.get_stats()
                        print("\nMemory stats:")
                        print(f"Active: {stats['active_tier']['tokens']}/{stats['active_tier']['capacity']} tokens")
                        print(f"Working: {stats['working_tier']['tokens']}/{stats['working_tier']['capacity']} tokens")
                        print(f"Archive: {stats['archive_tier']['tokens']}/{stats['archive_tier']['capacity']} tokens")
                else:
                    print(f"Error from Ollama: {response.status_code} - {response.text}")
                
            except Exception as e:
                print(f"Error communicating with Ollama: {e}")
        
        # Test forgetting and remembering
        print("\n\nTesting memory management...")
        
        # Flush memory
        print("\n[Test] Flushing memory...")
        context_manager.flush()
        
        # Try to remember previously stated info
        print("\n[Test] User: What's my name and where do I live?")
        context_manager.add_message("What's my name and where do I live?", segment_type="user")
        
        full_context = context_manager.get_full_context()
        print(f"\nContext after flush: {len(full_context)} chars")
        
        try:
            response = requests.post(
                f"{args.host}/api/generate",
                json={
                    "model": args.model,
                    "prompt": full_context,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get("response", "")
                print(f"[Test] Assistant: {assistant_response}")
                context_manager.add_message(assistant_response, segment_type="assistant")
            else:
                print(f"Error from Ollama: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
        
        # Explicitly remember a fact
        print("\n[Test] Explicitly remembering fact...")
        context_manager.explicitly_remember("Alice lives in Boston and has a dog named Max")
        
        # Ask again
        print("\n[Test] User: What's my name and where do I live?")
        context_manager.add_message("What's my name and where do I live?", segment_type="user")
        
        full_context = context_manager.get_full_context()
        
        try:
            response = requests.post(
                f"{args.host}/api/generate",
                json={
                    "model": args.model,
                    "prompt": full_context,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get("response", "")
                print(f"[Test] Assistant: {assistant_response}")
            else:
                print(f"Error from Ollama: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
    
    finally:
        # Clean up
        context_manager.close()
        print("\nOllama integration test completed")

if __name__ == "__main__":
    test_with_ollama() 