#!/usr/bin/env python3
"""
AdaptiveContext Example Usage

This script demonstrates how to use the AdaptiveContext system with Ollama.
"""

import json
import time
import requests
import argparse

from adaptive_context import CortexFlowManager, CortexFlowConfig

def send_to_ollama(prompt, model="llama3", ollama_host="http://localhost:11434"):
    """Send a prompt to Ollama API and return the response."""
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

def main():
    parser = argparse.ArgumentParser(description="AdaptiveContext example")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--active-tokens", type=int, default=2000, help="Active tier token limit")
    parser.add_argument("--working-tokens", type=int, default=4000, help="Working tier token limit")
    parser.add_argument("--archive-tokens", type=int, default=6000, help="Archive tier token limit")
    parser.add_argument("--db", default="memory.db", help="Knowledge store database path")
    args = parser.parse_args()
    
    # Configure AdaptiveContext
    config = CortexFlowConfig(
        active_tier_tokens=args.active_tokens,
        working_tier_tokens=args.working_tokens,
        archive_tier_tokens=args.archive_tokens,
        knowledge_store_path=args.db,
        ollama_host=args.host,
        default_model=args.model
    )
    
    # Initialize context manager
    context_manager = CortexFlowManager(config)
    
    print(f"AdaptiveContext initialized with model {args.model}")
    print("Type 'exit' to quit, 'stats' to see memory stats, 'flush' to clear context")
    print("Use 'remember: [text]' to explicitly store information")
    
    # Add system prompt
    system_prompt = """You are a helpful AI assistant with enhanced memory capabilities. 
    You can remember important information from earlier in the conversation. 
    Your memory is organized into active, working, and long-term tiers."""
    
    context_manager.add_message(system_prompt, segment_type="system")
    
    try:
        while True:
            # Get user input
            user_input = input("\n> ")
            
            if user_input.lower() == "exit":
                break
                
            elif user_input.lower() == "stats":
                # Display memory statistics
                stats = context_manager.get_stats()
                print("\nAdaptiveContext Memory Stats:")
                print(json.dumps(stats, indent=2))
                continue
                
            elif user_input.lower() == "flush":
                # Clear context
                context_manager.flush()
                print("Context memory flushed")
                continue
                
            elif user_input.lower().startswith("remember:"):
                # Explicitly store information
                text_to_remember = user_input[9:].strip()
                if context_manager.explicitly_remember(text_to_remember):
                    print("Information stored in knowledge base")
                else:
                    print("Failed to store information")
                continue
            
            # Add user message to context
            context_manager.add_message(user_input, segment_type="user")
            
            # Get full context
            full_context = context_manager.get_full_context()
            
            # Send to Ollama
            start_time = time.time()
            response = send_to_ollama(full_context, model=args.model, ollama_host=args.host)
            end_time = time.time()
            
            # Add assistant response to context
            context_manager.add_message(response, segment_type="assistant")
            
            # Print response and timing
            print(f"\n{response}")
            print(f"\n(Response time: {end_time - start_time:.2f}s)")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        # Clean up
        context_manager.close()

if __name__ == "__main__":
    main() 