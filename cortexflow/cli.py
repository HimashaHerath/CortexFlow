#!/usr/bin/env python3
"""
Command line interface for CortexFlow.
"""

import argparse
import logging
import sys

from cortexflow import CortexFlowManager, CortexFlowConfig
from cortexflow.version import __version__

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CortexFlow CLI - Cognitive-inspired memory optimization for LLMs"
    )
    parser.add_argument(
        "--version", action="version", version=f"CortexFlow {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with CortexFlow")
    chat_parser.add_argument("--model", default="llama3", help="Ollama model to use")
    chat_parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    chat_parser.add_argument("--db", default=":memory:", help="Knowledge store path")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze memory usage")
    analyze_parser.add_argument("--db", required=True, help="Knowledge store path to analyze")
    
    # Execute the command
    args = parser.parse_args()
    
    if args.command == "chat":
        run_chat(args)
    elif args.command == "analyze":
        run_analyze(args)
    else:
        parser.print_help()
        return 1
        
    return 0

def run_chat(args):
    """Run interactive chat."""
    config = CortexFlowConfig(
        ollama_host=args.host,
        default_model=args.model,
        knowledge_store_path=args.db
    )
    
    manager = CortexFlowManager(config)
    
    print(f"CortexFlow Chat (model: {args.model}, db: {args.db})")
    print("Type 'exit' to quit, 'stats' to see memory stats")
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                break
                
            # Check for stats command
            if user_input.lower() == "stats":
                stats = manager.get_stats()
                print("\n--- Memory Stats ---")
                if "memory" in stats:
                    memory_stats = stats["memory"]
                    print(f"Message count: {memory_stats.get('message_count', 0)}")
                    if "tiers" in memory_stats:
                        tiers = memory_stats["tiers"]
                        for tier_name, tier_stats in tiers.items():
                            print(f"{tier_name.capitalize()}: {tier_stats.get('used', 0)}/{tier_stats.get('limit', 0)} tokens ({tier_stats.get('fullness', 0)*100:.1f}%)")
                if "dynamic_weighting" in stats:
                    print("\n--- Dynamic Weighting ---")
                    dw_stats = stats["dynamic_weighting"]
                    if dw_stats.get("enabled", False):
                        print(f"Enabled: {dw_stats.get('enabled', False)}")
                        if "current_weights" in dw_stats:
                            weights = dw_stats["current_weights"]
                            for tier, weight in weights.items():
                                print(f"{tier.capitalize()}: {weight*100:.1f}%")
                continue
                
            # Generate response
            print("\nCortexFlow: ", end="", flush=True)
            
            # Use streaming if possible
            try:
                for chunk in manager.generate_response_stream(user_input):
                    print(chunk, end="", flush=True)
                print()  # Add newline after response
            except (AttributeError, NotImplementedError):
                # Fall back to non-streaming if streaming not available
                response = manager.generate_response(user_input)
                print(response)
    finally:
        manager.close()

def run_analyze(args):
    """Analyze memory usage from a knowledge store."""
    try:
        config = CortexFlowConfig(
            knowledge_store_path=args.db
        )
        
        manager = CortexFlowManager(config)
        
        stats = manager.get_stats()
        
        print(f"CortexFlow Analysis - DB: {args.db}")
        print("\n=== Memory Usage ===")
        
        if "memory" in stats:
            memory_stats = stats["memory"]
            print(f"Total messages: {memory_stats.get('message_count', 0)}")
            
            if "tiers" in memory_stats:
                tiers = memory_stats["tiers"]
                print("\nMemory Tiers:")
                for tier_name, tier_stats in tiers.items():
                    print(f"- {tier_name.capitalize()}:")
                    print(f"  - Used: {tier_stats.get('used', 0)} tokens")
                    print(f"  - Limit: {tier_stats.get('limit', 0)} tokens")
                    print(f"  - Segments: {tier_stats.get('segment_count', 0)}")
                    print(f"  - Fullness: {tier_stats.get('fullness', 0)*100:.1f}%")
        
        # Display knowledge statistics if available
        if hasattr(manager, 'knowledge_store'):
            print("\n=== Knowledge Store ===")
            if hasattr(manager.knowledge_store, 'get_stats'):
                knowledge_stats = manager.knowledge_store.get_stats()
                print(f"Facts: {knowledge_stats.get('fact_count', 'N/A')}")
                print(f"Knowledge items: {knowledge_stats.get('knowledge_count', 'N/A')}")
                print(f"Vector search enabled: {knowledge_stats.get('vector_enabled', False)}")
                print(f"BM25 search enabled: {knowledge_stats.get('bm25_enabled', False)}")
            else:
                print("Knowledge statistics not available")
        
        print("\nAnalysis complete.")
    except Exception as e:
        print(f"Error analyzing database: {e}")
        return 1
    finally:
        if 'manager' in locals():
            manager.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 