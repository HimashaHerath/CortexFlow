#!/usr/bin/env python3
"""
AdaptiveContext Dynamic Memory Weighting Test

This script tests the Dynamic Memory Weighting functionality in AdaptiveContext.
"""

import argparse
import json
import time
import matplotlib.pyplot as plt
from adaptive_context import CortexFlowManager, CortexFlowConfig

def test_dynamic_weighting():
    """Test the Dynamic Memory Weighting functionality."""
    parser = argparse.ArgumentParser(description="Test AdaptiveContext Dynamic Memory Weighting")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--active-tokens", type=int, default=4000, help="Initial active tier token limit")
    parser.add_argument("--working-tokens", type=int, default=8000, help="Initial working tier token limit")
    parser.add_argument("--archive-tokens", type=int, default=12000, help="Initial archive tier token limit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--plot", action="store_true", help="Generate plots of tier allocations")
    args = parser.parse_args()
    
    print(f"Testing AdaptiveContext Dynamic Memory Weighting with model {args.model}...")
    
    # Initialize with in-memory storage and Dynamic Weighting enabled
    config = CortexFlowConfig(
        active_token_limit=args.active_tokens,
        working_token_limit=args.working_tokens,
        archive_token_limit=args.archive_tokens,
        knowledge_store_path=":memory:",
        ollama_host=args.host,
        default_model=args.model,
        use_dynamic_weighting=True,  # Enable Dynamic Weighting
        dynamic_weighting_learning_rate=0.2,  # Faster learning for test purposes
        verbose_logging=args.verbose
    )
    
    context_manager = CortexFlowManager(config)
    
    try:
        # Set up system message
        context_manager.add_message(
            "system",
            "You are a helpful AI assistant with dynamic memory capabilities."
        )
        
        # Test with different query types to observe dynamic weighting
        
        # Test 1: Simple query (should reduce active tier allocation)
        print("\n[Test 1] Simple Query")
        query = "What time is it?"
        print(f"User: {query}")
        
        # Get initial stats
        initial_stats = context_manager.get_stats()
        print("Initial memory allocation:")
        print_tier_stats(initial_stats)
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Get updated stats
        updated_stats = context_manager.get_stats()
        print("Updated memory allocation after simple query:")
        print_tier_stats(updated_stats)
        
        # Test 2: Complex query (should increase active tier allocation)
        print("\n[Test 2] Complex Query")
        query = "Can you explain the relationship between quantum mechanics and general relativity, focusing on the challenges of reconciling these theories in extreme gravitational environments like black holes?"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Get updated stats
        updated_stats = context_manager.get_stats()
        print("Updated memory allocation after complex query:")
        print_tier_stats(updated_stats)
        
        # Test 3: Code-related query (should adjust for code content)
        print("\n[Test 3] Code-Related Query")
        query = "Write a Python function to implement the quicksort algorithm with detailed comments explaining how it works."
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Get updated stats
        updated_stats = context_manager.get_stats()
        print("Updated memory allocation after code query:")
        print_tier_stats(updated_stats)
        
        # Test 4: Data-related query (should adjust for data content)
        print("\n[Test 4] Data-Related Query")
        query = "Can you analyze this JSON data and tell me what insights you can derive from it? ```json\n{\"users\": [{\"id\": 1, \"name\": \"Alice\", \"age\": 28, \"purchases\": 12}, {\"id\": 2, \"name\": \"Bob\", \"age\": 35, \"purchases\": 4}, {\"id\": 3, \"name\": \"Charlie\", \"age\": 42, \"purchases\": 18}]}\n```"
        print(f"User: {query}")
        
        context_manager.add_message("user", query)
        start_time = time.time()
        response = context_manager.generate_response()
        end_time = time.time()
        
        print(f"Assistant: {response}")
        print(f"Response time: {end_time - start_time:.2f}s")
        
        # Get updated stats
        updated_stats = context_manager.get_stats()
        print("Updated memory allocation after data query:")
        print_tier_stats(updated_stats)
        
        # Test 5: Multiple related queries (should adapt based on history)
        print("\n[Test 5] Multiple Related Queries")
        
        queries = [
            "What are the main features of Python?",
            "How does Python handle memory management?",
            "Can you explain Python's GIL (Global Interpreter Lock)?",
            "What are the differences between Python 2 and Python 3?",
            "How can I optimize Python code for better performance?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            
            context_manager.add_message("user", query)
            start_time = time.time()
            response = context_manager.generate_response()
            end_time = time.time()
            
            print(f"Assistant: {response}")
            print(f"Response time: {end_time - start_time:.2f}s")
            
            # Get updated stats
            updated_stats = context_manager.get_stats()
            print(f"Updated memory allocation after query {i}:")
            print_tier_stats(updated_stats)
            
            # Short pause between queries
            time.sleep(1)
        
        # Get final dynamic weighting stats
        dynamic_stats = context_manager.get_dynamic_weighting_stats()
        print("\nFinal Dynamic Weighting Stats:")
        print(json.dumps(dynamic_stats, indent=2))
        
        # Generate plots if requested
        if args.plot:
            plot_tier_allocations(context_manager)
        
    finally:
        # Clean up
        context_manager.close()
        print("\nDynamic Memory Weighting test completed")

def print_tier_stats(stats):
    """Print memory tier statistics in a readable format."""
    if not stats or 'memory' not in stats or 'tiers' not in stats['memory']:
        print("No tier statistics available")
        return
    
    tiers = stats['memory']['tiers']
    
    print(f"  Active tier: {tiers['active']['used']}/{tiers['active']['limit']} tokens "
          f"({tiers['active']['fullness']*100:.1f}% full)")
    print(f"  Working tier: {tiers['working']['used']}/{tiers['working']['limit']} tokens "
          f"({tiers['working']['fullness']*100:.1f}% full)")
    print(f"  Archive tier: {tiers['archive']['used']}/{tiers['archive']['limit']} tokens "
          f"({tiers['archive']['fullness']*100:.1f}% full)")

def plot_tier_allocations(context_manager):
    """Generate plots of tier allocations over time."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import PercentFormatter
        
        # Get dynamic weighting stats
        stats = context_manager.get_dynamic_weighting_stats()
        if not stats or not stats.get('enabled', False):
            print("Dynamic weighting stats not available for plotting")
            return
        
        # Get memory stats for tier usage history
        memory_stats = context_manager.get_stats().get('memory', {}).get('tier_stats', {})
        
        # Plot tier weights over time
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Current tier weights
        plt.subplot(2, 2, 1)
        weights = stats.get('current_weights', {})
        labels = list(weights.keys())
        sizes = [weights[k] for k in labels]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Current Tier Weight Distribution')
        
        # Plot 2: Current tier limits
        plt.subplot(2, 2, 2)
        limits = stats.get('current_limits', {})
        tiers = list(limits.keys())
        values = [limits[k] for k in tiers]
        plt.bar(tiers, values)
        plt.title('Current Tier Token Limits')
        plt.ylabel('Tokens')
        
        # Plot 3: Document type distribution
        plt.subplot(2, 2, 3)
        doc_types = stats.get('document_type_distribution', {})
        types = list(doc_types.keys())
        counts = [doc_types[k] for k in types]
        plt.bar(types, counts)
        plt.title('Document Type Distribution')
        plt.ylabel('Count')
        
        # Plot 4: Query complexity over time
        plt.subplot(2, 2, 4)
        complexity = stats.get('recent_query_complexity', [])
        if complexity:
            plt.plot(complexity, marker='o')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            plt.title('Recent Query Complexity')
            plt.ylabel('Complexity Score')
            plt.xlabel('Query Index')
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('dynamic_weighting_stats.png')
        print("Saved plot to dynamic_weighting_stats.png")
        
        # Plot tier usage history if available
        if memory_stats:
            plt.figure(figsize=(12, 6))
            
            for i, tier in enumerate(['active', 'working', 'archive']):
                if tier in memory_stats and 'usage_history' in memory_stats[tier]:
                    history = memory_stats[tier]['usage_history']
                    if not history:
                        continue
                        
                    # Extract data
                    timestamps = [entry.get('timestamp', 0) for entry in history]
                    # Convert to relative time in minutes
                    if timestamps:
                        start_time = min(timestamps)
                        times = [(t - start_time) / 60 for t in timestamps]
                    else:
                        times = []
                        
                    fullness = [entry.get('fullness_ratio', 0) for entry in history]
                    
                    # Plot fullness ratio over time
                    plt.plot(times, fullness, marker='o', label=f"{tier.capitalize()} Tier")
            
            plt.title('Memory Tier Fullness Over Time')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Fullness Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            
            plt.tight_layout()
            plt.savefig('tier_usage_history.png')
            print("Saved plot to tier_usage_history.png")
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    test_dynamic_weighting() 