#!/usr/bin/env python3
"""
CortexFlow Example Usage

This script demonstrates how to use the CortexFlow system with Ollama.
"""

import argparse
import json
import time

import requests

from cortexflow import (
    CortexFlowConfig,
    CortexFlowManager,
    KnowledgeStoreConfig,
    LLMConfig,
    MemoryConfig,
)


def send_to_ollama(prompt, model="llama3", ollama_host="http://localhost:11434"):
    """Send a prompt to Ollama API and return the response."""
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )

        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error communicating with Ollama: {e}"


def main():
    parser = argparse.ArgumentParser(description="CortexFlow example")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama API host"
    )
    parser.add_argument(
        "--active-tokens", type=int, default=2000, help="Active tier token limit"
    )
    parser.add_argument(
        "--working-tokens", type=int, default=4000, help="Working tier token limit"
    )
    parser.add_argument(
        "--archive-tokens", type=int, default=6000, help="Archive tier token limit"
    )
    parser.add_argument(
        "--db", default="memory.db", help="Knowledge store database path"
    )
    args = parser.parse_args()

    # Configure CortexFlow using nested dataclass config
    config = CortexFlowConfig(
        memory=MemoryConfig(
            active_token_limit=args.active_tokens,
            working_token_limit=args.working_tokens,
            archive_token_limit=args.archive_tokens,
        ),
        knowledge_store=KnowledgeStoreConfig(
            knowledge_store_path=args.db,
        ),
        llm=LLMConfig(
            ollama_host=args.host,
            default_model=args.model,
        ),
    )

    # Initialize context manager
    context_manager = CortexFlowManager(config)

    print(f"CortexFlow initialized with model {args.model}")
    print("Type 'exit' to quit, 'stats' to see memory stats, 'clear' to clear context")
    print("Use 'remember: [text]' to explicitly store information")

    # Add system prompt
    system_prompt = """You are a helpful AI assistant with enhanced memory capabilities.
    You can remember important information from earlier in the conversation.
    Your memory is organized into active, working, and long-term tiers."""

    context_manager.add_message("system", system_prompt)

    try:
        while True:
            # Get user input
            user_input = input("\n> ")

            if user_input.lower() == "exit":
                break

            elif user_input.lower() == "stats":
                # Display memory statistics
                stats = context_manager.get_stats()
                print("\nCortexFlow Memory Stats:")
                print(json.dumps(stats, indent=2))
                continue

            elif user_input.lower() == "clear":
                # Clear context memory
                context_manager.clear_memory()
                print("Context memory cleared")
                continue

            elif user_input.lower().startswith("remember:"):
                # Explicitly store information in the knowledge store
                text_to_remember = user_input[9:].strip()
                ids = context_manager.add_knowledge(
                    text_to_remember, source="user_input"
                )
                if ids:
                    print("Information stored in knowledge base")
                else:
                    print("Failed to store information")
                continue

            # Add user message to context
            context_manager.add_message("user", user_input)

            # Get conversation context (messages + relevant knowledge)
            context = context_manager.get_conversation_context()

            # Build a prompt string from the context for Ollama
            prompt_parts = []
            for msg in context.get("messages", []):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")

            # Include any retrieved knowledge
            knowledge_items = context.get("knowledge", [])
            if knowledge_items:
                knowledge_text = "\n".join(
                    item.get("text", item.get("content", ""))
                    for item in knowledge_items
                    if item.get("text") or item.get("content")
                )
                if knowledge_text:
                    prompt_parts.insert(0, f"Relevant knowledge:\n{knowledge_text}\n")

            full_prompt = "\n".join(prompt_parts)

            # Send to Ollama
            start_time = time.time()
            response = send_to_ollama(
                full_prompt, model=args.model, ollama_host=args.host
            )
            end_time = time.time()

            # Add assistant response to context
            context_manager.add_message("assistant", response)

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
