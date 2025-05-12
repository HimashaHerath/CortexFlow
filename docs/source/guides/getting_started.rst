Getting Started
==============

This guide will help you get started with CortexFlow.

Basic Usage
----------

Here's a simple example of how to use CortexFlow:

.. code-block:: python

    from cortexflow import CortexFlowManager, CortexFlowConfig

    # Configure with custom settings
    config = CortexFlowConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000,
        default_model="llama3"  # Use your preferred Ollama model
    )

    # Create the context manager
    context_manager = CortexFlowManager(config)

    # Add messages to the context
    context_manager.add_message("system", "You are a helpful AI assistant.")
    context_manager.add_message("user", "What is the capital of France?")

    # Generate a response
    response = context_manager.generate_response()
    print(f"Assistant: {response}")

    # Explicitly store important information
    context_manager.remember_knowledge("The user's name is Alice and she lives in Boston.")

    # Clean up when done
    context_manager.close()

Advanced Features
---------------

Chain of Agents
~~~~~~~~~~~~~~

CortexFlow includes a Chain of Agents framework that breaks down complex reasoning into specialized roles:

.. code-block:: python

    # Enable Chain of Agents in your configuration
    config = CortexFlowConfig(
        use_chain_of_agents=True,
        chain_complexity_threshold=5,  # Only use for reasonably complex queries
        # ... other config options
    )

    # Create the manager with this configuration
    manager = CortexFlowManager(config)

    # Add a complex query
    manager.add_message("user", "What connection exists between quantum physics and consciousness?")
    
    # Generate response (automatically uses Chain of Agents for complex queries)
    response = manager.generate_response()

See the :doc:`chain_of_agents` guide for more details. 