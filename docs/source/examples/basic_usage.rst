Basic Usage
==========

This example demonstrates basic usage of CortexFlow.

Simple Conversation
-----------------

.. code-block:: python

    from cortexflow import CortexFlowManager, CortexFlowConfig

    # Configure with default settings
    config = CortexFlowConfig()
    
    # Create the context manager
    manager = CortexFlowManager(config)
    
    # Add system message to define assistant behavior
    manager.add_message("system", "You are a helpful AI assistant.")
    
    # Add user message
    manager.add_message("user", "What is the capital of France?")
    
    # Generate a response
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Continue the conversation
    manager.add_message("user", "What's the population of Paris?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Clean up when done
    manager.close()

Multi-turn Conversation
---------------------

.. code-block:: python

    from cortexflow import CortexFlowManager, CortexFlowConfig

    # Configure with custom settings
    config = CortexFlowConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000,
        default_model="llama3"
    )
    
    # Create the context manager
    manager = CortexFlowManager(config)
    
    # Start a multi-turn conversation
    manager.add_message("system", "You are a helpful assistant specializing in geography.")
    
    # First question
    manager.add_message("user", "What's the capital of Japan?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Second question builds on previous context
    manager.add_message("user", "What's the population of that city?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Third question tests memory
    manager.add_message("user", "What's the main airport serving this city?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Fourth question changes topic
    manager.add_message("user", "What's the capital of Australia?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Fifth question should rely on memory compression
    manager.add_message("user", "Let's go back to Japan. What's the name of their parliament?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Close the manager
    manager.close()

Explicitly Managing Knowledge
--------------------------

.. code-block:: python

    from cortexflow import CortexFlowManager, CortexFlowConfig

    config = CortexFlowConfig(
        knowledge_db_path="user_knowledge.db"
    )
    manager = CortexFlowManager(config)
    
    # Add system context
    manager.add_message("system", "You are a helpful assistant.")
    
    # Explicitly remember important user information
    manager.remember_knowledge("The user's name is Alice Smith.")
    manager.remember_knowledge("Alice lives in Boston, Massachusetts.")
    manager.remember_knowledge("Alice is allergic to peanuts.")
    manager.remember_knowledge("Alice's favorite color is purple.")
    
    # Query that should use the stored knowledge
    manager.add_message("user", "Can you recommend some restaurants near me?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Another query that should use stored knowledge
    manager.add_message("user", "What desserts should I avoid?")
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Clean up
    manager.close() 