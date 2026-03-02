Usage
=====

Here's a simple example of using CortexFlow:

.. code-block:: python

   from cortexflow import CortexFlowManager, CortexFlowConfig, MemoryConfig, LLMConfig

   # Create a configuration using nested config
   config = CortexFlowConfig(
       memory=MemoryConfig(active_token_limit=4096),
       llm=LLMConfig(default_model="llama3", backend="ollama"),
   )

   # Initialize the manager
   manager = CortexFlowManager(config)

   # Add messages and generate a response
   manager.add_message("system", "You are a helpful AI assistant.")
   manager.add_message("user", "Tell me about machine learning")
   response = manager.generate_response()
   print(f"Assistant: {response}")

   # Clean up
   manager.close()

For more detailed examples, see the examples directory in the repository.
