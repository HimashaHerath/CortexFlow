Usage
=====

Here's a simple example of using CortexFlow:

.. code-block:: python

   from cortexflow import CortexFlowManager, CortexFlowConfig

   # Create a configuration
   config = CortexFlowConfig(
       model_name="llama3",
       vector_dimensions=1536,
       api_base="http://localhost:11434/api"
   )

   # Initialize the manager
   manager = CortexFlowManager(config)

   # Use the manager for LLM interactions
   response = manager.generate("Tell me about machine learning")
   print(response)

For more detailed examples, see the examples directory in the repository. 