CortexFlowManager
================

The CortexFlowManager is the main entry point for the CortexFlow library. It manages the multi-tier memory system and provides methods for adding messages, generating responses, and managing knowledge.

.. autoclass:: cortexflow.CortexFlowManager
   :members:
   :undoc-members:
   :show-inheritance:

Core Methods
----------

.. automethod:: cortexflow.CortexFlowManager.__init__
.. automethod:: cortexflow.CortexFlowManager.add_message
.. automethod:: cortexflow.CortexFlowManager.generate_response
.. automethod:: cortexflow.CortexFlowManager.remember_knowledge
.. automethod:: cortexflow.CortexFlowManager.recall_knowledge
.. automethod:: cortexflow.CortexFlowManager.close

Example Usage
-----------

.. code-block:: python

    from cortexflow import CortexFlowManager, CortexFlowConfig
    
    # Create a configuration
    config = CortexFlowConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000
    )
    
    # Initialize the manager
    manager = CortexFlowManager(config)
    
    # Add messages
    manager.add_message("system", "You are a helpful assistant.")
    manager.add_message("user", "What's the capital of France?")
    
    # Generate a response
    response = manager.generate_response()
    print(f"Assistant: {response}")
    
    # Close the manager when done
    manager.close() 