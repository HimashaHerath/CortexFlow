CortexFlowConfig
===============

The CortexFlowConfig class holds configuration settings for the CortexFlow system.

.. autoclass:: cortexflow.CortexFlowConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Parameters
---------------------

Memory Configuration
~~~~~~~~~~~~~~~~

* ``active_token_limit`` - Maximum tokens in the active memory tier
* ``working_token_limit`` - Maximum tokens in the working memory tier
* ``archive_token_limit`` - Maximum tokens in the archive memory tier

Model Configuration
~~~~~~~~~~~~~~~

* ``default_model`` - Model used for generating responses
* ``compression_model`` - Model used for memory compression
* ``importance_model`` - Model used for importance classification

Feature Toggles
~~~~~~~~~~~~

* ``use_compression`` - Enable/disable progressive compression
* ``use_importance_classification`` - Enable/disable importance scoring
* ``use_vector_store`` - Enable/disable vector-based retrieval
* ``use_chain_of_agents`` - Enable/disable Chain of Agents framework
* ``use_self_reflection`` - Enable/disable self-verification

Advanced Settings
~~~~~~~~~~~~~~

* ``compression_ratio`` - Target compression level
* ``importance_threshold`` - Minimum importance score
* ``chain_complexity_threshold`` - When to use Chain of Agents
* ``knowledge_db_path`` - Path to persistent storage
* ``vector_distance_threshold`` - Similarity threshold for vector retrieval
* ``max_knowledge_results`` - Maximum items to retrieve from knowledge store

Example Usage
-----------

.. code-block:: python

    from cortexflow import CortexFlowConfig
    
    # Create a basic configuration
    config = CortexFlowConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000,
        default_model="llama3"
    )
    
    # Advanced configuration
    advanced_config = CortexFlowConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000,
        default_model="llama3",
        compression_model="llama3",
        importance_model="llama3-instruct",
        use_compression=True,
        compression_ratio=0.6,
        use_importance_classification=True,
        importance_threshold=0.7,
        use_vector_store=True,
        knowledge_db_path="cortexflow.db",
        vector_distance_threshold=0.75,
        max_knowledge_results=10,
        use_chain_of_agents=True,
        chain_complexity_threshold=5,
        use_self_reflection=True
    ) 