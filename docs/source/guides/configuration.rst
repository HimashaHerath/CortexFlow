Configuration
=============

CortexFlow offers flexible configuration options to adapt the system to your needs.

Basic Configuration
----------------

The primary way to configure CortexFlow is through the ``CortexFlowConfig`` class:

.. code-block:: python

    from cortexflow import CortexFlowManager, CortexFlowConfig
    
    config = CortexFlowConfig(
        active_token_limit=2000,
        working_token_limit=4000,
        archive_token_limit=6000,
        default_model="llama3",
        importance_threshold=0.6,
        use_compression=True
    )
    
    manager = CortexFlowManager(config)

Memory Tier Configuration
----------------------

You can adjust the size of each memory tier:

.. code-block:: python

    config = CortexFlowConfig(
        # Memory tier token limits
        active_token_limit=2000,  # Most recent/important context
        working_token_limit=4000,  # Medium-term storage
        archive_token_limit=6000,  # Long-term storage
    )

Model Configuration
----------------

Configure which LLM to use:

.. code-block:: python

    config = CortexFlowConfig(
        # Model settings
        default_model="llama3",  # Base model for general responses
        compression_model="llama3",  # Model for summary generation
        importance_model="llama3-instruct",  # Model for importance classification
    )

Advanced Features Configuration
---------------------------

Enable or disable specific features:

.. code-block:: python

    config = CortexFlowConfig(
        # Feature toggles
        use_compression=True,  # Enable progressive compression
        use_importance_classification=True,  # Use importance scoring
        use_vector_store=True,  # Enable vector-based retrieval
        use_chain_of_agents=True,  # Enable Chain of Agents framework
        use_self_reflection=True,  # Enable self-verification
        
        # Feature-specific settings
        compression_ratio=0.6,  # Target compression level
        importance_threshold=0.7,  # Minimum importance score
        chain_complexity_threshold=5,  # When to use Chain of Agents
        knowledge_db_path="knowledge.db"  # Path to persistent storage
    )

Persistence Configuration
---------------------

Configure how knowledge is stored and retrieved:

.. code-block:: python

    config = CortexFlowConfig(
        # Knowledge store settings
        knowledge_db_path="cortexflow.db",
        vector_distance_threshold=0.75,
        max_knowledge_results=10,
        
        # GraphRAG settings
        use_graph_rag=True,
        graph_db_path="graph_store.db",
        entity_extraction_threshold=0.6
    ) 