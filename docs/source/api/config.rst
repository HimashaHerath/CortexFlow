CortexFlowConfig
===============

The ``CortexFlowConfig`` class holds configuration settings for the CortexFlow system.
It uses a nested dataclass architecture where each subsystem has its own config class.

.. autoclass:: cortexflow.CortexFlowConfig
   :members:
   :undoc-members:
   :show-inheritance:

Sub-Configuration Classes
----------------------

CortexFlow configuration is organized into the following nested sections:

MemoryConfig
~~~~~~~~~~~

Controls the multi-tier memory system.

.. autoclass:: cortexflow.config.MemoryConfig
   :members:
   :undoc-members:

* ``active_token_limit`` - Maximum tokens in the active memory tier (default: 4096)
* ``working_token_limit`` - Maximum tokens in the working memory tier (default: 8192)
* ``archive_token_limit`` - Maximum tokens in the archive memory tier (default: 16384)
* ``use_dynamic_weighting`` - Enable dynamic tier weighting (default: False)

LLMConfig
~~~~~~~~

Configures which LLM backend and model to use.

.. autoclass:: cortexflow.config.LLMConfig
   :members:
   :undoc-members:

* ``default_model`` - Model used for generating responses
* ``backend`` - LLM backend: ``"ollama"`` or ``"vertex_ai"`` (default: ``"ollama"``)
* ``ollama_host`` - Host URL for Ollama (default: ``"http://localhost:11434"``)
* ``vertex_project_id`` - GCP project ID for Vertex AI
* ``vertex_location`` - GCP region for Vertex AI
* ``vertex_credentials_path`` - Path to service account JSON for Vertex AI

KnowledgeStoreConfig
~~~~~~~~~~~~~~~~~~

Settings for knowledge persistence and retrieval.

.. autoclass:: cortexflow.config.KnowledgeStoreConfig
   :members:
   :undoc-members:

* ``knowledge_store_path`` - Path to persistent storage (default: ``"cortexflow.db"``)
* ``retrieval_type`` - Retrieval strategy: ``"hybrid"``, ``"bm25"``, ``"dense"`` (default: ``"hybrid"``)
* ``use_reranking`` - Enable result reranking (default: True)

GraphRagConfig
~~~~~~~~~~~~~

Settings for graph-based retrieval-augmented generation.

.. autoclass:: cortexflow.config.GraphRagConfig
   :members:
   :undoc-members:

* ``use_graph_rag`` - Enable GraphRAG (default: False)
* ``enable_multi_hop_queries`` - Allow multi-hop graph traversal (default: False)
* ``max_graph_hops`` - Maximum traversal depth (default: 3)

AgentConfig
~~~~~~~~~~

Settings for the Chain of Agents framework.

.. autoclass:: cortexflow.config.AgentConfig
   :members:
   :undoc-members:

* ``use_chain_of_agents`` - Enable Chain of Agents (default: False)
* ``chain_complexity_threshold`` - Minimum complexity to trigger agents (default: 5)

ReflectionConfig
~~~~~~~~~~~~~~~

Settings for self-reflection and verification.

.. autoclass:: cortexflow.config.ReflectionConfig
   :members:
   :undoc-members:

* ``use_self_reflection`` - Enable self-verification (default: False)
* ``reflection_relevance_threshold`` - Minimum relevance score (default: 0.6)

Other Config Sections
~~~~~~~~~~~~~~~~~~~

* ``OntologyConfig`` - Ontology evolution settings
* ``MetadataConfig`` - Provenance and confidence tracking
* ``UncertaintyConfig`` - Contradiction detection and handling
* ``PerformanceConfig`` - Cache sizes and TTL settings
* ``ClassifierConfig`` - ML classifier settings
* ``InferenceConfig`` - Inference engine settings

ConfigBuilder
-----------

The ``ConfigBuilder`` class provides a fluent API for constructing ``CortexFlowConfig``
instances:

.. autoclass:: cortexflow.config.ConfigBuilder
   :members:
   :undoc-members:

Example Usage
-----------

Nested dataclass style:

.. code-block:: python

    from cortexflow import CortexFlowConfig, MemoryConfig, LLMConfig

    config = CortexFlowConfig(
        memory=MemoryConfig(
            active_token_limit=2000,
            working_token_limit=4000,
            archive_token_limit=6000,
        ),
        llm=LLMConfig(
            default_model="llama3",
            backend="ollama",
        ),
    )

Builder pattern style:

.. code-block:: python

    from cortexflow import ConfigBuilder

    config = (ConfigBuilder()
        .with_memory(active_token_limit=2000, working_token_limit=4000)
        .with_llm(default_model="llama3")
        .with_graph_rag(use_graph_rag=True)
        .build())

Vertex AI configuration:

.. code-block:: python

    from cortexflow import ConfigBuilder

    config = (ConfigBuilder()
        .with_llm(backend="vertex_ai", default_model="gemini-2.0-flash")
        .with_memory(active_token_limit=8192)
        .build())

Loading from a dictionary (useful for YAML/JSON files):

.. code-block:: python

    from cortexflow import CortexFlowConfig

    config = CortexFlowConfig.from_dict({
        "active_token_limit": 2000,
        "default_model": "llama3",
        "use_graph_rag": True,
    })
