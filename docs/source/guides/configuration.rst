Configuration
=============

CortexFlow offers flexible configuration options to adapt the system to your needs.
Configuration uses a nested dataclass structure, with each subsystem having its own
config class. You can also use the fluent ``ConfigBuilder`` for convenience.

Nested Configuration
-----------------

The primary way to configure CortexFlow is through ``CortexFlowConfig`` with nested
sub-config dataclasses:

.. code-block:: python

    from cortexflow import CortexFlowManager, CortexFlowConfig, MemoryConfig, LLMConfig

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

    manager = CortexFlowManager(config)

Builder Pattern
--------------

The ``ConfigBuilder`` provides a fluent API for building configuration:

.. code-block:: python

    from cortexflow import CortexFlowManager, ConfigBuilder

    config = (ConfigBuilder()
        .with_memory(active_token_limit=2000, working_token_limit=4000, archive_token_limit=6000)
        .with_llm(default_model="llama3", backend="ollama")
        .build())

    manager = CortexFlowManager(config)

Memory Tier Configuration
----------------------

You can adjust the size of each memory tier:

.. code-block:: python

    from cortexflow import CortexFlowConfig, MemoryConfig

    config = CortexFlowConfig(
        memory=MemoryConfig(
            active_token_limit=2000,   # Most recent/important context
            working_token_limit=4000,  # Medium-term storage
            archive_token_limit=6000,  # Long-term storage
            use_dynamic_weighting=True,
        ),
    )

LLM Configuration (Ollama)
-----------------------

Configure which LLM to use with the local Ollama backend:

.. code-block:: python

    from cortexflow import CortexFlowConfig, LLMConfig

    config = CortexFlowConfig(
        llm=LLMConfig(
            backend="ollama",
            default_model="llama3",
            ollama_host="http://localhost:11434",
        ),
    )

Vertex AI Configuration
--------------------

CortexFlow supports Google Vertex AI as an alternative LLM backend. Use the
``ConfigBuilder`` shorthand or configure ``LLMConfig`` directly:

.. code-block:: python

    from cortexflow import ConfigBuilder

    # Using the builder (recommended)
    config = (ConfigBuilder()
        .with_llm(backend="vertex_ai", default_model="gemini-2.0-flash")
        .with_memory(active_token_limit=8192)
        .build())

Or with explicit Vertex AI settings:

.. code-block:: python

    from cortexflow import CortexFlowConfig, LLMConfig

    config = CortexFlowConfig(
        llm=LLMConfig(
            backend="vertex_ai",
            default_model="gemini-2.0-flash",
            vertex_project_id="my-gcp-project",
            vertex_location="us-central1",
            vertex_credentials_path="/path/to/service-account.json",
        ),
    )

Or via the dedicated ``with_vertex_ai`` builder method:

.. code-block:: python

    from cortexflow import ConfigBuilder

    config = (ConfigBuilder()
        .with_vertex_ai(
            project_id="my-gcp-project",
            location="us-central1",
            default_model="gemini-2.0-flash",
            credentials_path="/path/to/service-account.json",
        )
        .with_memory(active_token_limit=8192)
        .build())

Advanced Features Configuration
---------------------------

Enable or disable specific features through their respective config sections:

.. code-block:: python

    from cortexflow import (
        CortexFlowConfig, MemoryConfig, LLMConfig,
        AgentConfig, ReflectionConfig, KnowledgeStoreConfig,
        GraphRagConfig,
    )

    config = CortexFlowConfig(
        memory=MemoryConfig(
            active_token_limit=2000,
            working_token_limit=4000,
            archive_token_limit=6000,
        ),
        llm=LLMConfig(
            default_model="llama3",
        ),
        agents=AgentConfig(
            use_chain_of_agents=True,
            chain_complexity_threshold=5,
        ),
        reflection=ReflectionConfig(
            use_self_reflection=True,
            reflection_relevance_threshold=0.6,
        ),
        knowledge_store=KnowledgeStoreConfig(
            knowledge_store_path="cortexflow.db",
        ),
        graph_rag=GraphRagConfig(
            use_graph_rag=True,
        ),
    )

Or equivalently with the builder:

.. code-block:: python

    from cortexflow import ConfigBuilder

    config = (ConfigBuilder()
        .with_memory(active_token_limit=2000, working_token_limit=4000, archive_token_limit=6000)
        .with_llm(default_model="llama3")
        .with_agents(use_chain_of_agents=True, chain_complexity_threshold=5)
        .with_reflection(use_self_reflection=True, reflection_relevance_threshold=0.6)
        .with_knowledge_store(knowledge_store_path="cortexflow.db")
        .with_graph_rag(use_graph_rag=True)
        .build())

Persistence Configuration
---------------------

Configure how knowledge is stored and retrieved:

.. code-block:: python

    from cortexflow import CortexFlowConfig, KnowledgeStoreConfig, GraphRagConfig

    config = CortexFlowConfig(
        knowledge_store=KnowledgeStoreConfig(
            knowledge_store_path="cortexflow.db",
            retrieval_type="hybrid",
            use_reranking=True,
            rerank_top_k=15,
        ),
        graph_rag=GraphRagConfig(
            use_graph_rag=True,
            enable_multi_hop_queries=True,
            max_graph_hops=3,
        ),
    )

Loading from a Dictionary
----------------------

You can also create a configuration from a flat dictionary (useful for loading
from YAML/JSON files):

.. code-block:: python

    from cortexflow import CortexFlowConfig

    config = CortexFlowConfig.from_dict({
        "active_token_limit": 2000,
        "working_token_limit": 4000,
        "archive_token_limit": 6000,
        "default_model": "llama3",
        "use_graph_rag": True,
    })

The ``from_dict`` method automatically routes each key to its correct nested
config section.
