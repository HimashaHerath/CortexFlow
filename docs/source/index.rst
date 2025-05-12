CortexFlow Documentation
=======================

CortexFlow is a cognitive-inspired memory optimization system for LLMs that implements a 
multi-tier architecture designed to maximize context utilization and enable complex reasoning.

.. image:: _static/architecture.png
   :width: 600px
   :alt: CortexFlow Architecture

Key Features
-----------

- Multi-tier memory management optimizes token usage
- Importance-based information retention
- Progressive context compression
- Knowledge store for long-term information persistence
- Dynamic memory tier weighting
- Vector-based knowledge retrieval
- Advanced retrieval techniques with GraphRAG
- Chain of Agents framework for complex queries
- Self-reflection for verifying knowledge relevance

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/getting_started
   guides/configuration
   guides/chain_of_agents
   guides/graph_rag
   guides/dynamic_weighting
   guides/self_reflection

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/manager
   api/config
   api/memory
   api/knowledge
   api/graph_store
   api/agent_chain
   api/dynamic_weighting
   api/reflection

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_usage
   examples/knowledge_retrieval
   examples/chain_of_agents
   examples/graph_rag
   examples/custom_providers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 