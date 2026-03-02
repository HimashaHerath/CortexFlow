CortexFlow Documentation
=======================

CortexFlow is a cognitive-inspired memory optimization system for LLMs that implements a
multi-tier architecture designed to maximize context utilization and enable complex reasoning.

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
- Vertex AI and Ollama backend support

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/getting_started
   guides/configuration
   guides/migration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/manager
   api/config

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: Design & Research

   ../chain_of_agents
   ../self_reflection
   ../dynamic_weighting
   ../research_improvements
   ../logical_reasoning
   ../performance_optimization
   ../uncertainty_handling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
