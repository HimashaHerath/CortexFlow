# CortexFlow Tests

This directory contains all test files for the CortexFlow system.

## Test Files

| File | Description |
|---|---|
| `test_agent_chain.py` | Chain of Agents multi-agent reasoning |
| `test_classifier.py` | Importance classification (rule-based, ML, LLM) |
| `test_compressor.py` | Context compression (extractive & abstractive) |
| `test_config.py` | Nested config dataclasses & ConfigBuilder |
| `test_dynamic_weighting.py` | Dynamic memory tier weighting |
| `test_extraction.py` | Entity and relation extraction |
| `test_graph_enhancements.py` | Graph store advanced features |
| `test_graph_traversal.py` | Weighted, bidirectional, constrained path queries |
| `test_inference.py` | Inference engine (forward/backward chaining, abduction) |
| `test_knowledge_integration.py` | Knowledge store integration tests |
| `test_knowledge_store.py` | Knowledge store unit tests |
| `test_llm_client.py` | LLM client abstraction (Ollama & Vertex AI) |
| `test_memory.py` | Multi-tier memory management |
| `test_memory_tiers.py` | Memory tier transitions and limits |
| `test_ontology.py` | Ontology system and concept hierarchies |
| `test_reasoning_engine.py` | Multi-step reasoning engine |
| `test_reflection.py` | Self-reflection and self-correction |
| `test_vertex_ai_benchmark.py` | Vertex AI integration benchmark (requires credentials) |

## Running Tests

Run all tests:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_memory.py -v
```

Run a specific test:

```bash
pytest tests/test_memory.py::TestConversationMemory::test_add_message -v
```

## Test Coverage

```bash
pip install pytest-cov
pytest --cov=cortexflow tests/
pytest --cov=cortexflow --cov-report=html tests/
```

Then open `htmlcov/index.html` to view the coverage report.
