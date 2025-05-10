# AdaptiveContext Tests

This directory contains test files for the AdaptiveContext system.

## Unit Tests

Unit tests are organized by module and use pytest. They focus on testing individual components in isolation.

### Running Unit Tests

Install pytest if you haven't already:

```bash
pip install pytest
```

Run all unit tests:

```bash
pytest tests/
```

Run tests for a specific module:

```bash
pytest tests/test_memory.py
pytest tests/test_dynamic_weighting.py
```

Run a specific test:

```bash
pytest tests/test_memory.py::TestConversationMemory::test_add_message
```

## Integration Tests

The project also includes several integration test scripts in the root directory:

- `test.py`: Basic functionality test
- `ollama_test.py`: Tests integration with Ollama models
- `vector_test.py`: Tests vector-based knowledge retrieval
- `advanced_retrieval_test.py`: Tests advanced retrieval techniques
- `graph_rag_benchmark.py`: Benchmarks GraphRAG functionality
- `dynamic_weighting_test.py`: Tests dynamic memory weighting
- `self_reflection_test.py`: Tests self-reflection functionality
- `coa_test.py`: Tests Chain of Agents functionality

### Running Integration Tests

Run an integration test script:

```bash
python test.py
python ollama_test.py
python dynamic_weighting_test.py
```

Some test scripts accept command-line arguments:

```bash
python dynamic_weighting_test.py --model llama3 --plot
python graph_rag_benchmark.py --verbose --keep-db
python coa_test.py --model gemma3:1b
```

## Test Coverage

To generate a test coverage report:

```bash
pip install pytest-cov
pytest --cov=adaptive_context tests/
```

Generate an HTML coverage report:

```bash
pytest --cov=adaptive_context --cov-report=html tests/
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

## Writing New Tests

When writing new tests:

1. Follow the naming convention: `test_*.py` for test files, `test_*` for test methods
2. Use descriptive test method names that explain what is being tested
3. Include docstrings that explain the purpose of the test
4. Test both normal operation and edge cases
5. Use pytest fixtures for setup/teardown when appropriate
6. Mock external dependencies when testing in isolation 