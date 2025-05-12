# CortexFlow Evaluation Framework

This directory contains the Evaluation Framework for the CortexFlow knowledge graph system, providing comprehensive benchmarks and metrics for assessing system performance.

## Overview

The Evaluation Framework includes:

1. **Multi-hop Reasoning Benchmarks**: Test the system's ability to perform complex reasoning across multiple knowledge nodes
2. **Knowledge Consistency Metrics**: Evaluate how knowledge consistency is maintained over time
3. **Automatic Test Generation**: Generate test cases to assess reasoning capabilities
4. **Reasoning Path Logging**: Trace and analyze reasoning paths for debugging and improvement

## Directory Structure

```
benchmark/
├── metrics/                     # Evaluation metrics implementations
│   ├── multi_hop_metrics.py     # Metrics for multi-hop reasoning
│   └── consistency_metrics.py   # Metrics for knowledge consistency
├── test_generation.py           # Automatic test generation utilities
├── evaluation_framework.py      # Main evaluation framework
├── setup_evaluation.py          # Setup script for the framework
└── README.md                    # This file
```

## Getting Started

### Setup

1. Run the setup script to initialize the evaluation environment:

```bash
python benchmark/setup_evaluation.py --results_dir evaluation_results --db_path path/to/db.sqlite
```

This will:
- Create necessary directories
- Generate a default configuration file
- Initialize the evaluation database

### Running Evaluations

Use the main evaluation framework script with different modes:

```bash
# Run a full evaluation
python benchmark/evaluation_framework.py --config evaluation_config.json --mode full

# Run only multi-hop reasoning benchmarks
python benchmark/evaluation_framework.py --mode multi_hop --n_tests 50

# Evaluate knowledge consistency
python benchmark/evaluation_framework.py --mode consistency --days 60

# Generate and run tests
python benchmark/evaluation_framework.py --mode test_gen --n_tests 40

# Analyze reasoning paths
python benchmark/evaluation_framework.py --mode reasoning
```

## Metrics

### Multi-hop Reasoning Metrics

- **Path Overlap**: Measures the overlap between expected and actual reasoning paths
- **Path Order**: Evaluates the correctness of the path traversal order
- **Entity Coverage**: Measures how well the expected entities are covered
- **Hop Accuracy**: Evaluates the accuracy of the number of hops

### Knowledge Consistency Metrics

- **Consistency Score**: Overall knowledge consistency over time
- **Stability Score**: Stability of knowledge representations
- **Contradiction Rate**: Rate of contradictions in the knowledge graph
- **Entity/Relation Consistency**: Specific consistency metrics for entities and relations

## Test Generation

The framework can automatically generate tests for:

- **Multi-hop reasoning**: Tests requiring multiple inference steps
- **Counterfactual reasoning**: Tests for handling non-existent relationships

## Results and Visualization

Evaluation results are stored in the specified results directory and include:

- JSON files with detailed metrics
- Visualizations of key performance indicators
- Reasoning path logs for analysis

## Extending the Framework

You can extend the framework by:

1. Adding new metrics in the `metrics/` directory
2. Creating custom test generators
3. Implementing additional visualization tools
4. Defining domain-specific benchmarks

## Integration with CortexFlow

The evaluation framework integrates with the main CortexFlow system by:

1. Using the same database
2. Accessing the knowledge graph through the CortexFlow manager
3. Utilizing the reasoning mechanisms of CortexFlow
4. Logging additional information for evaluation

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- NetworkX
- SQLite3 