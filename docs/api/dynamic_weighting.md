# Dynamic Weighting API Reference

The Dynamic Weighting module provides functionality to adaptively allocate tokens between memory tiers based on conversation characteristics and queries.

## DynamicWeightingEngine

`DynamicWeightingEngine` is the main class responsible for dynamic memory tier weighting.

### Initialization

```python
from adaptive_context.dynamic_weighting import DynamicWeightingEngine
from adaptive_context.config import CortexFlowConfig

# Create a configuration
config = CortexFlowConfig(
    use_dynamic_weighting=True,
    dynamic_weighting_learning_rate=0.1,
    dynamic_weighting_min_tier_size=1000
)

# Initialize the dynamic weighting engine
engine = DynamicWeightingEngine(config)
```

### Methods

#### analyze_query_complexity

Analyzes the complexity of a user query.

```python
complexity_score = engine.analyze_query_complexity(query)
```

**Parameters:**
- `query` (str): The user's query text

**Returns:**
- `float`: Complexity score between 0.0 and 1.0

#### analyze_document_type

Analyzes the type of content in a document.

```python
doc_type = engine.analyze_document_type(content)
```

**Parameters:**
- `content` (str): Document content to analyze

**Returns:**
- `str`: Document type ("code", "text", "data", or "mixed")

#### calculate_optimal_weights

Calculates optimal tier weights based on query complexity and document type.

```python
optimal_weights = engine.calculate_optimal_weights(query_complexity, document_type)
```

**Parameters:**
- `query_complexity` (float): Complexity score (0.0-1.0)
- `document_type` (str): Type of document ("code", "text", "data", or "mixed")

**Returns:**
- `dict`: Dictionary of tier weights (active, working, archive)

#### update_tier_allocations

Updates memory tier allocations based on current weights.

```python
new_limits = engine.update_tier_allocations()
```

**Returns:**
- `dict`: Dictionary with updated token limits for each tier

#### process_query

Processes a query and updates memory allocations.

```python
new_limits = engine.process_query(query, context_content=None)
```

**Parameters:**
- `query` (str): User query to analyze
- `context_content` (str, optional): Recent context content to analyze

**Returns:**
- `dict`: Dictionary with updated token limits for each tier

#### get_stats

Gets statistics about the dynamic weighting engine.

```python
stats = engine.get_stats()
```

**Returns:**
- `dict`: Dictionary with statistics and current state

#### reset_to_defaults

Resets weighting to default values.

```python
engine.reset_to_defaults()
```

## Integration with CortexFlowManager

The DynamicWeightingEngine is integrated with CortexFlowManager and can be used as follows:

```python
from adaptive_context import CortexFlowManager, CortexFlowConfig

# Enable Dynamic Weighting in configuration
config = CortexFlowConfig(
    active_token_limit=4000,
    working_token_limit=8000, 
    archive_token_limit=12000,
    use_dynamic_weighting=True,
    dynamic_weighting_learning_rate=0.1
)

# Create the context manager with this configuration
manager = CortexFlowManager(config)

# Add a message - this will trigger dynamic weighting analysis
manager.add_message("user", "What is the capital of France?")

# Generate a response
response = manager.generate_response()

# Get statistics about dynamic weighting
stats = manager.get_dynamic_weighting_stats()
print(stats)

# Reset to default weights if needed
manager.reset_dynamic_weighting()
```

## Configuration Options

The following configuration options are available for Dynamic Weighting:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_dynamic_weighting` | bool | False | Whether to enable Dynamic Weighting |
| `dynamic_weighting_learning_rate` | float | 0.1 | Learning rate for weight adjustments (0.0-1.0) |
| `dynamic_weighting_min_tier_size` | int | 1000 | Minimum token size for any tier |
| `dynamic_weighting_default_ratios` | dict | `{"active": 0.25, "working": 0.35, "archive": 0.40}` | Default tier ratios |

## Feature Details

### Query Complexity Analysis

Query complexity is calculated based on multiple factors:

- Question type (why questions are more complex than what questions)
- Length and word count
- Entity and number counts
- Code-related indicators
- Multi-part questions

The complexity score ranges from 0.0 (very simple) to 1.0 (very complex).

### Document Type Detection

Document types are detected using indicators:

- Code indicators: function/class definitions, imports, programming syntax
- Data indicators: JSON structures, tables, dataframes
- Text: Natural language without specific code or data indicators
- Mixed: Combination of the above

### Weight Calculation

Optimal weights are calculated based on:

1. Query complexity (complex queries get more active memory)
2. Document type (code benefits from more active memory, data from more working memory)
3. Historical patterns (sustained high complexity leads to more active and working memory)

Weights are gradually adjusted using the learning rate to avoid abrupt changes.

## Example

Here's a detailed example of using dynamic weighting:

```python
from adaptive_context import CortexFlowManager, CortexFlowConfig

# Create configuration with Dynamic Weighting enabled
config = CortexFlowConfig(
    active_token_limit=4000,
    working_token_limit=8000,
    archive_token_limit=12000,
    use_dynamic_weighting=True,
    dynamic_weighting_learning_rate=0.2  # Faster learning for demonstration
)

# Initialize the manager
manager = CortexFlowManager(config)

# Add a system message
manager.add_message("system", "You are a helpful AI assistant with dynamic memory capabilities.")

# Get initial tier stats
initial_stats = manager.get_stats()
print("Initial tier distribution:")
print(f"Active: {initial_stats['memory']['tiers']['active']['limit']} tokens")
print(f"Working: {initial_stats['memory']['tiers']['working']['limit']} tokens")
print(f"Archive: {initial_stats['memory']['tiers']['archive']['limit']} tokens")

# Process a simple query (will reduce active tier allocation)
manager.add_message("user", "What time is it?")
manager.generate_response()

# Check updated tier stats
simple_query_stats = manager.get_stats()
print("\nAfter simple query:")
print(f"Active: {simple_query_stats['memory']['tiers']['active']['limit']} tokens")

# Process a complex query (will increase active tier allocation)
manager.add_message("user", "Explain the relationship between quantum mechanics and general relativity.")
manager.generate_response()

# Check updated tier stats
complex_query_stats = manager.get_stats()
print("\nAfter complex query:")
print(f"Active: {complex_query_stats['memory']['tiers']['active']['limit']} tokens")

# Process a code-related query
manager.add_message("user", "Write a Python function to implement quicksort.")
manager.generate_response()

# Get dynamic weighting stats
dw_stats = manager.get_dynamic_weighting_stats()
print("\nDynamic Weighting Stats:")
print(f"Current weights: {dw_stats['current_weights']}")
print(f"Current limits: {dw_stats['current_limits']}")
``` 