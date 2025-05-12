# AdaptiveContext API Reference

This section provides detailed API documentation for the AdaptiveContext system.

## Core Modules

- [CortexFlowManager](manager.md) - The main manager class for the system
- [ConversationMemory](memory.md) - Memory system with multi-tier architecture
- [KnowledgeStore](knowledge.md) - Long-term knowledge storage
- [GraphStore](graph_store.md) - Knowledge graph storage for complex queries

## Feature Modules

- [DynamicWeightingEngine](dynamic_weighting.md) - Dynamic memory tier weighting
- [AgentChain](agent_chain.md) - Chain of Agents framework
- [ReflectionEngine](reflection.md) - Self-reflection and verification

## Support Modules

- [Classifier](classifier.md) - Content and importance classification
- [Compressor](compressor.md) - Context compression
- [Configuration](config.md) - System configuration

## Usage Examples

### Basic Usage

```python
from adaptive_context import CortexFlowManager, CortexFlowConfig

# Configure with custom settings
config = CortexFlowConfig(
    active_token_limit=4000,
    working_token_limit=8000, 
    archive_token_limit=12000,
    use_dynamic_weighting=True,
    use_graph_rag=True,
    use_chain_of_agents=True,
    use_self_reflection=True
)

# Create the context manager
manager = CortexFlowManager(config)

# Add messages to the context
manager.add_message("system", "You are a helpful AI assistant.")
manager.add_message("user", "What is the capital of France?")
response = manager.generate_response()
print(f"Assistant: {response}")

# Get stats about memory usage
stats = manager.get_stats()
print(stats)

# Clean up when done
manager.close()
```

### Knowledge Store Usage

```python
from adaptive_context import CortexFlowManager, CortexFlowConfig

config = CortexFlowConfig(knowledge_store_path="my_knowledge.db")
manager = CortexFlowManager(config)

# Add knowledge to the store
manager.remember_knowledge(
    "Paris is the capital of France and has a population of about 2.2 million people.",
    source="Geography Facts"
)

# Query the knowledge store
knowledge = manager.get_knowledge("Tell me about Paris")
for item in knowledge:
    print(f"Knowledge: {item['text']}")
    print(f"Source: {item['source']}")
    print(f"Relevance: {item['score']}")
```

### Dynamic Weighting Usage

```python
from adaptive_context import CortexFlowManager, CortexFlowConfig

config = CortexFlowConfig(
    active_token_limit=4000,
    working_token_limit=8000, 
    archive_token_limit=12000,
    use_dynamic_weighting=True,
    dynamic_weighting_learning_rate=0.1
)

manager = CortexFlowManager(config)

# Use the system - dynamic weighting happens automatically
manager.add_message("user", "What is quantum computing?")
response = manager.generate_response()

# Get statistics about dynamic weighting
stats = manager.get_dynamic_weighting_stats()
print(f"Current weights: {stats['current_weights']}")
print(f"Current limits: {stats['current_limits']}")

# Reset to default weights if needed
manager.reset_dynamic_weighting()
``` 