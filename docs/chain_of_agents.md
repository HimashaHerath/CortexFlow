# Chain of Agents (CoA) Framework

## Overview

The Chain of Agents (CoA) framework in AdaptiveContext enables complex query processing through a sequence of specialized agents, each with a distinct role in the reasoning process. This approach is based on research from Google's "Chain of Agents: Large Language Models Collaborating on Long Context Tasks" (2025).

The primary benefit of this approach is the ability to break down complex reasoning tasks into specialized sub-tasks, allowing for more focused and effective processing of multi-hop queries, especially those requiring integration of multiple pieces of information.

## Architecture

The framework consists of three main components:

1. **Base Agent Class**: Provides the interface and common functionality for all agents
2. **Specialized Agents**: Individual agents with specific roles in the reasoning chain
3. **AgentChainManager**: Coordinates the sequential processing of agents

### Agent Types

The current implementation includes three specialized agents:

#### Explorer Agent

**Role**: Broadly search for relevant information without directly answering the query.

**Key Features**:
- Uses higher retrieval thresholds to gather diverse knowledge
- Employs both vector-based and graph-based retrieval methods
- Focuses on identifying key entities and concepts
- Explores related topics and potential connections

#### Analyzer Agent

**Role**: Examine relationships between facts discovered by the Explorer.

**Key Features**:
- Identifies connections between pieces of information
- Looks for patterns and inconsistencies
- Organizes information into a coherent structure
- Highlights the most important relationships relevant to the query

#### Synthesizer Agent

**Role**: Combine insights from previous agents to generate a comprehensive answer.

**Key Features**:
- Integrates information from Explorer and Analyzer agents
- Formulates a coherent, comprehensive response
- Ensures the answer directly addresses the original query
- Includes relevant facts and relationships in a structured format

## Implementation Details

### Base Agent Class

The `Agent` class defines the common interface for all agents:

```python
class Agent:
    def __init__(
        self, 
        name: str, 
        role: str, 
        config: CortexFlowConfig,
        knowledge_store: Optional[KnowledgeStore] = None
    ):
        # Initialize agent with name, role, and configuration
        
    def process(
        self, 
        query: str,
        context: Dict[str, Any], 
        agent_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        # Process the query with this agent's specialized capability
```

### AgentChainManager

The `AgentChainManager` coordinates the sequential execution of agents:

```python
class AgentChainManager:
    def __init__(self, config: CortexFlowConfig, knowledge_store: KnowledgeStore):
        # Initialize the agent chain
        
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Process a query through the chain of agents
```

### Integration with CortexFlowManager

The Chain of Agents is integrated with the main `CortexFlowManager` through:

1. Configuration options in `CortexFlowConfig`
2. Automatic engagement for complex queries
3. Fallback to standard processing when needed

## Usage

### Basic Usage

Enable and use the Chain of Agents in your application:

```python
from adaptive_context import CortexFlowManager, CortexFlowConfig

# Configure with Chain of Agents enabled
config = CortexFlowConfig(
    use_chain_of_agents=True,
    chain_complexity_threshold=5,  # Only use for reasonably complex queries
    # ... other config options
)

# Create the manager
manager = CortexFlowManager(config)

# Add system message
manager.add_message("system", "You are a helpful AI assistant.")

# Add user query
manager.add_message("user", "What connections exist between quantum physics and consciousness?")

# Generate response (will automatically use Chain of Agents for complex queries)
response = manager.generate_response()
print(response)
```

### Testing

A dedicated test script is provided to evaluate the Chain of Agents functionality:

```bash
# Basic usage
python coa_test.py

# With a specific model
python coa_test.py --model llama3.2

# With verbose logging
python coa_test.py --model llama3.2 --verbose

# With custom memory settings
python coa_test.py --active-tokens 2000 --working-tokens 4000 --archive-tokens 8000
```

## Performance Considerations

The Chain of Agents approach provides more thorough and structured reasoning at the cost of increased processing time. Each agent requires one or more LLM calls, leading to longer overall response times compared to single-agent approaches.

Key performance considerations:
- Average response time: 60-90 seconds for complex queries with 3 agents
- Memory usage: Similar to standard processing
- Token usage: Higher due to multiple LLM calls

## Future Improvements

See [TODO.md](../TODO.md) for planned enhancements to the Chain of Agents framework.

## Example Output

Here's an example of the Chain of Agents processing a complex query:

```
Query: Is the tallest mountain in Japan located on the same island as the capital city?

Explorer Agent Results:
- Identified key entities: "tallest mountain in Japan", "capital city", "island"
- Found relevant knowledge about Japan's geography, mountains, and capital
- Discovered that Mount Fuji is the tallest mountain in Japan
- Found information about Tokyo being Japan's capital
- Explored Japan's main islands and their geography

Analyzer Agent Results:
- Established relationship between Mount Fuji and its location
- Connected Tokyo's location to specific islands
- Analyzed the geographical relationship between mountains and cities in Japan
- Identified patterns in Japanese geographical distribution
- Verified consistency across different knowledge sources

Synthesizer Agent Results:
After synthesizing information from both the Explorer and Analyzer Agents, it can be concluded that the tallest mountain in Japan is located on the island of Honshu, which is also where the capital city Tokyo is situated.

Therefore, the answer is: Yes, the tallest mountain in Japan is located on the same island as the capital city Tokyo.
```

## Technical Requirements

- Python 3.7+
- Ollama with a suitable model (llama3, mistral, gemma, etc.)
- Dependencies: requests, networkx (for graph-based exploration)
- Optional: spacy for enhanced entity extraction 