# Logical Reasoning Mechanisms

CortexFlow's logical reasoning module enables powerful inference capabilities over the knowledge graph, allowing the system to derive new knowledge, answer complex questions, and generate hypotheses when information is incomplete.

## Overview

The reasoning engine implements three main inference mechanisms:

1. **Backward Chaining**: For answering "why" questions by tracing logical dependencies between facts
2. **Forward Chaining**: For discovering novel implications by applying rules to existing knowledge
3. **Abductive Reasoning**: For hypothesis generation when information is incomplete

These mechanisms form a comprehensive reasoning system that enhances the knowledge graph with logical inference capabilities.

## Logical Rules

At the core of the inference engine are logical rules, which define how new facts can be derived from existing ones. Each rule consists of:

- **Premises**: Conditions that must be satisfied for the rule to apply
- **Conclusion**: The new fact that can be derived if all premises are satisfied
- **Confidence**: A score (0.0-1.0) representing the confidence in the derived fact
- **Metadata**: Additional information about the rule

For example, a rule for transitivity of the "is_a" relation might look like:

```python
{
    "name": "transitivity_is_a",
    "premise": [
        {"source": "?X", "relation": "is_a", "target": "?Y"},
        {"source": "?Y", "relation": "is_a", "target": "?Z"}
    ],
    "conclusion": {"source": "?X", "relation": "is_a", "target": "?Z"},
    "confidence": 0.9,
    "metadata": {"category": "transitivity"}
}
```

This rule states that if X is a Y and Y is a Z, then X is a Z, with a confidence of 0.9.

## Backward Chaining

Backward chaining is a goal-driven reasoning approach that starts with a query and works backward to find the supporting facts. This is particularly useful for answering "why" questions.

When a user asks "Why is X true?", the system:

1. Extracts the fact pattern from the question (e.g., "dog is_a animal")
2. Checks if this fact exists directly in the knowledge base
3. If not, looks for rules whose conclusions match the query
4. Recursively attempts to prove the premises of matching rules
5. Constructs an explanation trail showing the reasoning steps

### Example: Answering "Why" Questions

```python
# Ask why a dog is an animal
explanation = manager.answer_why_question("Why is a dog an animal?")

# The explanation includes a step-by-step reasoning trail
# [query] Finding explanation for: dog is_a animal
# [fact] Known fact: dog is_a mammal
# [fact] Known fact: mammal is_a animal
# [inference] Applied rule 'transitivity_is_a' to derive: dog is_a animal
# [conclusion] Therefore, dog is_a animal is established.
```

## Forward Chaining

Forward chaining is a data-driven approach that applies rules to existing facts to derive new knowledge. This is useful for discovering novel implications in the knowledge graph.

When forward chaining is applied, the system:

1. Looks for facts that match the premises of rules
2. When all premises of a rule are satisfied, derives the conclusion
3. Adds the new fact to the knowledge base with appropriate confidence
4. Continues this process iteratively, potentially using newly derived facts

### Example: Discovering Novel Implications

```python
# Discover new facts through forward chaining
new_facts = manager.generate_novel_implications(iterations=2)

# Sample output:
# dog is_a vertebrate (Rule: animals_are_vertebrates)
# cat is_a vertebrate (Rule: animals_are_vertebrates)
# dog has_property fur (Rule: property_inheritance)
```

## Abductive Reasoning

Abductive reasoning generates hypotheses that could explain an observation, even when complete information is not available. This is useful for hypothesis generation and exploratory analysis.

When abductive reasoning is applied, the system:

1. Takes an observation as input
2. Finds rules whose conclusions match the observation
3. Treats the premises of these rules as potential explanations
4. Ranks hypotheses by confidence, factoring in whether they are known facts

### Example: Generating Hypotheses

```python
# Generate hypotheses to explain an observation
hypotheses = manager.generate_hypotheses("Eagles have hollow bones")

# Sample output:
# eagle is_a bird (Known fact, Confidence: 0.85)
# hollow_bones enables flight (Novel hypothesis, Confidence: 0.65)
# birds have_property hollow_bones (Novel hypothesis, Confidence: 0.70)
```

## Integration with Knowledge Graph

The inference engine is tightly integrated with the CortexFlow knowledge graph:

- It can access all facts in the graph for reasoning
- Newly inferred facts can be added to the graph with appropriate metadata
- The ontology is used to enhance reasoning (e.g., for hierarchical relationships)
- Confidence scores are propagated through inference chains

## Configuration Options

The reasoning engine can be configured through the CortexFlowConfig:

- `use_inference_engine`: Enable or disable the inference engine (default: False)
- `max_inference_depth`: Maximum recursion depth for backward chaining (default: 5)
- `inference_confidence_threshold`: Minimum confidence threshold for inference (default: 0.6)
- `max_forward_chain_iterations`: Maximum iterations for forward chaining (default: 3)
- `abductive_reasoning_enabled`: Enable or disable abductive reasoning (default: True)
- `max_abductive_hypotheses`: Maximum number of hypotheses to generate (default: 5)

## Usage Example

```python
from cortexflow import CortexFlowManager, CortexFlowConfig

# Enable inference engine in configuration
config = CortexFlowConfig(
    use_graph_rag=True,
    use_inference_engine=True,
    knowledge_store_path="knowledge.db",
    max_inference_depth=5,
    inference_confidence_threshold=0.6
)

# Create manager instance
manager = CortexFlowManager(config)

# Add knowledge to the graph
knowledge = manager.knowledge_store
graph = knowledge.graph_store

# Add entities and relationships
graph.add_entity("bird", "category")
graph.add_entity("eagle", "animal_species")
graph.add_relation("eagle", "is_a", "bird")

# Add a logical rule
manager.add_logical_rule(
    name="birds_can_fly",
    premise_patterns=[
        {"source": "?X", "relation": "is_a", "target": "bird"}
    ],
    conclusion_pattern={"source": "?X", "relation": "can_fly", "target": "true"},
    confidence=0.8
)

# Use backward chaining
explanation = manager.answer_why_question("Why can an eagle fly?")

# Use forward chaining
new_facts = manager.generate_novel_implications()

# Use abductive reasoning
hypotheses = manager.generate_hypotheses("Eagles have hollow bones")
```

For a complete demonstration, see the [reasoning demo script](../examples/reasoning_demo.py). 