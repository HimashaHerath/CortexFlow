# Uncertainty and Contradictions Handling

CortexFlow now includes robust mechanisms to handle uncertainty and contradictions in the knowledge graph. These features enable the system to:

1. Revise beliefs when new contradictory information arrives
2. Represent uncertainty explicitly with confidence scores and probability distributions
3. Resolve conflicts based on source reliability and recency
4. Reason with incomplete information

## Configuration

Enable uncertainty handling in the CortexFlowConfig:

```python
config = CortexFlowConfig(
    # Enable uncertainty handling
    use_uncertainty_handling=True,
    
    # Automatically detect and resolve contradictions
    auto_detect_contradictions=True,
    
    # Default contradiction resolution strategy
    # Available options: "auto", "recency", "confidence", "reliability", "weighted", "keep_both"
    default_contradiction_strategy="weighted",
    
    # Weight factors for the weighted resolution strategy
    recency_weight=0.6,  # Weight for recency in contradiction resolution
    reliability_weight=0.4,  # Weight for source reliability in contradiction resolution
    
    # Confidence threshold for high-confidence assertions
    confidence_threshold=0.7,
    
    # Type of uncertainty representation
    # Available options: "confidence", "distribution", "both"
    uncertainty_representation="confidence",
    
    # Enable reasoning with incomplete information
    reason_with_incomplete_info=True
)
```

## Key Components

### UncertaintyHandler

The `UncertaintyHandler` class provides the core functionality for handling uncertainty and contradictions:

- Detect contradictions in the knowledge graph
- Resolve contradictions using different strategies
- Maintain source reliability information
- Represent uncertainty with confidence scores and probability distributions
- Track belief revision history
- Reason with incomplete information

## Detecting and Resolving Contradictions

Contradictions are automatically detected when new knowledge is added to the system. The system considers two pieces of information contradictory when they assert different values for the same entity and relation.

### Resolution Strategies

1. **Auto**: Automatically choose the best strategy based on available information
2. **Recency**: Prefer more recent information
3. **Confidence**: Prefer information with higher confidence scores
4. **Reliability**: Prefer information from more reliable sources
5. **Weighted**: Use a weighted approach considering both recency and source reliability
6. **Keep Both**: Keep both pieces of information with reduced confidence

Example:

```python
# Detect contradictions
contradictions = manager.detect_contradictions()

# Resolve a contradiction using a specific strategy
if contradictions:
    resolution = manager.resolve_contradiction(contradictions[0], strategy="reliability")
    print(f"Resolved value: {resolution['resolved_value']}")
    print(f"Confidence: {resolution['confidence']}")
```

## Source Reliability

The system maintains reliability scores for different knowledge sources, which can be used in contradiction resolution.

```python
# Update source reliability
manager.update_source_reliability("scientific_journal", 0.95)
manager.update_source_reliability("social_media", 0.3)

# Get source reliability
reliability = manager.get_source_reliability("scientific_journal")
```

## Uncertainty Representation

### Confidence Scores

Every piece of information in the knowledge graph is associated with a confidence score (0.0-1.0) that represents the system's certainty about that information.

```python
# Add knowledge with confidence
manager.remember_knowledge(
    "Jupiter has 79 known moons.",
    source="astronomy_website",
    confidence=0.9
)
```

### Probability Distributions

For more complex uncertainty, the system can represent information using probability distributions.

```python
# Add a probability distribution
manager.add_probability_distribution(
    entity_id=entity_id,
    relation_id=relation_id,
    distribution_type="discrete",
    distribution_data={
        "values": ["1.5°C", "2.0°C", "2.5°C", "3.0°C", "3.5°C", "4.0°C", "4.5°C"],
        "probabilities": [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]
    }
)

# Get a probability distribution
distribution = manager.get_probability_distribution(entity_id, relation_id)
```

## Belief Revision

The system maintains a history of belief revisions, tracking how information has changed over time and why.

```python
# Get belief revision history
revisions = manager.get_belief_revision_history(entity_id=entity_id)

for revision in revisions:
    print(f"Changed from '{revision['previous_value']}' to '{revision['new_value']}'")
    print(f"Reason: {revision['revision_reason']}")
    print(f"Confidence: {revision['confidence_before']} → {revision['confidence_after']}")
```

## Reasoning with Incomplete Information

When the system doesn't have all the information needed to answer a query, it can:

1. Identify what information is missing
2. Find the best partial matches
3. Provide an answer with appropriate confidence based on available information

```python
# Define a query with required fields
query = {
    "question": "What is the capital of France and its population?",
    "required_fields": ["capital", "population"]
}

# Reason with available knowledge
result = manager.reason_with_incomplete_information(query, available_knowledge)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Missing information: {result['missing_information']}")
```

## Integration with the Knowledge Graph

The uncertainty handling features are deeply integrated with the knowledge graph:

1. Confidence scores are stored with entities and relationships
2. Contradictions are detected based on graph structure
3. Belief revisions update the graph connections
4. Source reliability affects how new information is integrated
5. Probability distributions provide richer representations for uncertain values

## Usage Example

See the complete example in `examples/uncertainty_handling_demo.py` that demonstrates:

1. Belief revision with contradictory information
2. Explicit uncertainty representation
3. Conflict resolution based on source reliability
4. Reasoning with incomplete information

## Benefits

1. **More Accurate Knowledge**: By tracking and resolving contradictions, the system maintains more accurate and up-to-date knowledge.

2. **Appropriate Uncertainty Expression**: The system can express appropriate levels of uncertainty rather than making overly confident assertions.

3. **Transparent Decision Making**: By tracking belief revisions and resolution strategies, the system provides transparency about how it handles contradictory information.

4. **Graceful Degradation**: Even with incomplete information, the system can still provide useful answers with appropriate confidence levels. 