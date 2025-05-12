"""
CortexFlow Reasoning Engine module.

This module provides advanced reasoning capabilities over the knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Generator
import json
import time
import copy
from enum import Enum

# Try importing graph libraries
try:
    import networkx as nx
    NETWORKX_ENABLED = True
except ImportError:
    NETWORKX_ENABLED = False
    logging.warning("networkx not found. Reasoning engine capabilities will be limited.")

from cortexflow.config import CortexFlowConfig
from cortexflow.interfaces import KnowledgeStoreInterface
from cortexflow.inference import InferenceEngine
from cortexflow.uncertainty_handler import UncertaintyHandler

logger = logging.getLogger('cortexflow')

class ReasoningStep:
    """Represents a single step in a multi-step reasoning process."""
    
    def __init__(
        self,
        step_id: str,
        description: str,
        input_entities: List[str] = None,
        output_entities: List[str] = None,
        relations_used: List[str] = None,
        confidence: float = 1.0,
        explanation: str = "",
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a reasoning step.
        
        Args:
            step_id: Unique identifier for this step
            description: Description of what this step does
            input_entities: List of input entity IDs or names
            output_entities: List of output entity IDs or names
            relations_used: List of relation types used in this step
            confidence: Confidence score for this step
            explanation: Human-readable explanation
            metadata: Additional metadata
        """
        self.step_id = step_id
        self.description = description
        self.input_entities = input_entities or []
        self.output_entities = output_entities or []
        self.relations_used = relations_used or []
        self.confidence = confidence
        self.explanation = explanation
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "input_entities": self.input_entities,
            "output_entities": self.output_entities,
            "relations_used": self.relations_used,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningStep':
        """Create from dictionary representation."""
        return cls(
            step_id=data.get("step_id", ""),
            description=data.get("description", ""),
            input_entities=data.get("input_entities", []),
            output_entities=data.get("output_entities", []),
            relations_used=data.get("relations_used", []),
            confidence=data.get("confidence", 1.0),
            explanation=data.get("explanation", ""),
            metadata=data.get("metadata", {})
        )


class ReasoningStrategy(Enum):
    """Enum for different reasoning strategies."""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    ABDUCTIVE = "abductive"
    BIDIRECTIONAL = "bidirectional"
    WEIGHTED_PATH = "weighted_path"
    CONSTRAINED_PATH = "constrained_path"


class ReasoningState:
    """Tracks the state of multi-step reasoning processes."""
    
    def __init__(self, query_id: str, original_query: str):
        """
        Initialize a reasoning state.
        
        Args:
            query_id: Unique identifier for the query
            original_query: The original user query
        """
        self.query_id = query_id
        self.original_query = original_query
        self.steps: List[ReasoningStep] = []
        self.current_entities: Set[str] = set()
        self.current_step_index: int = 0
        self.completed: bool = False
        self.result: Dict[str, Any] = {}
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the state."""
        self.steps.append(step)
        self.current_entities.update(step.output_entities)
        self.current_step_index += 1
        
    def complete(self, result: Dict[str, Any]) -> None:
        """Mark the reasoning process as complete."""
        self.completed = True
        self.result = result
        self.end_time = time.time()
        
    def get_duration(self) -> float:
        """Get the duration of the reasoning process."""
        end = self.end_time or time.time()
        return end - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "steps": [step.to_dict() for step in self.steps],
            "current_entities": list(self.current_entities),
            "current_step_index": self.current_step_index,
            "completed": self.completed,
            "result": self.result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.get_duration(),
            "metadata": self.metadata
        }


class QueryPlanner:
    """Plans and decomposes complex queries into reasoning steps."""
    
    def __init__(self, knowledge_store: KnowledgeStoreInterface, config: CortexFlowConfig = None):
        """
        Initialize the query planner.
        
        Args:
            knowledge_store: Knowledge store interface
            config: Configuration
        """
        self.knowledge_store = knowledge_store
        self.config = config or CortexFlowConfig()
        self.inference_engine = getattr(knowledge_store, 'inference_engine', None)
        self.graph_store = getattr(knowledge_store, 'graph_store', None)
        
    def plan_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Break down a complex query into reasoning steps.
        
        Args:
            query: The user query
            
        Returns:
            List of reasoning steps
        """
        # Extract entities and relations from the query
        entities, relations = self._extract_query_components(query)
        
        # Create a plan based on query type
        if self._is_causal_query(query):
            return self._plan_causal_query(query, entities, relations)
        elif self._is_comparison_query(query):
            return self._plan_comparison_query(query, entities, relations)
        elif self._is_temporal_query(query):
            return self._plan_temporal_query(query, entities, relations)
        else:
            return self._plan_generic_query(query, entities, relations)
    
    def _extract_query_components(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract entities and relations from the query."""
        entities = []
        relations = []
        
        if self.graph_store:
            # Extract entities using the graph store's entity extraction
            extracted_entities = self.graph_store.extract_entities(query)
            entities = [e.get("text") for e in extracted_entities]
            
            # Extract relations using the graph store's relation extraction
            extracted_relations = self.graph_store.extract_relations(query)
            relations = [r[1] for r in extracted_relations]  # r[1] is the predicate/relation
        
        return entities, relations
    
    def _is_causal_query(self, query: str) -> bool:
        """Check if this is a causal query."""
        causal_indicators = ["cause", "effect", "result", "lead", "impact", "influence", "why"]
        return any(indicator in query.lower() for indicator in causal_indicators)
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if this is a comparison query."""
        comparison_indicators = ["compare", "difference", "similar", "versus", "vs", "better", "worse"]
        return any(indicator in query.lower() for indicator in comparison_indicators)
    
    def _is_temporal_query(self, query: str) -> bool:
        """Check if this is a temporal query."""
        temporal_indicators = ["when", "before", "after", "during", "timeline", "history", "evolution"]
        return any(indicator in query.lower() for indicator in temporal_indicators)
    
    def _plan_causal_query(self, query: str, entities: List[str], relations: List[str]) -> List[Dict[str, Any]]:
        """Plan steps for a causal query."""
        steps = []
        
        # Step 1: Identify relevant entities
        steps.append({
            "description": "Identify relevant entities and their properties",
            "strategy": ReasoningStrategy.WEIGHTED_PATH.value,
            "entities": entities
        })
        
        # Step 2: Find causal relationships
        steps.append({
            "description": "Discover causal relationships between entities",
            "strategy": ReasoningStrategy.FORWARD_CHAINING.value,
            "focus_relations": ["causes", "leads_to", "results_in", "affects"]
        })
        
        # Step 3: Evaluate strength of causal links
        steps.append({
            "description": "Evaluate the strength and confidence of causal relationships",
            "strategy": ReasoningStrategy.WEIGHTED_PATH.value
        })
        
        return steps
    
    def _plan_comparison_query(self, query: str, entities: List[str], relations: List[str]) -> List[Dict[str, Any]]:
        """Plan steps for a comparison query."""
        steps = []
        
        # Step 1: Retrieve properties of both entities
        steps.append({
            "description": "Retrieve properties of entities to be compared",
            "strategy": ReasoningStrategy.BIDIRECTIONAL.value,
            "entities": entities
        })
        
        # Step 2: Find common relationships
        steps.append({
            "description": "Identify common relationships and attributes",
            "strategy": ReasoningStrategy.CONSTRAINED_PATH.value
        })
        
        # Step 3: Analyze differences
        steps.append({
            "description": "Analyze differences between entities",
            "strategy": ReasoningStrategy.ABDUCTIVE.value
        })
        
        return steps
    
    def _plan_temporal_query(self, query: str, entities: List[str], relations: List[str]) -> List[Dict[str, Any]]:
        """Plan steps for a temporal query."""
        steps = []
        
        # Step a: Identify temporal markers
        steps.append({
            "description": "Identify temporal markers and events",
            "strategy": ReasoningStrategy.CONSTRAINED_PATH.value,
            "entities": entities,
            "focus_relations": ["occurred_on", "happened_during", "started_at", "ended_at"]
        })
        
        # Step 2: Establish timeline
        steps.append({
            "description": "Establish chronological ordering of events",
            "strategy": ReasoningStrategy.FORWARD_CHAINING.value
        })
        
        return steps
    
    def _plan_generic_query(self, query: str, entities: List[str], relations: List[str]) -> List[Dict[str, Any]]:
        """Plan steps for a generic query."""
        steps = []
        
        # Step 1: Entity exploration
        steps.append({
            "description": "Explore relevant entities and their properties",
            "strategy": ReasoningStrategy.BIDIRECTIONAL.value,
            "entities": entities
        })
        
        # Step 2: Relation discovery
        if relations:
            steps.append({
                "description": "Discover relationships between entities",
                "strategy": ReasoningStrategy.CONSTRAINED_PATH.value,
                "focus_relations": relations
            })
        else:
            steps.append({
                "description": "Discover key relationships between entities",
                "strategy": ReasoningStrategy.WEIGHTED_PATH.value
            })
        
        return steps


class ReasoningEngine:
    """Coordinates inference processes and multi-step reasoning."""
    
    def __init__(self, knowledge_store: KnowledgeStoreInterface, config: CortexFlowConfig = None):
        """
        Initialize the reasoning engine.
        
        Args:
            knowledge_store: Knowledge store interface
            config: Configuration
        """
        self.knowledge_store = knowledge_store
        self.config = config or CortexFlowConfig()
        
        # Get reference to graph store and inference engine
        self.graph_store = getattr(knowledge_store, 'graph_store', None)
        self.inference_engine = getattr(knowledge_store, 'inference_engine', None)
        if not self.inference_engine and self.graph_store:
            self.inference_engine = InferenceEngine(knowledge_store, config)
            
        # Get reference to uncertainty handler
        self.uncertainty_handler = getattr(knowledge_store, 'uncertainty_handler', None)
        if not self.uncertainty_handler and self.graph_store:
            self.uncertainty_handler = UncertaintyHandler(config, self.graph_store)
        
        # Initialize query planner
        self.query_planner = QueryPlanner(knowledge_store, config)
        
        # Active reasoning states
        self.active_states: Dict[str, ReasoningState] = {}
        
        logger.info("Reasoning engine initialized")
    
    def reason(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform multi-step reasoning to answer a complex query.
        
        Args:
            query: The user query
            context: Additional context
            
        Returns:
            Reasoning results
        """
        context = context or {}
        query_id = f"query_{int(time.time())}"
        
        # Create a new reasoning state
        state = ReasoningState(query_id, query)
        self.active_states[query_id] = state
        
        try:
            # Plan the query steps
            plan = self.query_planner.plan_query(query)
            
            # Execute each step in the plan
            for i, step_plan in enumerate(plan):
                step_id = f"{query_id}_step_{i+1}"
                step_description = step_plan.get("description", f"Step {i+1}")
                strategy = step_plan.get("strategy", ReasoningStrategy.FORWARD_CHAINING.value)
                
                # Execute the reasoning step
                step_result = self._execute_reasoning_step(
                    query=query,
                    strategy=strategy,
                    state=state,
                    step_plan=step_plan
                )
                
                # Create and add the reasoning step
                step = ReasoningStep(
                    step_id=step_id,
                    description=step_description,
                    input_entities=step_plan.get("entities", []),
                    output_entities=step_result.get("output_entities", []),
                    relations_used=step_result.get("relations_used", []),
                    confidence=step_result.get("confidence", 1.0),
                    explanation=step_result.get("explanation", ""),
                    metadata=step_result
                )
                state.add_step(step)
                
                # Check if we have an answer
                if step_result.get("is_final_answer", False):
                    break
            
            # Prepare the final result
            result = self._prepare_result(state)
            state.complete(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during reasoning: {e}")
            state.complete({
                "error": str(e),
                "partial_result": self._prepare_result(state)
            })
            return state.result
    
    def _execute_reasoning_step(
        self, 
        query: str,
        strategy: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a reasoning step using the specified strategy."""
        if strategy == ReasoningStrategy.FORWARD_CHAINING.value:
            return self._forward_chaining_step(query, state, step_plan)
        elif strategy == ReasoningStrategy.BACKWARD_CHAINING.value:
            return self._backward_chaining_step(query, state, step_plan)
        elif strategy == ReasoningStrategy.ABDUCTIVE.value:
            return self._abductive_reasoning_step(query, state, step_plan)
        elif strategy == ReasoningStrategy.BIDIRECTIONAL.value:
            return self._bidirectional_search_step(query, state, step_plan)
        elif strategy == ReasoningStrategy.WEIGHTED_PATH.value:
            return self._weighted_path_step(query, state, step_plan)
        elif strategy == ReasoningStrategy.CONSTRAINED_PATH.value:
            return self._constrained_path_step(query, state, step_plan)
        else:
            return self._default_reasoning_step(query, state, step_plan)
    
    def _forward_chaining_step(
        self, 
        query: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a forward chaining reasoning step."""
        if not self.inference_engine:
            return {
                "error": "Inference engine not available",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform forward chaining without an inference engine."
            }
        
        # Apply forward chaining to derive new facts
        focus_relations = step_plan.get("focus_relations", [])
        inferred_facts = self.inference_engine.forward_chain(iterations=3)
        
        # Filter to focus on specific relations if specified
        if focus_relations:
            inferred_facts = [
                fact for fact in inferred_facts
                if fact.get("relation") in focus_relations
            ]
            
        # Extract entities from inferred facts
        output_entities = []
        for fact in inferred_facts:
            output_entities.append(fact.get("source", ""))
            output_entities.append(fact.get("target", ""))
        
        # Calculate overall confidence
        confidence = 0.0
        if inferred_facts:
            confidence = sum(fact.get("confidence", 0.0) for fact in inferred_facts) / len(inferred_facts)
        
        return {
            "inferred_facts": inferred_facts,
            "output_entities": list(set(output_entities)),
            "relations_used": list(set(fact.get("relation", "") for fact in inferred_facts)),
            "confidence": confidence,
            "explanation": f"Inferred {len(inferred_facts)} new facts using forward chaining."
        }
    
    def _backward_chaining_step(
        self, 
        query: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a backward chaining reasoning step."""
        if not self.inference_engine:
            return {
                "error": "Inference engine not available",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform backward chaining without an inference engine."
            }
        
        # Extract target fact from plan or state
        target_entities = step_plan.get("entities", [])
        if not target_entities and state.current_entities:
            target_entities = list(state.current_entities)
            
        if not target_entities:
            return {
                "error": "No target entities for backward chaining",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform backward chaining without target entities."
            }
        
        # Create a query fact from the first target entity
        query_fact = {"source": target_entities[0], "relation": "?", "target": "?"}
        
        # Apply backward chaining
        success, explanation = self.inference_engine.backward_chain(query_fact, depth=3)
        
        # Extract entities from explanation
        output_entities = []
        relations_used = []
        
        for step in explanation:
            fact = step.get("fact", {})
            if "source" in fact:
                output_entities.append(fact["source"])
            if "target" in fact:
                output_entities.append(fact["target"])
            if "relation" in fact:
                relations_used.append(fact["relation"])
        
        return {
            "success": success,
            "explanation_trail": explanation,
            "output_entities": list(set(output_entities)),
            "relations_used": list(set(relations_used)),
            "confidence": 1.0 if success else 0.0,
            "explanation": f"Backward chaining {'successful' if success else 'failed'} with {len(explanation)} steps."
        }
    
    def _abductive_reasoning_step(
        self, 
        query: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an abductive reasoning step."""
        if not self.inference_engine:
            return {
                "error": "Inference engine not available",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform abductive reasoning without an inference engine."
            }
        
        # Extract observation from plan or state
        target_entities = step_plan.get("entities", [])
        if not target_entities and state.current_entities:
            target_entities = list(state.current_entities)
            
        if not target_entities:
            return {
                "error": "No target entities for abductive reasoning",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform abductive reasoning without target entities."
            }
        
        # Create an observation fact from the first target entity
        observation = {"source": "?", "relation": "?", "target": target_entities[0]}
        
        # Apply abductive reasoning
        hypotheses = self.inference_engine.abductive_reasoning(observation, max_hypotheses=3)
        
        # Extract entities from hypotheses
        output_entities = []
        relations_used = []
        
        for hypothesis in hypotheses:
            facts = hypothesis.get("facts", [])
            for fact in facts:
                if "source" in fact:
                    output_entities.append(fact["source"])
                if "target" in fact:
                    output_entities.append(fact["target"])
                if "relation" in fact:
                    relations_used.append(fact["relation"])
        
        # Calculate confidence
        confidence = 0.0
        if hypotheses:
            confidence = sum(h.get("confidence", 0.0) for h in hypotheses) / len(hypotheses)
        
        return {
            "hypotheses": hypotheses,
            "output_entities": list(set(output_entities)),
            "relations_used": list(set(relations_used)),
            "confidence": confidence,
            "explanation": f"Generated {len(hypotheses)} hypotheses using abductive reasoning."
        }
    
    def _bidirectional_search_step(
        self, 
        query: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a bidirectional search step."""
        if not self.graph_store:
            return {
                "error": "Graph store not available",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform bidirectional search without a graph store."
            }
        
        # Extract start and end entities
        entities = step_plan.get("entities", [])
        if len(entities) < 2:
            # If we only have one entity or none, try to use current entities
            if state.current_entities:
                entities = list(state.current_entities)
                
        if len(entities) < 2:
            return {
                "error": "Not enough entities for bidirectional search",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Need at least two entities for bidirectional search."
            }
        
        # Get the start and end entities
        start_entity = entities[0]
        end_entity = entities[1]
        
        # Execute bidirectional search
        paths = self.graph_store.bidirectional_search(
            start_entity=start_entity,
            end_entity=end_entity,
            max_hops=3
        )
        
        if not paths:
            return {
                "error": "No paths found",
                "output_entities": entities,
                "confidence": 0.0,
                "explanation": f"No paths found between {start_entity} and {end_entity}."
            }
        
        # Extract entities and relations from paths
        output_entities = entities.copy()
        relations_used = []
        
        for path in paths:
            for step in path:
                if "source" in step:
                    output_entities.append(step["source"])
                if "target" in step:
                    output_entities.append(step["target"])
                if "relation" in step:
                    relations_used.append(step["relation"])
        
        # Generate path explanation
        path_explanation = self._generate_path_explanation(paths[0])
        
        return {
            "paths": paths,
            "output_entities": list(set(output_entities)),
            "relations_used": list(set(relations_used)),
            "confidence": 1.0,
            "explanation": f"Found {len(paths)} paths between {start_entity} and {end_entity}.",
            "path_explanation": path_explanation
        }
    
    def _weighted_path_step(
        self, 
        query: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a weighted path search step."""
        if not self.graph_store:
            return {
                "error": "Graph store not available",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform weighted path search without a graph store."
            }
        
        # Extract start and end entities
        entities = step_plan.get("entities", [])
        if len(entities) < 2:
            # If we only have one entity or none, try to use current entities
            if state.current_entities:
                entities = list(state.current_entities)
                
        if len(entities) < 2:
            return {
                "error": "Not enough entities for weighted path search",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Need at least two entities for weighted path search."
            }
        
        # Get the start and end entities
        start_entity = entities[0]
        end_entity = entities[1]
        
        # Execute weighted path search
        paths = self.graph_store.weighted_path_query(
            start_entity=start_entity,
            end_entity=end_entity,
            max_hops=3,
            importance_weight=0.6,
            confidence_weight=0.4
        )
        
        if not paths:
            return {
                "error": "No paths found",
                "output_entities": entities,
                "confidence": 0.0,
                "explanation": f"No paths found between {start_entity} and {end_entity}."
            }
        
        # Extract entities and relations from paths
        output_entities = entities.copy()
        relations_used = []
        
        for path in paths:
            for step in path:
                if "source" in step:
                    output_entities.append(step["source"])
                if "target" in step:
                    output_entities.append(step["target"])
                if "relation" in step:
                    relations_used.append(step["relation"])
        
        # Generate path explanation
        path_explanation = self._generate_path_explanation(paths[0])
        
        return {
            "paths": paths,
            "output_entities": list(set(output_entities)),
            "relations_used": list(set(relations_used)),
            "confidence": 1.0,
            "explanation": f"Found {len(paths)} weighted paths between {start_entity} and {end_entity}.",
            "path_explanation": path_explanation
        }
    
    def _constrained_path_step(
        self, 
        query: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a constrained path search step."""
        if not self.graph_store:
            return {
                "error": "Graph store not available",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform constrained path search without a graph store."
            }
        
        # Extract start and end entities
        entities = step_plan.get("entities", [])
        if len(entities) < 2:
            # If we only have one entity or none, try to use current entities
            if state.current_entities:
                entities = list(state.current_entities)
                
        if len(entities) < 2:
            return {
                "error": "Not enough entities for constrained path search",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Need at least two entities for constrained path search."
            }
        
        # Get the start and end entities
        start_entity = entities[0]
        end_entity = entities[1]
        
        # Get allowed relations
        allowed_relations = step_plan.get("focus_relations", None)
        
        # Execute constrained path search
        paths = self.graph_store.constrained_path_search(
            start_entity=start_entity,
            end_entity=end_entity,
            allowed_relations=allowed_relations,
            max_hops=3
        )
        
        if not paths:
            return {
                "error": "No paths found",
                "output_entities": entities,
                "confidence": 0.0,
                "explanation": f"No paths found between {start_entity} and {end_entity} with the given constraints."
            }
        
        # Extract entities and relations from paths
        output_entities = entities.copy()
        relations_used = []
        
        for path in paths:
            for step in path:
                if "source" in step:
                    output_entities.append(step["source"])
                if "target" in step:
                    output_entities.append(step["target"])
                if "relation" in step:
                    relations_used.append(step["relation"])
        
        # Generate path explanation
        path_explanation = self._generate_path_explanation(paths[0])
        
        return {
            "paths": paths,
            "output_entities": list(set(output_entities)),
            "relations_used": list(set(relations_used)),
            "confidence": 1.0,
            "explanation": f"Found {len(paths)} constrained paths between {start_entity} and {end_entity}.",
            "path_explanation": path_explanation
        }
    
    def _default_reasoning_step(
        self, 
        query: str,
        state: ReasoningState,
        step_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a default reasoning step."""
        # Extract entities
        entities = step_plan.get("entities", [])
        if not entities and state.current_entities:
            entities = list(state.current_entities)
            
        if not entities:
            return {
                "error": "No entities for reasoning",
                "output_entities": [],
                "confidence": 0.0,
                "explanation": "Cannot perform reasoning without entities."
            }
        
        # Get entity neighbors for each entity
        neighbors = []
        output_entities = entities.copy()
        relations_used = []
        
        if self.graph_store:
            for entity in entities:
                entity_neighbors = self.graph_store.get_entity_neighbors(
                    entity=entity,
                    direction="both",
                    limit=5
                )
                neighbors.extend(entity_neighbors)
                
                # Extract entities and relations
                for neighbor in entity_neighbors:
                    if "entity" in neighbor:
                        output_entities.append(neighbor["entity"])
                    if "relation" in neighbor:
                        relations_used.append(neighbor["relation"])
        
        return {
            "neighbors": neighbors,
            "output_entities": list(set(output_entities)),
            "relations_used": list(set(relations_used)),
            "confidence": 1.0 if neighbors else 0.0,
            "explanation": f"Explored {len(neighbors)} neighboring entities."
        }
    
    def _generate_path_explanation(self, path: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable explanation of a path.
        
        Args:
            path: Path as a list of steps
            
        Returns:
            Human-readable explanation
        """
        if not path:
            return "No path available to explain."
            
        explanation = []
        
        for i, step in enumerate(path):
            source = step.get("source", "unknown")
            relation = step.get("relation", "unknown relation")
            target = step.get("target", "unknown")
            confidence = step.get("confidence", 0.0)
            
            explanation.append(f"{source} {relation} {target} (confidence: {confidence:.2f})")
        
        return " → ".join(explanation)
    
    def _prepare_result(self, state: ReasoningState) -> Dict[str, Any]:
        """Prepare the final result from a reasoning state."""
        if not state.steps:
            return {
                "answer": "Unable to reason about this query.",
                "confidence": 0.0,
                "reasoning_steps": [],
                "entities_discovered": [],
                "success": False
            }
            
        # Collect all unique entities discovered
        all_entities = set()
        for step in state.steps:
            all_entities.update(step.input_entities)
            all_entities.update(step.output_entities)
            
        # Calculate overall confidence
        overall_confidence = sum(step.confidence for step in state.steps) / len(state.steps)
        
        # Get the reasoning steps
        reasoning_steps = [step.to_dict() for step in state.steps]
        
        # Generate a concise answer
        answer = self._generate_answer(state)
        
        return {
            "answer": answer,
            "confidence": overall_confidence,
            "reasoning_steps": reasoning_steps,
            "entities_discovered": list(all_entities),
            "success": True
        }
    
    def _generate_answer(self, state: ReasoningState) -> str:
        """Generate a concise answer from the reasoning state."""
        if not state.steps:
            return "Unable to reason about this query."
            
        # Get the last step with the highest confidence
        best_step = max(state.steps, key=lambda s: s.confidence)
        
        # Extract the key information
        if best_step.metadata.get("paths"):
            # If we found paths, use the explanation
            return best_step.metadata.get("path_explanation", best_step.explanation)
        elif best_step.metadata.get("hypotheses"):
            # If we generated hypotheses, use the top one
            hypotheses = best_step.metadata.get("hypotheses", [])
            if hypotheses:
                return hypotheses[0].get("explanation", best_step.explanation)
        
        # Default to the step explanation
        return best_step.explanation


# Register the reasoning engine in the module init
def register_reasoning_engine(knowledge_store, config=None):
    """
    Register the reasoning engine with a knowledge store.
    
    Args:
        knowledge_store: Knowledge store instance
        config: Configuration
    """
    reasoning_engine = ReasoningEngine(knowledge_store, config)
    setattr(knowledge_store, 'reasoning_engine', reasoning_engine)
    return reasoning_engine 