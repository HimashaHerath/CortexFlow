"""
CortexFlow Inference Engine module.

This module provides logical reasoning capabilities over the knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union, Generator
import json
import time
import re
import copy

# Try importing graph libraries
try:
    import networkx as nx
    NETWORKX_ENABLED = True
except ImportError:
    NETWORKX_ENABLED = False
    logging.warning("networkx not found. Inference engine capabilities will be limited.")

from cortexflow.config import CortexFlowConfig
from cortexflow.interfaces import KnowledgeStoreInterface

class LogicalRule:
    """Represents a logical rule for inference."""
    
    def __init__(self, 
                 name: str,
                 premise: List[Dict[str, Any]], 
                 conclusion: Dict[str, Any],
                 confidence: float = 0.8,
                 metadata: Dict[str, Any] = None):
        """
        Initialize a logical rule.
        
        Args:
            name: Rule name
            premise: List of conditions that must be true for the rule to apply
            conclusion: The conclusion to derive if premises are true
            confidence: Rule confidence (0.0-1.0)
            metadata: Additional rule metadata
        """
        self.name = name
        self.premise = premise
        self.conclusion = conclusion
        self.confidence = confidence
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        """String representation of the rule."""
        return f"Rule({self.name}: IF {self.premise} THEN {self.conclusion})"


class InferenceEngine:
    """Provides logical reasoning capabilities over the knowledge graph."""
    
    def __init__(self, knowledge_store: KnowledgeStoreInterface, config: CortexFlowConfig = None):
        """
        Initialize the inference engine.
        
        Args:
            knowledge_store: Knowledge store instance
            config: Configuration for the inference engine
        """
        self.knowledge_store = knowledge_store
        self.config = config or CortexFlowConfig()
        
        # Get reference to graph store
        self.graph_store = getattr(knowledge_store, 'graph_store', None)
        if not self.graph_store:
            logging.warning("Graph store not available. Inference engine capabilities will be limited.")
        
        # Knowledge base of rules
        self.rules: List[LogicalRule] = []
        
        # Cache for inferred facts to avoid redundant computation
        self.inference_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize rule base with default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize the rule base with default logical rules."""
        # Transitivity rule for "is_a" relations
        self.add_rule(
            name="transitivity_is_a",
            premise=[
                {"source": "?X", "relation": "is_a", "target": "?Y"},
                {"source": "?Y", "relation": "is_a", "target": "?Z"}
            ],
            conclusion={"source": "?X", "relation": "is_a", "target": "?Z"},
            confidence=0.9,
            metadata={"category": "transitivity"}
        )
        
        # Transitivity rule for "part_of" relations
        self.add_rule(
            name="transitivity_part_of",
            premise=[
                {"source": "?X", "relation": "part_of", "target": "?Y"},
                {"source": "?Y", "relation": "part_of", "target": "?Z"}
            ],
            conclusion={"source": "?X", "relation": "part_of", "target": "?Z"},
            confidence=0.85,
            metadata={"category": "transitivity"}
        )
        
        # Rule for inheritance of properties
        self.add_rule(
            name="property_inheritance",
            premise=[
                {"source": "?X", "relation": "is_a", "target": "?Y"},
                {"source": "?Y", "relation": "has_property", "target": "?P"}
            ],
            conclusion={"source": "?X", "relation": "has_property", "target": "?P"},
            confidence=0.8,
            metadata={"category": "inheritance"}
        )
    
    def add_rule(self, name: str, premise: List[Dict[str, Any]], 
                 conclusion: Dict[str, Any], confidence: float = 0.8,
                 metadata: Dict[str, Any] = None) -> None:
        """
        Add a new logical rule to the rule base.
        
        Args:
            name: Rule name
            premise: List of conditions that must be true for the rule to apply
            conclusion: The conclusion to derive if premises are true
            confidence: Rule confidence (0.0-1.0)
            metadata: Additional rule metadata
        """
        rule = LogicalRule(name, premise, conclusion, confidence, metadata)
        self.rules.append(rule)
        logging.info(f"Added inference rule: {rule.name}")
    
    def get_rules(self) -> List[LogicalRule]:
        """Get all rules in the rule base."""
        return self.rules
    
    def clear_cache(self) -> None:
        """Clear the inference cache."""
        self.inference_cache.clear()
    
    def forward_chain(self, iterations: int = 3) -> List[Dict[str, Any]]:
        """
        Apply forward chaining to derive new facts.
        
        Args:
            iterations: Maximum number of iterations to run
            
        Returns:
            List of newly inferred facts
        """
        if not self.graph_store:
            logging.warning("Graph store not available. Forward chaining cannot be performed.")
            return []
        
        inferred_facts = []
        iteration = 0
        
        while iteration < iterations:
            new_facts_in_iteration = []
            
            # For each rule in the knowledge base
            for rule in self.rules:
                # Get variable bindings that satisfy the premises
                bindings_list = self._match_premises(rule.premise)
                
                # Apply each binding to the conclusion
                for bindings in bindings_list:
                    inferred_fact = self._apply_bindings(rule.conclusion, bindings)
                    
                    # Check if this is a new fact
                    if self._is_new_fact(inferred_fact):
                        fact_id = self._add_inferred_fact(inferred_fact, rule)
                        if fact_id:
                            inferred_fact["id"] = fact_id
                            inferred_fact["rule"] = rule.name
                            inferred_fact["confidence"] = rule.confidence
                            new_facts_in_iteration.append(inferred_fact)
                            inferred_facts.append(inferred_fact)
            
            # If no new facts were derived in this iteration, we've reached a fixed point
            if not new_facts_in_iteration:
                break
                
            iteration += 1
            logging.info(f"Forward chaining iteration {iteration}: {len(new_facts_in_iteration)} new facts")
        
        return inferred_facts
    
    def backward_chain(self, query: Dict[str, Any], depth: int = 3) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Apply backward chaining to answer a query.
        
        Args:
            query: The query to answer (in the form of a fact pattern)
            depth: Maximum recursion depth
            
        Returns:
            Tuple of (success, explanation trail)
        """
        if not self.graph_store:
            logging.warning("Graph store not available. Backward chaining cannot be performed.")
            return False, []
        
        # Check if the query is directly provable from the knowledge base
        if self._is_fact_in_kb(query):
            return True, [{"fact": query, "source": "knowledge_base", "confidence": 1.0}]
        
        # If we've reached the maximum depth, stop recursion
        if depth <= 0:
            return False, []
        
        explanation = []
        
        # Find rules whose conclusions match the query
        matching_rules = self._find_rules_for_query(query)
        
        # Try each matching rule
        for rule in matching_rules:
            premises_satisfied = True
            premise_explanations = []
            
            # Variable bindings from matching the conclusion
            initial_bindings = self._match_conclusion(rule.conclusion, query)
            
            if not initial_bindings:
                continue
                
            # Check each premise with the initial bindings
            for premise in rule.premise:
                # Apply the current bindings to the premise
                bound_premise = self._apply_bindings(premise, initial_bindings)
                
                # Recursively prove the premise
                premise_satisfied, premise_explanation = self.backward_chain(bound_premise, depth - 1)
                
                if not premise_satisfied:
                    premises_satisfied = False
                    break
                    
                premise_explanations.extend(premise_explanation)
            
            # If all premises are satisfied, we've proven the query
            if premises_satisfied:
                # Add the rule application to the explanation
                explanation = premise_explanations + [{
                    "rule_applied": rule.name,
                    "result": query,
                    "confidence": rule.confidence
                }]
                
                return True, explanation
        
        # If we get here, no rules could prove the query
        return False, explanation
    
    def abductive_reasoning(self, observation: Dict[str, Any], max_hypotheses: int = 3) -> List[Dict[str, Any]]:
        """
        Perform abductive reasoning to generate hypotheses explaining an observation.
        
        Args:
            observation: The observed fact to explain
            max_hypotheses: Maximum number of hypotheses to generate
            
        Returns:
            List of hypotheses that could explain the observation
        """
        if not self.graph_store:
            logging.warning("Graph store not available. Abductive reasoning cannot be performed.")
            return []
        
        hypotheses = []
        
        # Find rules whose conclusions match the observation
        matching_rules = self._find_rules_for_query(observation)
        
        # For each matching rule, construct hypotheses
        for rule in matching_rules:
            # Variable bindings from matching the conclusion
            bindings = self._match_conclusion(rule.conclusion, observation)
            
            if not bindings:
                continue
            
            # Create hypotheses from the premises
            for premise in rule.premise:
                # Apply bindings to the premise
                hypothesis = self._apply_bindings(premise, bindings)
                
                # Check if this is a known fact
                is_known = self._is_fact_in_kb(hypothesis)
                
                # Add to hypotheses with appropriate confidence
                hypotheses.append({
                    "hypothesis": hypothesis,
                    "confidence": rule.confidence * (0.9 if is_known else 0.5),
                    "rule": rule.name,
                    "is_known": is_known
                })
                
                # Limit the number of hypotheses
                if len(hypotheses) >= max_hypotheses:
                    break
            
            # Limit the number of hypotheses
            if len(hypotheses) >= max_hypotheses:
                break
        
        # Sort hypotheses by confidence
        hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        
        return hypotheses[:max_hypotheses]
    
    def answer_why_question(self, query: str) -> List[Dict[str, Any]]:
        """
        Answer a 'why' question using backward chaining.
        
        Args:
            query: The why question to answer
            
        Returns:
            List of explanation steps
        """
        # Extract the fact pattern from the why question
        fact_pattern = self._extract_fact_from_question(query)
        
        if not fact_pattern:
            return [{"error": "Could not extract a clear fact pattern from the question"}]
        
        # Apply backward chaining
        is_proven, explanation = self.backward_chain(fact_pattern)
        
        if is_proven:
            return self._format_explanation(explanation, fact_pattern)
        else:
            # Try abductive reasoning if backward chaining fails
            hypotheses = self.abductive_reasoning(fact_pattern)
            
            if hypotheses:
                return [{
                    "type": "hypothesis",
                    "message": "The fact could not be proven directly, but these hypotheses might explain it:",
                    "hypotheses": hypotheses
                }]
            else:
                return [{
                    "type": "negative",
                    "message": "The system could not find evidence to prove or explain this fact."
                }]
    
    def _extract_fact_from_question(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Extract a fact pattern from a why question.
        
        Args:
            question: The why question
            
        Returns:
            Extracted fact pattern or None
        """
        # Remove "why" prefix and question mark
        clean_question = question.lower().replace("why ", "").replace("?", "").strip()
        
        # Try to extract subject-predicate-object pattern
        # This is a simplified extraction and should be replaced with a more sophisticated NLP approach
        
        # Try simple patterns like "X is Y" or "X has Y"
        is_pattern = re.match(r"(\w+)\s+is\s+(\w+)", clean_question)
        has_pattern = re.match(r"(\w+)\s+has\s+(\w+)", clean_question)
        
        if is_pattern:
            source, target = is_pattern.groups()
            return {
                "source": source,
                "relation": "is_a",
                "target": target
            }
        elif has_pattern:
            source, target = has_pattern.groups()
            return {
                "source": source,
                "relation": "has_property",
                "target": target
            }
        else:
            # Try to extract entities and a relation from the graph store
            if self.graph_store:
                entities = self.graph_store.extract_entities(clean_question)
                if len(entities) >= 2:
                    relations = self.graph_store.extract_relations(clean_question)
                    if relations:
                        # Use the first relation found
                        s, p, o = relations[0]
                        return {
                            "source": s,
                            "relation": p,
                            "target": o
                        }
            
            # Fallback: Try to split the sentence into parts
            parts = clean_question.split()
            if len(parts) >= 3:
                return {
                    "source": parts[0],
                    "relation": parts[1],
                    "target": parts[2]
                }
                
        return None
    
    def _format_explanation(self, explanation: List[Dict[str, Any]], original_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format the explanation trail in a user-friendly way.
        
        Args:
            explanation: The explanation trail from backward chaining
            original_query: The original query
            
        Returns:
            Formatted explanation
        """
        formatted = [{
            "type": "query",
            "message": f"Finding explanation for: {self._fact_to_str(original_query)}"
        }]
        
        # Track the confidence level
        min_confidence = 1.0
        
        for step in explanation:
            if "fact" in step:
                fact_str = self._fact_to_str(step["fact"])
                formatted.append({
                    "type": "fact",
                    "message": f"Known fact: {fact_str}",
                    "confidence": step.get("confidence", 1.0)
                })
                min_confidence = min(min_confidence, step.get("confidence", 1.0))
            elif "rule_applied" in step:
                formatted.append({
                    "type": "inference",
                    "message": f"Applied rule '{step['rule_applied']}' to derive: {self._fact_to_str(step['result'])}",
                    "confidence": step.get("confidence", 0.8)
                })
                min_confidence = min(min_confidence, step.get("confidence", 0.8))
        
        # Add final conclusion
        formatted.append({
            "type": "conclusion",
            "message": f"Therefore, {self._fact_to_str(original_query)} is established.",
            "confidence": min_confidence
        })
        
        return formatted
    
    def _fact_to_str(self, fact: Dict[str, Any]) -> str:
        """Convert a fact dictionary to a string."""
        return f"{fact.get('source', '?')} {fact.get('relation', '?')} {fact.get('target', '?')}"
    
    def _match_premises(self, premises: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Find all variable bindings that satisfy the premises.
        
        Args:
            premises: List of premise patterns with variables
            
        Returns:
            List of variable bindings (dictionaries mapping variable names to values)
        """
        if not premises:
            return [{}]  # Empty binding satisfies empty premises
            
        if not self.graph_store:
            return []
        
        # Start with the first premise
        first_premise = premises[0]
        bindings_list = self._find_bindings_for_pattern(first_premise)
        
        # If only one premise, return its bindings
        if len(premises) == 1:
            return bindings_list
        
        # Process remaining premises
        for premise in premises[1:]:
            new_bindings_list = []
            
            for bindings in bindings_list:
                # Apply current bindings to the premise
                bound_premise = self._apply_bindings(premise, bindings)
                
                # Find new bindings that satisfy this premise and are consistent with existing bindings
                additional_bindings = self._find_bindings_for_pattern(bound_premise)
                
                for additional in additional_bindings:
                    # Check consistency between existing and new bindings
                    merged_bindings = self._merge_bindings(bindings, additional)
                    if merged_bindings:
                        new_bindings_list.append(merged_bindings)
            
            # Update the bindings list
            bindings_list = new_bindings_list
            
            # If no bindings satisfy the premises so far, return empty list
            if not bindings_list:
                return []
        
        return bindings_list
    
    def _find_bindings_for_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Find variable bindings that satisfy a single pattern.
        
        Args:
            pattern: Fact pattern with variables
            
        Returns:
            List of variable bindings
        """
        bindings_list = []
        
        # Extract components and identify variables
        source = pattern.get("source", "")
        relation = pattern.get("relation", "")
        target = pattern.get("target", "")
        
        source_is_var = source.startswith("?")
        relation_is_var = relation.startswith("?")
        target_is_var = target.startswith("?")
        
        # If no variables, just check if the fact exists
        if not source_is_var and not relation_is_var and not target_is_var:
            if self._is_fact_in_kb(pattern):
                return [{}]  # Empty binding indicates the constant pattern is satisfied
            return []
        
        # Query the graph store based on known components
        query_params = {}
        if not source_is_var:
            query_params["source_entity"] = source
        if not relation_is_var:
            query_params["relation_type"] = relation
        if not target_is_var:
            query_params["target_entity"] = target
        
        # Query for matching facts
        matching_facts = self._query_graph_facts(**query_params)
        
        # Create bindings for each matching fact
        for fact in matching_facts:
            bindings = {}
            
            if source_is_var:
                bindings[source[1:]] = fact["source"]
            if relation_is_var:
                bindings[relation[1:]] = fact["relation"]
            if target_is_var:
                bindings[target[1:]] = fact["target"]
            
            bindings_list.append(bindings)
        
        return bindings_list
    
    def _apply_bindings(self, pattern: Dict[str, Any], bindings: Dict[str, str]) -> Dict[str, Any]:
        """
        Apply variable bindings to a pattern.
        
        Args:
            pattern: Fact pattern with variables
            bindings: Variable bindings to apply
            
        Returns:
            Pattern with variables replaced by their bindings
        """
        result = copy.deepcopy(pattern)
        
        for key, value in result.items():
            if isinstance(value, str) and value.startswith("?"):
                var_name = value[1:]  # Remove '?' prefix
                if var_name in bindings:
                    result[key] = bindings[var_name]
        
        return result
    
    def _merge_bindings(self, bindings1: Dict[str, str], bindings2: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Merge two sets of variable bindings if they are consistent.
        
        Args:
            bindings1: First set of bindings
            bindings2: Second set of bindings
            
        Returns:
            Merged bindings or None if inconsistent
        """
        result = copy.deepcopy(bindings1)
        
        for var, value in bindings2.items():
            if var in result and result[var] != value:
                return None  # Inconsistent bindings
            result[var] = value
        
        return result
    
    def _find_rules_for_query(self, query: Dict[str, Any]) -> List[LogicalRule]:
        """
        Find rules whose conclusions match the given query.
        
        Args:
            query: The query to match against rule conclusions
            
        Returns:
            List of matching rules
        """
        matching_rules = []
        
        for rule in self.rules:
            # Check if the rule's conclusion pattern matches the query
            if self._patterns_match(rule.conclusion, query):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _match_conclusion(self, conclusion: Dict[str, Any], query: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Match a rule conclusion against a query, extracting variable bindings.
        
        Args:
            conclusion: The rule conclusion with variables
            query: The query to match
            
        Returns:
            Variable bindings or None if no match
        """
        bindings = {}
        
        for key, value in conclusion.items():
            if key not in query:
                return None
                
            query_value = query[key]
            
            if isinstance(value, str) and value.startswith("?"):
                # This is a variable, bind it to the query value
                var_name = value[1:]  # Remove '?' prefix
                if var_name in bindings and bindings[var_name] != query_value:
                    return None  # Inconsistent binding
                bindings[var_name] = query_value
            elif value != query_value:
                return None  # Constant values don't match
        
        return bindings
    
    def _patterns_match(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """
        Check if two patterns are compatible (can be unified).
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            True if patterns can be unified
        """
        for key in pattern1:
            if key not in pattern2:
                return False
                
            value1 = pattern1[key]
            value2 = pattern2[key]
            
            # If either value is a variable, they can match
            if isinstance(value1, str) and value1.startswith("?"):
                continue
            if isinstance(value2, str) and value2.startswith("?"):
                continue
                
            # If both are constants, they must be equal
            if value1 != value2:
                return False
        
        return True
    
    def _is_fact_in_kb(self, fact: Dict[str, Any]) -> bool:
        """
        Check if a fact exists in the knowledge base.
        
        Args:
            fact: The fact to check
            
        Returns:
            True if the fact exists
        """
        if not self.graph_store:
            return False
        
        source = fact.get("source", "")
        relation = fact.get("relation", "")
        target = fact.get("target", "")
        
        # Skip if any component is a variable
        if any(isinstance(v, str) and v.startswith("?") for v in [source, relation, target]):
            return False
        
        # Check for the fact in the graph store
        relations = self.graph_store.query_relations(
            source_entity=source,
            relation_type=relation,
            target_entity=target,
            limit=1
        )
        
        return len(relations) > 0
    
    def _is_new_fact(self, fact: Dict[str, Any]) -> bool:
        """
        Check if a fact is new (not already in the knowledge base).
        
        Args:
            fact: The fact to check
            
        Returns:
            True if the fact is new
        """
        # Skip if any component is a variable
        source = fact.get("source", "")
        relation = fact.get("relation", "")
        target = fact.get("target", "")
        
        if any(isinstance(v, str) and v.startswith("?") for v in [source, relation, target]):
            return False
        
        # Create a fact key for cache lookup
        fact_key = f"{source}|{relation}|{target}"
        
        # Check cache first
        if fact_key in self.inference_cache:
            return False
        
        # Check if fact exists in the knowledge base
        if self._is_fact_in_kb(fact):
            # Add to cache
            self.inference_cache[fact_key] = fact
            return False
        
        return True
    
    def _add_inferred_fact(self, fact: Dict[str, Any], rule: LogicalRule) -> Optional[int]:
        """
        Add an inferred fact to the knowledge base.
        
        Args:
            fact: The fact to add
            rule: The rule that produced this fact
            
        Returns:
            ID of the added fact, or None if addition failed
        """
        if not self.graph_store:
            return None
        
        source = fact.get("source", "")
        relation = fact.get("relation", "")
        target = fact.get("target", "")
        
        # Skip if any component is a variable
        if any(isinstance(v, str) and v.startswith("?") for v in [source, relation, target]):
            return None
        
        # Add to cache
        fact_key = f"{source}|{relation}|{target}"
        self.inference_cache[fact_key] = fact
        
        try:
            # Add the relation to the graph store
            result = self.graph_store.add_relation(
                source_entity=source,
                relation_type=relation,
                target_entity=target,
                confidence=rule.confidence,
                metadata={
                    "inferred": True,
                    "rule": rule.name,
                    "timestamp": time.time()
                },
                provenance="inference_engine"
            )
            
            # Get the relation ID
            if result:
                relations = self.graph_store.query_relations(
                    source_entity=source,
                    relation_type=relation,
                    target_entity=target,
                    limit=1
                )
                
                if relations:
                    return relations[0].get("id")
            
            return None
        except Exception as e:
            logging.error(f"Error adding inferred fact to graph: {e}")
            return None
    
    def _query_graph_facts(self, source_entity: str = None, relation_type: str = None, 
                         target_entity: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query facts from the graph store.
        
        Args:
            source_entity: Optional source entity filter
            relation_type: Optional relation type filter
            target_entity: Optional target entity filter
            limit: Maximum number of results
            
        Returns:
            List of matching facts
        """
        if not self.graph_store:
            return []
        
        try:
            relations = self.graph_store.query_relations(
                source_entity=source_entity,
                relation_type=relation_type,
                target_entity=target_entity,
                limit=limit
            )
            
            facts = []
            for relation in relations:
                facts.append({
                    "source": relation.get("source_entity", ""),
                    "relation": relation.get("relation_type", ""),
                    "target": relation.get("target_entity", ""),
                    "confidence": relation.get("confidence", 0.5),
                    "id": relation.get("id", 0)
                })
                
            return facts
        except Exception as e:
            logging.error(f"Error querying graph facts: {e}")
            return [] 