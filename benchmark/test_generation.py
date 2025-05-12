"""
Automatic Test Generation for CortexFlow Reasoning Capabilities.

This module provides functionality to automatically generate test cases
for evaluating various reasoning capabilities of the CortexFlow system.
"""

import random
import json
import datetime
from typing import List, Dict, Any, Tuple, Optional, Set, Callable
import networkx as nx
import re

class ReasoningTestGenerator:
    """Generate test cases for reasoning capability assessment."""
    
    def __init__(self, graph_store=None, knowledge_store=None):
        """
        Initialize the test generator.
        
        Args:
            graph_store: Optional reference to the graph store
            knowledge_store: Optional reference to the knowledge store
        """
        self.graph_store = graph_store
        self.knowledge_store = knowledge_store
        self.test_suites = {}
        
        # Default entity types for testing
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", 
            "TECHNOLOGY", "EVENT", "PRODUCT", "FIELD"
        ]
        
        # Default relation types for testing
        self.relation_types = [
            "created", "invented", "discovered", "works_for", "founded", 
            "located_in", "part_of", "used_for", "specializes_in"
        ]
    
    def _extract_graph_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and relations from the graph store.
        
        Returns:
            Tuple of (entities, relations)
        """
        if not self.graph_store:
            return [], []
        
        entities = []
        relations = []
        
        # Extract entities
        if hasattr(self.graph_store, "graph") and self.graph_store.graph:
            for node_id in self.graph_store.graph.nodes():
                node_data = self.graph_store.graph.nodes[node_id]
                entity = {
                    "id": node_id,
                    "name": node_data.get("name", f"Entity_{node_id}"),
                    "type": node_data.get("type", "UNKNOWN")
                }
                entities.append(entity)
            
            # Extract relations
            for src, dst, data in self.graph_store.graph.edges(data=True):
                relation = {
                    "subject": self.graph_store.graph.nodes[src].get("name", f"Entity_{src}"),
                    "predicate": data.get("type", "related_to"),
                    "object": self.graph_store.graph.nodes[dst].get("name", f"Entity_{dst}"),
                    "weight": data.get("weight", 1.0)
                }
                relations.append(relation)
        
        return entities, relations
    
    def _get_db_entities_relations(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get entities and relations from the database.
        
        Returns:
            Tuple of (entities, relations)
        """
        if not self.knowledge_store:
            return [], []
        
        entities = []
        relations = []
        
        # Get entities and relations from the knowledge store
        if hasattr(self.knowledge_store, "get_all_entities"):
            entities = self.knowledge_store.get_all_entities()
        
        if hasattr(self.knowledge_store, "get_all_relations"):
            relations = self.knowledge_store.get_all_relations()
        
        return entities, relations
    
    def generate_multi_hop_tests(self, num_tests: int = 10) -> List[Dict[str, Any]]:
        """
        Generate test cases for multi-hop reasoning.
        
        Args:
            num_tests: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        tests = []
        
        # Get entities and relations
        entities, relations = self._extract_graph_data()
        
        if not entities or not relations:
            entities, relations = self._get_db_entities_relations()
        
        if not entities or not relations:
            # Create synthetic test data if no real data available
            return self._generate_synthetic_multi_hop_tests(num_tests)
        
        # Build a directed graph for path finding
        G = nx.DiGraph()
        
        # Add nodes
        for entity in entities:
            G.add_node(entity["id"], name=entity["name"], type=entity["type"])
        
        # Add edges
        entity_name_to_id = {entity["name"]: entity["id"] for entity in entities}
        
        for relation in relations:
            subject = relation["subject"]
            object_ = relation["object"]
            
            if subject in entity_name_to_id and object_ in entity_name_to_id:
                subject_id = entity_name_to_id[subject]
                object_id = entity_name_to_id[object_]
                
                G.add_edge(
                    subject_id, 
                    object_id, 
                    predicate=relation["predicate"],
                    weight=relation.get("weight", 1.0)
                )
        
        # Generate test cases
        for _ in range(num_tests):
            # Pick random nodes that have paths between them
            connected_nodes = []
            
            # Try to find connected nodes (max 10 attempts)
            for _ in range(10):
                if len(G.nodes()) < 2:
                    break
                    
                source_id = random.choice(list(G.nodes()))
                targets = []
                
                # Find nodes that are 2 or 3 hops away
                for node in G.nodes():
                    if node == source_id:
                        continue
                        
                    try:
                        paths = list(nx.all_simple_paths(G, source_id, node, cutoff=3))
                        if paths and any(len(path) >= 3 for path in paths):  # At least 2 hops
                            targets.append((node, paths))
                    except nx.NetworkXNoPath:
                        continue
                
                if targets:
                    target_id, paths = random.choice(targets)
                    # Choose a path with 2 or 3 hops
                    valid_paths = [p for p in paths if len(p) >= 3]
                    if valid_paths:
                        path = random.choice(valid_paths)
                        connected_nodes = path
                        break
            
            if not connected_nodes:
                continue
            
            # Create test case
            source_name = G.nodes[connected_nodes[0]].get("name", f"Entity_{connected_nodes[0]}")
            target_name = G.nodes[connected_nodes[-1]].get("name", f"Entity_{connected_nodes[-1]}")
            
            # Create expected path
            expected_path = []
            for i in range(len(connected_nodes) - 1):
                src_id = connected_nodes[i]
                dst_id = connected_nodes[i + 1]
                
                src_name = G.nodes[src_id].get("name", f"Entity_{src_id}")
                dst_name = G.nodes[dst_id].get("name", f"Entity_{dst_id}")
                
                edge_data = G.get_edge_data(src_id, dst_id)
                predicate = edge_data.get("predicate", "related_to") if edge_data else "related_to"
                
                expected_path.append(src_name)
                expected_path.append(predicate)
            
            expected_path.append(target_name)
            
            # Create expected entities (all entities in the path)
            expected_entities = [G.nodes[node_id].get("name", f"Entity_{node_id}") for node_id in connected_nodes]
            
            # Generate query
            query = f"What is the connection between {source_name} and {target_name}?"
            
            # Create test case
            test_case = {
                "query": query,
                "expected_path": expected_path,
                "expected_entities": expected_entities,
                "hop_count": len(connected_nodes) - 1,
                "source": source_name,
                "target": target_name
            }
            
            tests.append(test_case)
        
        # If we couldn't generate enough tests, supplement with synthetic ones
        if len(tests) < num_tests:
            synthetic_tests = self._generate_synthetic_multi_hop_tests(num_tests - len(tests))
            tests.extend(synthetic_tests)
        
        return tests[:num_tests]  # Ensure we return exactly num_tests
    
    def _generate_synthetic_multi_hop_tests(self, num_tests: int) -> List[Dict[str, Any]]:
        """
        Generate synthetic test cases for multi-hop reasoning.
        
        Args:
            num_tests: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        tests = []
        
        # Sample entities for different domains
        domains = {
            "tech": {
                "entities": ["Python", "JavaScript", "TensorFlow", "PyTorch", "Google", "Microsoft", "Amazon", "Facebook"],
                "relations": ["created", "developed", "uses", "supports", "invested_in"]
            },
            "science": {
                "entities": ["Einstein", "Newton", "Relativity", "Gravity", "Physics", "Mathematics", "CERN", "NASA"],
                "relations": ["discovered", "formulated", "proved", "studied", "works_for"]
            },
            "business": {
                "entities": ["Elon Musk", "Jeff Bezos", "Tesla", "SpaceX", "Blue Origin", "Silicon Valley", "Wall Street"],
                "relations": ["founded", "leads", "invested_in", "located_in", "competes_with"]
            }
        }
        
        # Generate test cases
        for _ in range(num_tests):
            # Choose a random domain
            domain_name = random.choice(list(domains.keys()))
            domain = domains[domain_name]
            
            # Choose random source and target entities
            if len(domain["entities"]) < 2:
                continue
                
            source, target = random.sample(domain["entities"], 2)
            
            # Create a random path between them
            path_length = random.randint(2, 4)  # 1-3 hops
            
            # Ensure path length is achievable with available entities
            max_possible_length = min(path_length, len(domain["entities"]))
            
            # Generate path entities
            path_entities = [source]
            remaining_entities = [e for e in domain["entities"] if e != source and e != target]
            
            # Add intermediate entities
            for _ in range(max_possible_length - 2):  # -2 for source and target
                if not remaining_entities:
                    break
                    
                entity = random.choice(remaining_entities)
                path_entities.append(entity)
                remaining_entities.remove(entity)
            
            path_entities.append(target)
            
            # Generate relations for the path
            expected_path = []
            for i in range(len(path_entities) - 1):
                src = path_entities[i]
                dst = path_entities[i + 1]
                
                expected_path.append(src)
                expected_path.append(random.choice(domain["relations"]))
            
            expected_path.append(target)
            
            # Generate query
            query = f"What is the connection between {source} and {target}?"
            
            # Create test case
            test_case = {
                "query": query,
                "expected_path": expected_path,
                "expected_entities": path_entities,
                "hop_count": len(path_entities) - 1,
                "source": source,
                "target": target,
                "synthetic": True
            }
            
            tests.append(test_case)
        
        return tests
    
    def generate_counterfactual_tests(self, num_tests: int = 5) -> List[Dict[str, Any]]:
        """
        Generate test cases for counterfactual reasoning.
        
        Args:
            num_tests: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        tests = []
        
        # Get entities and relations
        entities, relations = self._extract_graph_data()
        
        if not entities or not relations:
            entities, relations = self._get_db_entities_relations()
        
        if not entities or not relations:
            # Create synthetic test data if no real data available
            return self._generate_synthetic_counterfactual_tests(num_tests)
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Create map of existing relations
        existing_relations = set()
        for relation in relations:
            subject = relation["subject"]
            predicate = relation["predicate"]
            object_ = relation["object"]
            existing_relations.add((subject, predicate, object_))
        
        # Generate test cases
        for _ in range(num_tests):
            # Strategy 1: Create a non-existent relation between existing entities
            if len(entities) >= 2:
                entity1, entity2 = random.sample(entities, 2)
                entity1_name = entity1.get("name", f"Entity_{entity1.get('id')}")
                entity2_name = entity2.get("name", f"Entity_{entity2.get('id')}")
                
                # Choose a relation type that doesn't exist between these entities
                available_predicates = [r for r in self.relation_types 
                                        if (entity1_name, r, entity2_name) not in existing_relations]
                
                if available_predicates:
                    predicate = random.choice(available_predicates)
                    
                    # Create query
                    if random.random() < 0.5:
                        query = f"Is there a relation where {entity1_name} {predicate} {entity2_name}?"
                    else:
                        query = f"Did {entity1_name} {predicate} {entity2_name}?"
                    
                    # Create test case
                    test_case = {
                        "query": query,
                        "expected_entities": [entity1_name, entity2_name],
                        "expected_path": [],
                        "hop_count": 0,
                        "counterfactual": True,
                        "description": f"False relation: {entity1_name} {predicate} {entity2_name}"
                    }
                    
                    tests.append(test_case)
            
            # Strategy 2: Mix entities from different domains that shouldn't be connected
            if len(entities_by_type) >= 2:
                # Choose two different entity types
                available_types = [t for t in entities_by_type if len(entities_by_type[t]) > 0]
                if len(available_types) >= 2:
                    type1, type2 = random.sample(available_types, 2)
                    
                    # Choose random entities of these types
                    entity1 = random.choice(entities_by_type[type1])
                    entity2 = random.choice(entities_by_type[type2])
                    
                    entity1_name = entity1.get("name", f"Entity_{entity1.get('id')}")
                    entity2_name = entity2.get("name", f"Entity_{entity2.get('id')}")
                    
                    # Create query
                    query = f"What is the connection between {entity1_name} and {entity2_name}?"
                    
                    # Create test case
                    test_case = {
                        "query": query,
                        "expected_entities": [entity1_name, entity2_name],
                        "expected_path": [],
                        "hop_count": 0,
                        "counterfactual": True,
                        "description": f"No connection between {entity1_name} and {entity2_name}"
                    }
                    
                    tests.append(test_case)
        
        # If we couldn't generate enough tests, supplement with synthetic ones
        if len(tests) < num_tests:
            synthetic_tests = self._generate_synthetic_counterfactual_tests(num_tests - len(tests))
            tests.extend(synthetic_tests)
        
        return tests[:num_tests]  # Ensure we return exactly num_tests
    
    def _generate_synthetic_counterfactual_tests(self, num_tests: int) -> List[Dict[str, Any]]:
        """
        Generate synthetic test cases for counterfactual reasoning.
        
        Args:
            num_tests: Number of test cases to generate
            
        Returns:
            List of test cases
        """
        tests = []
        
        # Sample entities for different domains
        domains = {
            "tech": ["Python", "JavaScript", "Google", "Microsoft"],
            "history": ["Napoleon", "Alexander the Great", "Rome", "Egypt"],
            "physics": ["Einstein", "Newton", "Quantum Mechanics", "Relativity"],
            "fiction": ["Harry Potter", "Luke Skywalker", "Middle Earth", "Narnia"]
        }
        
        # Sample predicates
        predicates = ["created", "discovered", "founded", "visited", "conquered", "studied"]
        
        for _ in range(num_tests):
            # Choose entities from different domains
            domain1, domain2 = random.sample(list(domains.keys()), 2)
            entity1 = random.choice(domains[domain1])
            entity2 = random.choice(domains[domain2])
            
            # Choose a random predicate
            predicate = random.choice(predicates)
            
            # Create query (50% yes/no questions, 50% open questions)
            if random.random() < 0.5:
                query = f"Did {entity1} {predicate} {entity2}?"
            else:
                query = f"What is the connection between {entity1} and {entity2}?"
            
            # Create test case
            test_case = {
                "query": query,
                "expected_entities": [entity1, entity2],
                "expected_path": [],
                "hop_count": 0,
                "counterfactual": True,
                "synthetic": True,
                "description": f"Synthetic counterfactual between domains {domain1} and {domain2}"
            }
            
            tests.append(test_case)
        
        return tests
    
    def generate_test_suite(self, suite_name: str = "default", 
                            num_multi_hop: int = 10, 
                            num_counterfactual: int = 5) -> Dict[str, Any]:
        """
        Generate a complete test suite with different types of tests.
        
        Args:
            suite_name: Name of the test suite
            num_multi_hop: Number of multi-hop reasoning tests
            num_counterfactual: Number of counterfactual reasoning tests
            
        Returns:
            Dictionary with test suite information
        """
        # Generate tests
        multi_hop_tests = self.generate_multi_hop_tests(num_multi_hop)
        counterfactual_tests = self.generate_counterfactual_tests(num_counterfactual)
        
        # Create test suite
        test_suite = {
            "name": suite_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "tests": {
                "multi_hop": multi_hop_tests,
                "counterfactual": counterfactual_tests
            },
            "metadata": {
                "total_tests": len(multi_hop_tests) + len(counterfactual_tests),
                "num_multi_hop": len(multi_hop_tests),
                "num_counterfactual": len(counterfactual_tests)
            }
        }
        
        # Store test suite
        self.test_suites[suite_name] = test_suite
        
        return test_suite
    
    def save_test_suite(self, suite_name: str, file_path: str) -> bool:
        """
        Save a test suite to a file.
        
        Args:
            suite_name: Name of the test suite to save
            file_path: Path to the output file
            
        Returns:
            True if successful, False otherwise
        """
        if suite_name not in self.test_suites:
            return False
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.test_suites[suite_name], f, indent=2)
            return True
        except Exception:
            return False
    
    def load_test_suite(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a test suite from a file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Dictionary with test suite information, or None if loading failed
        """
        try:
            with open(file_path, 'r') as f:
                test_suite = json.load(f)
            
            if "name" in test_suite:
                self.test_suites[test_suite["name"]] = test_suite
            
            return test_suite
        except Exception:
            return None 