"""
Knowledge coordination for CortexFlow.

Extracts knowledge/graph operations from CortexFlowManager into a focused
class.  CortexFlowManager delegates to an instance of
KnowledgeCoordinator for knowledge storage, retrieval, contradiction
handling, uncertainty operations, and multi-hop reasoning.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any

logger = logging.getLogger("cortexflow")


class KnowledgeCoordinator:
    """Coordinates knowledge storage, retrieval, contradiction detection,
    uncertainty handling, and multi-hop graph reasoning."""

    def __init__(
        self,
        config,
        knowledge_store,
        uncertainty_handler,
    ):
        """
        Args:
            config: CortexFlowConfig instance.
            knowledge_store: KnowledgeStore instance.
            uncertainty_handler: UncertaintyHandler or None.
        """
        self.config = config
        self.knowledge_store = knowledge_store
        self.uncertainty_handler = uncertainty_handler

    # ------------------------------------------------------------------
    # Knowledge storage
    # ------------------------------------------------------------------

    def remember_knowledge(
        self, text: str, source: str = None, confidence: float = None
    ) -> list[int]:
        """
        Store important knowledge in the knowledge store.

        DEPRECATED: Use add_knowledge() instead.

        Args:
            text: Text to remember
            source: Optional source of the knowledge
            confidence: Optional confidence value for the knowledge

        Returns:
            List of IDs for the stored knowledge
        """
        warnings.warn(
            "remember_knowledge() is deprecated; use add_knowledge() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_knowledge(text, source, confidence)

    def add_knowledge(
        self, text: str, source: str = None, confidence: float = None
    ) -> list[int]:
        """
        Store important knowledge in the knowledge store.

        Args:
            text: Text to remember
            source: Optional source of the knowledge
            confidence: Optional confidence value for the knowledge

        Returns:
            List of IDs for the stored knowledge
        """
        # Set default confidence if none provided
        if confidence is None:
            confidence = 0.95

        item_ids = self.knowledge_store.add_knowledge(
            text, source=source, confidence=confidence
        )

        # Check for contradictions if enabled
        if self.uncertainty_handler:
            # Support both old and new config structure
            auto_detect = False
            contradiction_strategy = "weighted"

            # Try new config structure first
            if hasattr(self.config, "uncertainty") and hasattr(
                self.config.uncertainty, "auto_detect_contradictions"
            ):
                auto_detect = self.config.uncertainty.auto_detect_contradictions
                contradiction_strategy = (
                    self.config.uncertainty.default_contradiction_strategy
                )
            # Fall back to old config structure
            elif hasattr(self.config, "auto_detect_contradictions"):
                auto_detect = self.config.auto_detect_contradictions
                contradiction_strategy = self.config.default_contradiction_strategy

            if auto_detect:
                try:
                    # Extract entity IDs from the added items
                    entity_ids = []
                    if hasattr(self.knowledge_store, "graph_store"):
                        # Get the entity IDs from the knowledge store's graph store
                        for item_id in item_ids:
                            entity_data = self.knowledge_store.get_knowledge_item(
                                item_id
                            )
                            if entity_data and "entity_id" in entity_data:
                                entity_ids.append(entity_data["entity_id"])

                    # Check for contradictions for each entity
                    for entity_id in entity_ids:
                        contradictions = self.uncertainty_handler.detect_contradictions(
                            entity_id=entity_id
                        )

                        # Auto-resolve contradictions if found
                        for contradiction in contradictions:
                            logger.info(
                                f"Detected contradiction for entity {contradiction.get('entity')}: "
                                f"{contradiction.get('target1')} vs {contradiction.get('target2')}"
                            )

                            # Resolve using the configured strategy
                            resolution = self.uncertainty_handler.resolve_contradiction(
                                contradiction, strategy=contradiction_strategy
                            )

                            logger.info(
                                f"Resolved contradiction using {resolution.get('strategy_used')} strategy: "
                                f"Selected '{resolution.get('resolved_value')}' with confidence {resolution.get('confidence')}"
                            )
                except Exception as e:
                    logger.error(f"Error detecting contradictions: {e}")

        return item_ids

    # ------------------------------------------------------------------
    # Contradiction / uncertainty
    # ------------------------------------------------------------------

    def detect_contradictions(
        self, entity_id=None, relation_type=None, max_results=100
    ) -> list[dict[str, Any]]:
        """
        Detect contradictions in the knowledge graph.

        Args:
            entity_id: Optional entity ID to check
            relation_type: Optional relation type to check
            max_results: Maximum number of results to return

        Returns:
            List of detected contradictions
        """
        if not self.uncertainty_handler:
            logger.warning(
                "Uncertainty handling is not enabled. Cannot detect contradictions."
            )
            return []

        return self.uncertainty_handler.detect_contradictions(
            entity_id=entity_id, relation_type=relation_type, max_results=max_results
        )

    def resolve_contradiction(
        self, contradiction: dict[str, Any], strategy: str = None
    ) -> dict[str, Any]:
        """
        Resolve a contradiction using the specified strategy.

        Args:
            contradiction: Contradiction to resolve
            strategy: Resolution strategy (auto, recency, confidence, reliability, or keep_both)

        Returns:
            Resolution result
        """
        if not self.uncertainty_handler:
            logger.warning(
                "Uncertainty handling is not enabled. Cannot resolve contradictions."
            )
            return {"error": "Uncertainty handling not enabled"}

        if strategy is None:
            strategy = self.config.default_contradiction_strategy

        return self.uncertainty_handler.resolve_contradiction(contradiction, strategy)

    def update_source_reliability(
        self,
        source_name: str,
        reliability_score: float,
        metadata: dict[str, Any] = None,
    ) -> None:
        """
        Update the reliability score for a knowledge source.

        Args:
            source_name: Name of the source
            reliability_score: Reliability score (0.0-1.0)
            metadata: Optional metadata about the source
        """
        if not self.uncertainty_handler:
            logger.warning(
                "Uncertainty handling is not enabled. Cannot update source reliability."
            )
            return

        self.uncertainty_handler.update_source_reliability(
            source_name=source_name,
            reliability_score=reliability_score,
            metadata=metadata,
        )

    def get_source_reliability(self, source_name: str) -> float:
        """
        Get the reliability score for a knowledge source.

        Args:
            source_name: Name of the source

        Returns:
            Reliability score (0.0-1.0)
        """
        if not self.uncertainty_handler:
            logger.warning(
                "Uncertainty handling is not enabled. Cannot get source reliability."
            )
            return 0.5  # Default medium reliability

        return self.uncertainty_handler.get_source_reliability(source_name)

    def add_probability_distribution(
        self,
        entity_id: int,
        relation_id: int,
        distribution_type: str,
        distribution_data: dict[str, Any],
    ) -> None:
        """
        Add a probability distribution to represent uncertainty about a fact.

        Args:
            entity_id: Entity ID
            relation_id: Relation ID
            distribution_type: Type of distribution (discrete, gaussian, etc.)
            distribution_data: Data representing the distribution
        """
        if not self.uncertainty_handler:
            logger.warning(
                "Uncertainty handling is not enabled. Cannot add probability distribution."
            )
            return

        self.uncertainty_handler.add_probability_distribution(
            entity_id=entity_id,
            relation_id=relation_id,
            distribution_type=distribution_type,
            distribution_data=distribution_data,
        )

    def get_probability_distribution(
        self, entity_id: int, relation_id: int
    ) -> dict[str, Any] | None:
        """
        Get the probability distribution for a fact.

        Args:
            entity_id: Entity ID
            relation_id: Relation ID

        Returns:
            Probability distribution data or None if not found
        """
        if not self.uncertainty_handler:
            logger.warning(
                "Uncertainty handling is not enabled. Cannot get probability distribution."
            )
            return None

        return self.uncertainty_handler.get_probability_distribution(
            entity_id=entity_id, relation_id=relation_id
        )

    def reason_with_incomplete_information(
        self, query: dict[str, Any], available_knowledge: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Reason with incomplete information to provide best possible answers.

        Args:
            query: The query to answer
            available_knowledge: Available knowledge to reason with

        Returns:
            Reasoning result with confidence and explanation
        """
        if not self.uncertainty_handler or not self.config.reason_with_incomplete_info:
            logger.warning("Reasoning with incomplete information is not enabled.")
            return {
                "answer": None,
                "confidence": 0,
                "explanation": ["Reasoning with incomplete information is not enabled"],
                "missing_information": [],
            }

        return self.uncertainty_handler.reason_with_incomplete_information(
            query=query, available_knowledge=available_knowledge
        )

    def get_belief_revision_history(
        self, entity_id: int = None, relation_id: int = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get the revision history for beliefs about an entity or relation.

        Args:
            entity_id: Optional entity ID filter
            relation_id: Optional relation ID filter
            limit: Maximum number of revisions to return

        Returns:
            List of belief revisions
        """
        if not self.uncertainty_handler:
            logger.warning(
                "Uncertainty handling is not enabled. Cannot get belief revision history."
            )
            return []

        return self.uncertainty_handler.get_belief_revision_history(
            entity_id=entity_id, relation_id=relation_id, limit=limit
        )

    # ------------------------------------------------------------------
    # Knowledge retrieval
    # ------------------------------------------------------------------

    def get_knowledge(self, query: str) -> list[dict[str, Any]]:
        """
        Get relevant knowledge for a query.

        Args:
            query: Query text

        Returns:
            List of relevant knowledge items
        """
        return self.knowledge_store.get_relevant_knowledge(query)

    # ------------------------------------------------------------------
    # Inference / reasoning via knowledge store
    # ------------------------------------------------------------------

    def answer_why_question(self, query: str) -> list[dict[str, Any]]:
        """
        Answer a why-question using backward chaining logical reasoning.

        Args:
            query: The why question to answer

        Returns:
            Explanation steps for the answer
        """
        if (
            not self.knowledge_store
            or not hasattr(self.knowledge_store, "use_inference_engine")
            or not self.knowledge_store.use_inference_engine
        ):
            return [{"type": "error", "message": "Inference engine is not enabled"}]

        try:
            return self.knowledge_store.inference_engine.answer_why_question(query)
        except Exception as e:
            logger.error(f"Error answering why question: {e}")
            return [
                {"type": "error", "message": f"Error processing question: {str(e)}"}
            ]

    def generate_novel_implications(
        self, iterations: int = None
    ) -> list[dict[str, Any]]:
        """
        Generate novel implications using forward chaining.

        Args:
            iterations: Number of forward chaining iterations (default uses config)

        Returns:
            List of newly inferred facts
        """
        if (
            not self.knowledge_store
            or not hasattr(self.knowledge_store, "use_inference_engine")
            or not self.knowledge_store.use_inference_engine
        ):
            return []

        try:
            if iterations is None:
                iterations = self.config.max_forward_chain_iterations

            return self.knowledge_store.inference_engine.forward_chain(
                iterations=iterations
            )
        except Exception as e:
            logger.error(f"Error generating implications: {e}")
            return []

    def generate_hypotheses(
        self, observation: str, max_hypotheses: int = None
    ) -> list[dict[str, Any]]:
        """
        Generate hypotheses to explain an observation using abductive reasoning.

        Args:
            observation: The observation to explain
            max_hypotheses: Maximum number of hypotheses to generate (default uses config)

        Returns:
            List of hypotheses that could explain the observation
        """
        if (
            not self.knowledge_store
            or not hasattr(self.knowledge_store, "use_inference_engine")
            or not self.knowledge_store.use_inference_engine
        ):
            return []

        try:
            if max_hypotheses is None:
                max_hypotheses = self.config.max_abductive_hypotheses

            return self.knowledge_store.generate_hypotheses(
                observation, max_hypotheses=max_hypotheses
            )
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return []

    def add_logical_rule(
        self,
        name: str,
        premise_patterns: list[dict[str, Any]],
        conclusion_pattern: dict[str, Any],
        confidence: float = 0.8,
    ) -> bool:
        """
        Add a logical rule to the inference engine.

        Args:
            name: Rule name
            premise_patterns: List of premise patterns that must be satisfied
            conclusion_pattern: The conclusion pattern to infer
            confidence: Rule confidence (0.0-1.0)

        Returns:
            Success status
        """
        if (
            not self.knowledge_store
            or not hasattr(self.knowledge_store, "use_inference_engine")
            or not self.knowledge_store.use_inference_engine
        ):
            return False

        try:
            self.knowledge_store.inference_engine.add_rule(
                name=name,
                premise=premise_patterns,
                conclusion=conclusion_pattern,
                confidence=confidence,
            )
            return True
        except Exception as e:
            logger.error(f"Error adding logical rule: {e}")
            return False

    # ------------------------------------------------------------------
    # Multi-hop reasoning
    # ------------------------------------------------------------------

    def multi_hop_query(self, query: str) -> dict[str, Any]:
        """
        Perform multi-hop reasoning on a query.

        Args:
            query: The query text

        Returns:
            Dictionary with path, entities, score, and other reasoning results
        """
        if not hasattr(self, "knowledge_store") or not hasattr(
            self.knowledge_store, "graph_store"
        ):
            logger.error("Graph store not available for multi-hop query")
            return {"path": [], "entities": [], "score": 0.0}

        result = {"path": [], "entities": [], "score": 0.0, "hop_count": 0}

        try:
            # Use the graph search to find relevant information
            graph_results = self.knowledge_store._graph_search(query, max_results=5)

            # Parse entity relations from query
            entity_pair = self._extract_entity_pair(query)

            if entity_pair:
                # Try to find paths between the entities
                start_entity, end_entity = entity_pair
                paths = self.knowledge_store.graph_store.path_query(
                    start_entity=start_entity,
                    end_entity=end_entity,
                    max_hops=self.config.max_graph_hops,
                )

                # If we found paths, format the result
                if paths and len(paths) > 0:
                    best_path = paths[0]  # Use the first path (should be shortest/best)

                    # Format into a linear path
                    formatted_path = []
                    for i, node in enumerate(best_path):
                        # Add the entity
                        formatted_path.append(
                            node.get("entity", f"Entity_{node.get('id')}")
                        )

                        # Add the relation to the next node if not the last node
                        if i < len(best_path) - 1 and "next_relation" in node:
                            formatted_path.append(
                                node["next_relation"].get("type", "related_to")
                            )

                    result["path"] = formatted_path
                    result["entities"] = [
                        node.get("entity", f"Entity_{node.get('id')}")
                        for node in best_path
                    ]
                    result["hop_count"] = len(best_path) - 1
                    result["score"] = 0.8  # Default score

            # If no direct path found, try to extract paths from graph search results
            if not result["path"] and graph_results:
                for item in graph_results:
                    # Look for graph_path type results
                    if item.get("type") == "graph_path":
                        path_text = item.get("text", "")
                        if path_text:
                            # Parse the path text into components
                            arrow = "\u2192"
                            path = [
                                p.strip()
                                for p in path_text.replace(arrow, arrow).split(arrow)
                            ]
                            result["path"] = path
                            result["score"] = item.get("score", 0.5)
                            result["hop_count"] = len(path) - 1 if path else 0
                            break

                    # Collect entities from all results
                    if "entities" in item:
                        result["entities"].extend(item.get("entities", []))

            # Ensure entities list is unique
            if result["entities"]:
                result["entities"] = list(set(result["entities"]))

            # Log the path if found
            if result["path"]:
                arrow = " \u2192 "
                logger.info(
                    f"Found path for query '{query}': {arrow.join(result['path'])}"
                )
            else:
                logger.info(f"No path found for query '{query}'")

            return result

        except Exception as e:
            logger.error(f"Error in multi_hop_query: {e}")
            return {"path": [], "entities": [], "score": 0.0}

    def _extract_entity_pair(self, query: str) -> tuple[str, str] | None:
        """
        Extract a pair of entities from a query for path finding.

        Args:
            query: The query text

        Returns:
            Tuple of (start_entity, end_entity) or None if not found
        """
        # Try to match common query patterns like "connection between X and Y"
        connection_pattern = r"(?:connection|relationship|relation)(?:\s+between\s+)([^,]+?)(?:\s+and\s+)([^?\.]+)"
        match = re.search(connection_pattern, query, re.IGNORECASE)

        if match:
            start_entity = match.group(1).strip()
            end_entity = match.group(2).strip()
            return (start_entity, end_entity)

        # Alternative patterns
        alt_pattern = r"(?:how\s+are|what\s+connects|is\s+there\s+a\s+connection\s+between)\s+([^,]+?)(?:\s+and\s+)([^?\.]+)"
        match = re.search(alt_pattern, query, re.IGNORECASE)

        if match:
            start_entity = match.group(1).strip()
            end_entity = match.group(2).strip()
            return (start_entity, end_entity)

        return None

    # ------------------------------------------------------------------
    # General query routing
    # ------------------------------------------------------------------

    def query(self, query_text: str) -> dict[str, Any]:
        """
        General query interface that routes to specialized query methods.

        Args:
            query_text: The query text

        Returns:
            Query result
        """
        # For multi-hop reasoning queries, use the multi_hop_query method
        if self.config.enable_multi_hop_queries and self._is_multi_hop_query(
            query_text
        ):
            logger.info(f"Routing to multi_hop_query: {query_text}")
            return self.multi_hop_query(query_text)

        # For standard queries, use knowledge retrieval
        logger.info(f"Performing standard knowledge retrieval: {query_text}")
        knowledge_items = self.knowledge_store.get_relevant_knowledge(query_text)

        # Format the result
        result = {
            "items": knowledge_items,
            "answer": self._extract_answer(query_text, knowledge_items),
            "score": max([item.get("score", 0) for item in knowledge_items])
            if knowledge_items
            else 0,
        }

        return result

    def _is_multi_hop_query(self, query: str) -> bool:
        """
        Determine if a query requires multi-hop reasoning.

        Args:
            query: The query text

        Returns:
            True if the query requires multi-hop reasoning
        """
        # Keywords that suggest multi-hop reasoning is needed
        multi_hop_indicators = [
            r"connection between",
            r"relationship between",
            r"related to",
            r"connect",
            r"path",
            r"link",
            r"how are .+ and .+ related",
            r"what is the connection",
            r"how does .+ relate to",
        ]

        # Check if any indicators are present
        for indicator in multi_hop_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                return True

        return False

    def _extract_answer(self, query: str, knowledge_items: list[dict[str, Any]]) -> str:
        """
        Extract an answer from knowledge items.

        Args:
            query: The query text
            knowledge_items: List of knowledge items

        Returns:
            Extracted answer string
        """
        if not knowledge_items:
            return "No information found."

        # Get the highest-scored item
        best_item = max(knowledge_items, key=lambda x: x.get("score", 0))

        # Return the text of the best item
        return best_item.get("text", "Information found but text extraction failed.")
