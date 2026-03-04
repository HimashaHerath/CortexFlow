"""
GraphMerger -- intelligent merging of new information into the knowledge graph.

Handles conflict detection, resolution strategies, and taxonomic relationship
discovery.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

from ._deps import FUZZY_MATCHING_ENABLED, fuzz


class GraphMerger:
    """
    GraphMerger component for intelligently combining new information from multiple sources
    into the existing knowledge graph, with conflict detection and resolution strategies.
    """

    def __init__(self, graph_store):
        """
        Initialize the graph merger.

        Args:
            graph_store: The GraphStore instance to work with
        """
        self.graph_store = graph_store
        self.conn = (
            graph_store.conn
            if graph_store.conn
            else sqlite3.connect(graph_store.db_path)
        )
        self.cursor = self.conn.cursor()

        # Track statistics
        self.stats = {
            "entities_added": 0,
            "entities_updated": 0,
            "entities_merged": 0,
            "relations_added": 0,
            "relations_updated": 0,
            "relations_inferred": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
        }

    def merge_entity(
        self,
        entity: str,
        entity_type: str = None,
        metadata: dict[str, Any] = None,
        provenance: str = None,
        confidence: float = 0.8,
        temporal_start: str = None,
        temporal_end: str = None,
        extraction_method: str = None,
    ) -> int:
        """
        Intelligently merge an entity with existing entities, handling duplicates and conflicts.

        Args:
            entity: Entity text to merge
            entity_type: Type of entity (e.g., PERSON, LOCATION)
            metadata: Additional entity metadata
            provenance: Source of the entity information
            confidence: Confidence score for this entity (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            extraction_method: Method used to extract this entity

        Returns:
            ID of the merged entity
        """
        # Check for exact match
        self.cursor.execute(
            "SELECT id, entity_type, metadata, confidence FROM graph_entities WHERE entity = ?",
            (entity,),
        )
        exact_match = self.cursor.fetchone()

        # Check for fuzzy matches if no exact match
        fuzzy_matches = []
        if not exact_match and FUZZY_MATCHING_ENABLED:
            self.cursor.execute(
                "SELECT id, entity, entity_type, metadata, confidence FROM graph_entities"
            )
            all_entities = self.cursor.fetchall()

            # Find potential matches using fuzzy string matching
            for row in all_entities:
                similarity = fuzz.ratio(entity.lower(), row[1].lower())
                if similarity >= 85:  # Threshold for fuzzy matching
                    fuzzy_matches.append(
                        {
                            "id": row[0],
                            "entity": row[1],
                            "entity_type": row[2],
                            "metadata": json.loads(row[3]) if row[3] else {},
                            "confidence": row[4],
                            "similarity": similarity,
                        }
                    )

            # Sort by similarity
            fuzzy_matches.sort(key=lambda x: x["similarity"], reverse=True)

        if exact_match:
            entity_id = exact_match[0]
            existing_type = exact_match[1]
            existing_metadata_str = exact_match[2]
            existing_confidence = exact_match[3]

            # Parse existing metadata
            if existing_metadata_str:
                try:
                    existing_metadata = json.loads(existing_metadata_str)
                except (TypeError, json.JSONDecodeError) as e:
                    logging.warning(
                        f"Failed to parse existing entity metadata, defaulting to empty: {e}"
                    )
                    existing_metadata = {}
            else:
                existing_metadata = {}

            # Determine if we need to update the existing entity
            should_update = False
            merged_metadata = dict(existing_metadata)  # Make a copy to modify

            # Only update if the new information has higher confidence or adds new metadata
            if confidence > existing_confidence:
                should_update = True

            if metadata:
                # Check if new metadata contains keys not in existing metadata or better values
                for key, value in metadata.items():
                    if key not in existing_metadata:
                        merged_metadata[key] = value
                        should_update = True
                    elif isinstance(value, list) and isinstance(
                        existing_metadata[key], list
                    ):
                        # Merge lists
                        combined = list(set(existing_metadata[key] + value))
                        if len(combined) > len(existing_metadata[key]):
                            merged_metadata[key] = combined
                            should_update = True
                    elif confidence > existing_confidence:
                        # Replace with higher confidence value
                        merged_metadata[key] = value
                        should_update = True
                        # Record the conflict
                        self.stats["conflicts_detected"] += 1
                        self.stats["conflicts_resolved"] += 1

            if should_update:
                # Update the existing entity
                try:
                    self.graph_store.add_entity(
                        entity=entity,
                        entity_type=entity_type or existing_type,
                        metadata=merged_metadata,
                        provenance=provenance,
                        confidence=max(confidence, existing_confidence),
                        temporal_start=temporal_start,
                        temporal_end=temporal_end,
                        extraction_method=extraction_method,
                        changed_by="graph_merger",
                    )
                    self.stats["entities_updated"] += 1
                except Exception as e:
                    logging.error(f"Error updating entity: {e}")

            return entity_id

        elif fuzzy_matches:
            # Use the best fuzzy match
            best_match = fuzzy_matches[0]
            entity_id = best_match["id"]

            # Also update the entity with any new metadata
            merged_metadata = dict(
                best_match["metadata"]
            )  # Start with existing metadata

            if metadata:
                # Merge metadata
                for key, value in metadata.items():
                    if key not in merged_metadata:
                        merged_metadata[key] = value
                    elif isinstance(value, list) and isinstance(
                        merged_metadata[key], list
                    ):
                        merged_metadata[key] = list(set(merged_metadata[key] + value))
                    elif confidence > best_match["confidence"]:
                        merged_metadata[key] = value

            # Add the current entity name as an alias
            merged_metadata["aliases"] = list(
                set(merged_metadata.get("aliases", []) + [entity])
            )

            # Track entity merger event
            merged_metadata["merged_with"] = merged_metadata.get("merged_with", []) + [
                {
                    "entity": entity,
                    "similarity": best_match["similarity"],
                    "timestamp": time.time(),
                    "provenance": provenance,
                }
            ]

            # Add as an alias to the best match
            try:
                self.graph_store.add_entity_alias(
                    entity_id=entity_id,
                    alias=entity,
                    confidence=confidence * (best_match["similarity"] / 100.0),
                )
            except Exception as e:
                logging.warning(f"Couldn't add alias, but continuing: {e}")

            # Update the entity with merged metadata
            try:
                self.graph_store.add_entity(
                    entity=best_match["entity"],
                    entity_type=entity_type or best_match["entity_type"],
                    metadata=merged_metadata,
                    provenance=provenance,
                    confidence=max(confidence, best_match["confidence"]),
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    extraction_method=extraction_method,
                    changed_by="graph_merger",
                )
            except Exception as e:
                logging.error(f"Error updating entity with merged metadata: {e}")

            self.stats["entities_merged"] += 1
            return entity_id

        else:
            # No match found, add as new entity
            try:
                entity_id = self.graph_store.add_entity(
                    entity=entity,
                    entity_type=entity_type,
                    metadata=metadata,
                    provenance=provenance,
                    confidence=confidence,
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    extraction_method=extraction_method,
                    changed_by="graph_merger",
                )
                self.stats["entities_added"] += 1
                return entity_id
            except Exception as e:
                logging.error(f"Error adding new entity: {e}")
                return -1

    def merge_relation(
        self,
        source_entity: str,
        relation_type: str,
        target_entity: str,
        weight: float = 1.0,
        metadata: dict[str, Any] = None,
        provenance: str = None,
        confidence: float = 0.5,
        temporal_start: str = None,
        temporal_end: str = None,
        extraction_method: str = None,
    ) -> bool:
        """
        Intelligently merge a relation with existing relations, handling duplicates and conflicts.

        Args:
            source_entity: Source entity text
            relation_type: Type of relation
            target_entity: Target entity text
            weight: Weight/importance of the relation
            metadata: Additional relation metadata
            provenance: Source of the relation information
            confidence: Confidence score for this relation (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            extraction_method: Method used to extract this relation

        Returns:
            True if relation was merged successfully
        """
        # First, get or merge the source and target entities
        source_id = self.merge_entity(
            entity=source_entity,
            provenance=provenance,
            confidence=confidence,
            extraction_method=extraction_method,
        )

        target_id = self.merge_entity(
            entity=target_entity,
            provenance=provenance,
            confidence=confidence,
            extraction_method=extraction_method,
        )

        # Check if relation already exists
        self.cursor.execute(
            """
            SELECT id, weight, metadata, confidence
            FROM graph_relationships
            WHERE source_id = ? AND target_id = ? AND relation_type = ?
        """,
            (source_id, target_id, relation_type),
        )
        existing = self.cursor.fetchone()

        if existing:
            existing[0]
            existing_weight = existing[1]
            existing_metadata = json.loads(existing[2]) if existing[2] else {}
            existing_confidence = existing[3]

            # Detect conflicts and determine if we should update
            should_update = False

            # Higher confidence or new metadata
            if confidence > existing_confidence:
                should_update = True
            elif metadata:
                # Check for new keys in metadata
                new_keys = set(metadata.keys()) - set(existing_metadata.keys())
                if new_keys:
                    should_update = True

                    # Merge metadata
                    for key in metadata:
                        if key in existing_metadata:
                            # If conflicting values, record conflict and resolve based on confidence
                            if metadata[key] != existing_metadata[key]:
                                self.stats["conflicts_detected"] += 1

                                # If both are lists, merge them
                                if isinstance(metadata[key], list) and isinstance(
                                    existing_metadata[key], list
                                ):
                                    existing_metadata[key] = list(
                                        set(existing_metadata[key] + metadata[key])
                                    )
                                    self.stats["conflicts_resolved"] += 1
                                elif confidence > existing_confidence:
                                    existing_metadata[key] = metadata[key]
                                    self.stats["conflicts_resolved"] += 1
                                # Otherwise keep existing value (implicitly)
                        else:
                            existing_metadata[key] = metadata[key]

                    metadata = existing_metadata

            if should_update:
                # Update with the merged information
                result = self.graph_store.add_relation(
                    source_entity=source_entity,
                    relation_type=relation_type,
                    target_entity=target_entity,
                    weight=max(weight, existing_weight),
                    metadata=metadata,
                    provenance=provenance,
                    confidence=max(confidence, existing_confidence),
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    extraction_method=extraction_method,
                    changed_by="graph_merger",
                )
                self.stats["relations_updated"] += 1
            else:
                result = True  # No update needed

            return result
        else:
            # Add new relation
            result = self.graph_store.add_relation(
                source_entity=source_entity,
                relation_type=relation_type,
                target_entity=target_entity,
                weight=weight,
                metadata=metadata,
                provenance=provenance,
                confidence=confidence,
                temporal_start=temporal_start,
                temporal_end=temporal_end,
                extraction_method=extraction_method,
                changed_by="graph_merger",
            )
            self.stats["relations_added"] += 1
            return result

    def discover_taxonomic_relationships(self) -> int:
        """
        Automatically detect and extract hierarchical relationships (is_a, part_of) from existing entities.

        Returns:
            Number of taxonomic relationships discovered
        """
        # Find all entities
        self.cursor.execute("SELECT id, entity, entity_type FROM graph_entities")
        all_entities = self.cursor.fetchall()

        # Build entity type index
        type_index = {}
        for row in all_entities:
            ent_id, ent_text, ent_type = row
            if ent_type:
                if ent_type not in type_index:
                    type_index[ent_type] = []
                type_index[ent_type].append((ent_id, ent_text))

        discovered = 0

        # Look for patterns suggesting taxonomic relationships
        for row in all_entities:
            ent_id, ent_text, ent_type = row
            # Check if this entity name contains another entity name
            # e.g., "Machine Learning Algorithm" contains "Algorithm"
            words = ent_text.lower().split()
            if len(words) > 1:
                for other_row in all_entities:
                    other_id, other_text, other_type = other_row
                    if (
                        other_id != ent_id
                        and len(other_text.split()) == 1
                        and other_text.lower() in words
                    ):
                        # Check if this looks like an is_a relationship
                        # If entity type matches, it's more likely to be an is_a relationship
                        if ent_type == other_text:
                            self._add_taxonomic_relation(
                                ent_id,
                                ent_text,
                                other_id,
                                other_text,
                                "instance_of",
                                confidence=0.85,
                                provenance="taxonomic_discovery",
                            )
                            discovered += 1
                        else:
                            self._add_taxonomic_relation(
                                ent_id,
                                ent_text,
                                other_id,
                                other_text,
                                "is_a",
                                confidence=0.7,
                                provenance="taxonomic_discovery",
                            )
                            discovered += 1

        # Discover instance_of relationships from entity types
        for ent_type, type_entities in type_index.items():
            # Find if the type exists as an entity
            self.cursor.execute(
                "SELECT id, entity FROM graph_entities WHERE entity = ?", (ent_type,)
            )
            type_entity = self.cursor.fetchone()

            if type_entity:
                type_id, type_text = type_entity
                # Add instance_of relationship for all entities of this type
                for ent_id, ent_text in type_entities:
                    self._add_taxonomic_relation(
                        ent_id,
                        ent_text,
                        type_id,
                        type_text,
                        "instance_of",
                        confidence=0.9,
                        provenance="type_based_taxonomy",
                    )
                    discovered += 1
            else:
                # Create a new entity for this type
                type_id = self.graph_store.add_entity(
                    entity=ent_type,
                    entity_type="TYPE",
                    metadata={"automatic": True, "is_type": True},
                    confidence=0.85,
                    extraction_method="taxonomic_discovery",
                    changed_by="graph_merger",
                )

                # Add instance_of relationship for all entities of this type
                for ent_id, ent_text in type_entities:
                    self._add_taxonomic_relation(
                        ent_id,
                        ent_text,
                        type_id,
                        ent_type,
                        "instance_of",
                        confidence=0.85,
                        provenance="type_based_taxonomy",
                    )
                    discovered += 1

        self.stats["relations_inferred"] += discovered
        return discovered

    def _add_taxonomic_relation(
        self,
        source_id: int,
        source_text: str,
        target_id: int,
        target_text: str,
        relation_type: str,
        confidence: float,
        provenance: str,
    ) -> bool:
        """
        Helper method to add a taxonomic relationship if it doesn't exist.

        Args:
            source_id: Source entity ID
            source_text: Source entity text
            target_id: Target entity ID
            target_text: Target entity text
            relation_type: Type of taxonomic relation (is_a, part_of, etc.)
            confidence: Confidence score
            provenance: Source of this taxonomic relationship

        Returns:
            True if relation was added
        """
        # Check if this relation already exists
        self.cursor.execute(
            """
            SELECT id FROM graph_relationships
            WHERE source_id = ? AND target_id = ? AND relation_type = ?
        """,
            (source_id, target_id, relation_type),
        )

        if not self.cursor.fetchone():
            # Add the relation
            self.graph_store.add_relation(
                source_entity=source_text,
                relation_type=relation_type,
                target_entity=target_text,
                confidence=confidence,
                provenance=provenance,
                extraction_method="taxonomic_inference",
                changed_by="graph_merger",
                metadata={"automatic": True, "taxonomic": True},
            )
            return True

        return False

    def merge_from_text(self, text: str, source: str) -> dict[str, int]:
        """
        Process text to extract entities and relations, then merge them into the graph.

        Args:
            text: Text to process
            source: Source of the text

        Returns:
            Dictionary with count of entities and relations merged
        """
        # Extract entities
        entities = self.graph_store.extract_entities(text)

        # Extract relations
        relations = self.graph_store.extract_relations(text)

        # Merge entities
        processed_entities = {}
        for entity in entities:
            entity_id = self.merge_entity(
                entity=entity["text"],
                entity_type=entity["type"],
                metadata=entity.get("metadata", {}),
                provenance=source,
                confidence=entity.get("confidence", 0.8),
                extraction_method=entity.get("source", "text_extraction"),
            )
            processed_entities[entity["text"]] = entity_id

        # Merge relations
        processed_relations = 0
        for subj, pred, obj in relations:
            if self.merge_relation(
                source_entity=subj,
                relation_type=pred,
                target_entity=obj,
                provenance=source,
                confidence=0.7,  # Default confidence for extracted relations
                extraction_method="text_extraction",
            ):
                processed_relations += 1

        return {"entities": len(processed_entities), "relations": processed_relations}

    def detect_conflicts(self) -> list[dict[str, Any]]:
        """
        Detect conflicting information in the knowledge graph.

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Find relationship conflicts (contradictory relationships)
        # e.g., A is_a B and B is_a A (cycle in taxonomy)
        self.cursor.execute("""
            SELECT r1.source_id, r1.target_id, r1.relation_type, r1.id,
                   r2.source_id, r2.target_id, r2.relation_type, r2.id
            FROM graph_relationships r1
            JOIN graph_relationships r2 ON r1.target_id = r2.source_id
                                       AND r2.target_id = r1.source_id
                                       AND r1.relation_type = r2.relation_type
            WHERE r1.relation_type IN ('is_a', 'subclass_of', 'instance_of')
              AND r1.source_id < r2.target_id  -- Avoid duplicates
        """)

        for row in self.cursor.fetchall():
            # Get entity names
            self.cursor.execute(
                "SELECT entity FROM graph_entities WHERE id = ?", (row[0],)
            )
            source1 = self.cursor.fetchone()[0]

            self.cursor.execute(
                "SELECT entity FROM graph_entities WHERE id = ?", (row[1],)
            )
            target1 = self.cursor.fetchone()[0]

            conflicts.append(
                {
                    "type": "cycle",
                    "description": f"Taxonomic cycle detected: {source1} {row[2]} {target1} and {target1} {row[6]} {source1}",
                    "relation_ids": [row[3], row[7]],
                    "entities": [source1, target1],
                    "relation_type": row[2],
                }
            )

        # Find attribute conflicts (different values for the same attribute)
        self.cursor.execute("""
            SELECT r1.source_id, r1.relation_type, r1.target_id, r1.confidence, r1.id,
                   r2.target_id, r2.confidence, r2.id
            FROM graph_relationships r1
            JOIN graph_relationships r2 ON r1.source_id = r2.source_id
                                       AND r1.relation_type = r2.relation_type
                                       AND r1.target_id != r2.target_id
            WHERE r1.relation_type LIKE 'has_%'
              AND r1.id < r2.id  -- Avoid duplicates
        """)

        for row in self.cursor.fetchall():
            # Get entity names
            self.cursor.execute(
                "SELECT entity FROM graph_entities WHERE id = ?", (row[0],)
            )
            source = self.cursor.fetchone()[0]

            self.cursor.execute(
                "SELECT entity FROM graph_entities WHERE id = ?", (row[2],)
            )
            target1 = self.cursor.fetchone()[0]

            self.cursor.execute(
                "SELECT entity FROM graph_entities WHERE id = ?", (row[5],)
            )
            target2 = self.cursor.fetchone()[0]

            conflicts.append(
                {
                    "type": "attribute",
                    "description": f"Attribute conflict: {source} {row[1]} {target1} (confidence: {row[3]}) vs {source} {row[1]} {target2} (confidence: {row[6]})",
                    "relation_ids": [row[4], row[7]],
                    "entities": [source, target1, target2],
                    "relation_type": row[1],
                }
            )

        return conflicts

    def resolve_conflicts(self, conflict_resolution: str = "confidence") -> int:
        """
        Automatically resolve detected conflicts in the knowledge graph.

        Args:
            conflict_resolution: Strategy for resolving conflicts:
                - 'confidence': Keep the relation with higher confidence
                - 'recency': Keep the more recent relation
                - 'provenance': Prioritize by source reliability

        Returns:
            Number of conflicts resolved
        """
        conflicts = self.detect_conflicts()
        resolved = 0

        for conflict in conflicts:
            if conflict["type"] == "cycle":
                # For taxonomy cycles, keep the relation with higher confidence or recency
                self.cursor.execute(
                    """
                    SELECT id, confidence, timestamp, provenance FROM graph_relationships
                    WHERE id IN (?, ?)
                """,
                    (conflict["relation_ids"][0], conflict["relation_ids"][1]),
                )

                relations = self.cursor.fetchall()

                if conflict_resolution == "confidence":
                    # Keep the relation with higher confidence
                    if relations[0][1] >= relations[1][1]:
                        relation_to_remove = relations[1][0]
                    else:
                        relation_to_remove = relations[0][0]

                elif conflict_resolution == "recency":
                    # Keep the more recent relation
                    if relations[0][2] >= relations[1][2]:
                        relation_to_remove = relations[1][0]
                    else:
                        relation_to_remove = relations[0][0]

                elif conflict_resolution == "provenance":
                    # Keep the relation from more reliable source
                    # This is a placeholder - implement source reliability logic
                    relation_to_remove = relations[1][0]

                # Delete the relation to be removed
                self.cursor.execute(
                    """
                    DELETE FROM graph_relationships WHERE id = ?
                """,
                    (relation_to_remove,),
                )

                resolved += 1

            elif conflict["type"] == "attribute":
                # For attribute conflicts, keep the attribute with higher confidence
                self.cursor.execute(
                    """
                    SELECT id, confidence FROM graph_relationships
                    WHERE id IN (?, ?)
                """,
                    (conflict["relation_ids"][0], conflict["relation_ids"][1]),
                )

                relations = self.cursor.fetchall()

                if conflict_resolution == "confidence":
                    # Keep the relation with higher confidence
                    if relations[0][1] >= relations[1][1]:
                        relation_to_remove = relations[1][0]
                    else:
                        relation_to_remove = relations[0][0]

                elif conflict_resolution == "recency":
                    # Similar logic as above
                    self.cursor.execute(
                        """
                        SELECT id, timestamp FROM graph_relationships
                        WHERE id IN (?, ?)
                    """,
                        (conflict["relation_ids"][0], conflict["relation_ids"][1]),
                    )

                    recency_data = self.cursor.fetchall()
                    if recency_data[0][1] >= recency_data[1][1]:
                        relation_to_remove = recency_data[1][0]
                    else:
                        relation_to_remove = recency_data[0][0]

                elif conflict_resolution == "provenance":
                    # Placeholder for source reliability
                    relation_to_remove = relations[1][0]

                # Delete the relation to be removed
                self.cursor.execute(
                    """
                    DELETE FROM graph_relationships WHERE id = ?
                """,
                    (relation_to_remove,),
                )

                resolved += 1

        self.conn.commit()
        self.stats["conflicts_resolved"] += resolved
        return resolved

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about the merging operations.

        Returns:
            Dictionary with counts of various operations
        """
        return self.stats
