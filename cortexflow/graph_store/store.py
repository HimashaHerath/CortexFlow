"""
Main GraphStore class -- knowledge graph storage and query functionality.

Handles DB connection, entity/relation CRUD, search operations, entity
extraction (NER), and relation extraction.  Path-finding methods are
provided by :class:`~cortexflow.graph_store.traversal.TraversalMixin`.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Any

from cortexflow.config import CortexFlowConfig

from ._deps import (
    FLAIR_ENABLED,
    FUZZY_MATCHING_ENABLED,
    NETWORKX_ENABLED,
    ONTOLOGY_ENABLED,
    SPACY_ENABLED,
    SPANBERT_ENABLED,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Ontology,
    Sentence,
    SequenceTagger,
    fuzz,
    nx,
    process,
    spacy,
    torch,
)
from .relation_extractor import RelationExtractor, _extract_svo_triples_from_sentence
from .schema import (
    VALID_COL_TYPE_PATTERN,
    VALID_TABLE_NAMES,
    add_metadata_columns,
    create_indexes,
    ensure_schema,
    validate_ddl_identifier,
)
from .traversal import TraversalMixin


class GraphStore(TraversalMixin):
    """Knowledge graph storage and query functionality for GraphRAG."""

    # Class-level singletons for heavy NLP models (loaded once per process)
    _flair_ner = None
    _flair_loaded = False
    _spanbert_tokenizer = None
    _spanbert_model = None
    _spanbert_loaded = False

    # Safety limit for graph path enumeration to prevent DoS on dense graphs
    MAX_PATHS_TO_ENUMERATE = 100

    # DDL allowlists for safe schema migration
    _VALID_TABLE_NAMES = VALID_TABLE_NAMES
    _VALID_COL_TYPE_PATTERN = VALID_COL_TYPE_PATTERN

    @staticmethod
    def _validate_ddl_identifier(value: str, allowed: set) -> None:
        """Validate a DDL identifier against an allowlist.

        Args:
            value: The identifier to validate (table name or column name).
            allowed: Set of permitted values.

        Raises:
            ValueError: If *value* is not in *allowed*.
        """
        validate_ddl_identifier(value, allowed)

    def __init__(self, config: CortexFlowConfig):
        """
        Initialize graph store.

        Args:
            config: CortexFlow configuration
        """
        self.config = config
        self.db_path = config.knowledge_store_path

        # For in-memory databases, we need to maintain a persistent connection
        self.conn = None
        if self.db_path == ":memory:":
            # For Windows compatibility, use empty string instead of ":memory:"
            self.conn = sqlite3.connect("")

        # Initialize NetworkX graph if available
        self.graph = nx.DiGraph() if NETWORKX_ENABLED else None

        # Initialize NER model for entity extraction if available
        self.nlp = None
        if SPACY_ENABLED:
            try:
                # Use a small efficient model by default
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("Spacy NER model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading Spacy model: {e}")

        # Flair and SpanBERT are class-level singletons (lazy-loaded on first use)

        # Initialize entity linking system
        self.entity_db = {}  # Map of canonical entities
        self.entity_embeddings = {}  # For semantic similarity between entities

        # Initialize relation extractor
        self.relation_extractor = RelationExtractor(self.nlp)

        # Initialize ontology if available
        self.ontology = None
        if ONTOLOGY_ENABLED:
            try:
                self.ontology = Ontology(self.db_path)
                logging.info("Ontology system initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing ontology: {e}")

        # Initialize graph database tables
        self._init_db()

        # Load existing graph from database
        self._load_graph_from_db()

        # Load entity database for entity linking
        self._load_entity_db()

        logging.info(
            f"Graph store initialized with NetworkX: {NETWORKX_ENABLED}, Spacy: {SPACY_ENABLED}, Flair: {FLAIR_ENABLED}, SpanBERT: {SPANBERT_ENABLED}, Ontology: {ONTOLOGY_ENABLED}"
        )

    def _ensure_flair_loaded(self):
        """Lazily load Flair NER model on first use (class-level singleton)."""
        if not GraphStore._flair_loaded and FLAIR_ENABLED:
            GraphStore._flair_loaded = True
            try:
                GraphStore._flair_ner = SequenceTagger.load(
                    "flair/ner-english-ontonotes-large"
                )
                logging.info("Flair NER model loaded successfully")
            except Exception as e:
                logging.warning(
                    f"Advanced NER is disabled (Flair model failed to load): {e}"
                )

    def _ensure_spanbert_loaded(self):
        """Lazily load SpanBERT model on first use (class-level singleton)."""
        if not GraphStore._spanbert_loaded and SPANBERT_ENABLED:
            GraphStore._spanbert_loaded = True
            try:
                GraphStore._spanbert_tokenizer = AutoTokenizer.from_pretrained(
                    "SpanBERT/spanbert-base-cased"
                )
                GraphStore._spanbert_model = (
                    AutoModelForTokenClassification.from_pretrained(
                        "SpanBERT/spanbert-base-cased"
                    )
                )
                logging.info("SpanBERT model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading SpanBERT model: {e}")

    def _load_entity_db(self):
        """Load existing entity database for entity linking."""
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Load all entities for linking
            cursor.execute(
                "SELECT id, entity, entity_type, metadata FROM graph_entities"
            )
            entities = cursor.fetchall()

            for entity in entities:
                canonical_name = entity["entity"]
                entity_id = entity["id"]
                entity_type = entity["entity_type"]
                metadata = json.loads(entity["metadata"]) if entity["metadata"] else {}

                # Store entity in the linking database
                self.entity_db[canonical_name] = {
                    "id": entity_id,
                    "type": entity_type,
                    "metadata": metadata,
                    "aliases": metadata.get("aliases", []),
                }

                # Add aliases to the lookup
                for alias in metadata.get("aliases", []):
                    self.entity_db[alias] = {
                        "canonical": canonical_name,
                        "id": entity_id,
                    }

        except Exception as e:
            logging.error(f"Error loading entity database: {e}")

        if self.conn is None:
            conn.close()

    def _init_db(self):
        """Initialize the SQLite database with required tables for graph storage."""
        if self.conn is not None:
            # We're using an in-memory database
            cursor = self.conn.cursor()
        else:
            # Using a file-based database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

        # Create tables and insert basic relation types
        ensure_schema(cursor)

        # Migrate existing tables to add columns that may be missing
        add_metadata_columns(cursor)

        # Create indexes for faster lookups (after migration so all columns exist)
        create_indexes(cursor)

        if self.conn is not None:
            self.conn.commit()
        else:
            conn.commit()
            conn.close()

    def _load_graph_from_db(self):
        """Load existing graph data from the database."""
        if not NETWORKX_ENABLED:
            return

        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Load all entities
            cursor.execute(
                "SELECT id, entity, entity_type, metadata FROM graph_entities"
            )
            entities = cursor.fetchall()

            for entity in entities:
                metadata = json.loads(entity["metadata"]) if entity["metadata"] else {}
                self.graph.add_node(
                    entity["id"],
                    name=entity["entity"],
                    entity_type=entity["entity_type"],
                    **metadata,
                )

            # Load all relationships
            cursor.execute("""
                SELECT source_id, target_id, relation_type, weight, metadata
                FROM graph_relationships
            """)
            relationships = cursor.fetchall()

            for rel in relationships:
                metadata = json.loads(rel["metadata"]) if rel["metadata"] else {}
                weight = rel["weight"] if rel["weight"] is not None else 1.0
                self.graph.add_edge(
                    rel["source_id"],
                    rel["target_id"],
                    relation=rel["relation_type"],
                    weight=weight,
                    **metadata,
                )

            logging.info(
                f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )

        except sqlite3.OperationalError as e:
            logging.error(f"Error loading graph from database: {e}")

        finally:
            if self.conn is None:
                conn.close()

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Extract named entities from text using multiple techniques including
        advanced NER models, entity linking, and fuzzy matching.

        NOTE: Entity extraction also exists in knowledge.py for search/retrieval purposes.
        This implementation is graph-focused (supports entity linking, multi-model NER,
        and domain-specific extraction) while knowledge.py's version is search-focused.
        Both are intentionally maintained as separate implementations.

        Args:
            text: Input text to extract entities from

        Returns:
            List of entity dictionaries with text, type, start, end
        """
        entities = []

        # 1. First try SpaCy NER if available
        if SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)

                # Get named entities from spaCy's NER
                for ent in doc.ents:
                    entities.append(
                        {
                            "text": ent.text,
                            "type": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "source": "spacy",
                        }
                    )

                # 2. Coreference resolution
                # NOTE: neuralcoref was removed (deprecated since 2020, incompatible with modern spaCy).
                # Coreference resolution is disabled until a replacement library is integrated.
                # Candidates: coreferee, spacy-experimental coref, or fastcoref.
            except Exception as e:
                logging.error(f"Error in SpaCy NER: {e}")

        # 3. Use Flair for NER if available (lazy-loaded on first use)
        self._ensure_flair_loaded()
        if FLAIR_ENABLED and GraphStore._flair_ner is not None:
            try:
                # Create Flair sentence
                flair_sentence = Sentence(text)

                # Run NER
                GraphStore._flair_ner.predict(flair_sentence)

                # Extract entities
                for entity in flair_sentence.get_spans("ner"):
                    # Calculate character offsets
                    start_pos = text.find(entity.text)
                    if start_pos >= 0:
                        end_pos = start_pos + len(entity.text)

                        # Check for overlap with existing entities
                        is_new_entity = True
                        for existing_entity in entities:
                            if (
                                start_pos >= existing_entity["start"]
                                and start_pos < existing_entity["end"]
                            ) or (
                                end_pos > existing_entity["start"]
                                and end_pos <= existing_entity["end"]
                            ):
                                is_new_entity = False
                                break

                        if is_new_entity:
                            entities.append(
                                {
                                    "text": entity.text,
                                    "type": entity.tag,
                                    "start": start_pos,
                                    "end": end_pos,
                                    "score": entity.score,
                                    "source": "flair",
                                }
                            )
            except Exception as e:
                logging.error(f"Error in Flair NER: {e}")

        # 4. Use SpanBERT for NER if available (lazy-loaded on first use)
        self._ensure_spanbert_loaded()
        if (
            SPANBERT_ENABLED
            and GraphStore._spanbert_model is not None
            and GraphStore._spanbert_tokenizer is not None
        ):
            try:
                # Tokenize input
                inputs = GraphStore._spanbert_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )

                # Get model predictions
                with torch.no_grad():
                    outputs = GraphStore._spanbert_model(**inputs)

                # Process predictions to extract entities
                # This is a simplified implementation and would need to be adapted for the specific model
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
                tokens = GraphStore._spanbert_tokenizer.convert_ids_to_tokens(
                    inputs.input_ids.squeeze().tolist()
                )

                # Map predictions to entity spans (simplified)
                current_entity = None
                current_type = None
                current_start = 0

                # Skip special tokens like [CLS]
                offset = 1
                char_offset = 0

                for i, (token, prediction) in enumerate(
                    zip(tokens[offset:], predictions[offset:])
                ):
                    # Skip special tokens
                    if token.startswith("##") or token in ["[SEP]", "[PAD]"]:
                        continue

                    # Basic BIO scheme processing (simplification)
                    if prediction > 0:  # Non-O tag
                        # Get entity type (simplified mapping)
                        entity_type = f"TYPE_{prediction}"

                        if current_entity is None:
                            # Start of new entity
                            current_entity = token.replace("##", "")
                            current_type = entity_type
                            current_start = char_offset
                        else:
                            # Continue current entity
                            current_entity += " " + token.replace("##", "")
                    else:
                        # End of entity
                        if current_entity is not None:
                            # Add entity if not already overlapping
                            entity_end = current_start + len(current_entity)

                            # Check for overlap
                            is_new_entity = True
                            for entity in entities:
                                if (
                                    current_start >= entity["start"]
                                    and current_start < entity["end"]
                                ) or (
                                    entity_end > entity["start"]
                                    and entity_end <= entity["end"]
                                ):
                                    is_new_entity = False
                                    break

                            if is_new_entity:
                                entities.append(
                                    {
                                        "text": current_entity,
                                        "type": current_type,
                                        "start": current_start,
                                        "end": entity_end,
                                        "source": "spanbert",
                                    }
                                )

                            current_entity = None
                            current_type = None

                    # Update character offset (simplified)
                    char_offset += len(token) + 1

                # Add final entity if there is one
                if current_entity is not None:
                    entity_end = current_start + len(current_entity)
                    entities.append(
                        {
                            "text": current_entity,
                            "type": current_type,
                            "start": current_start,
                            "end": entity_end,
                            "source": "spanbert",
                        }
                    )

            except Exception as e:
                logging.error(f"Error in SpanBERT NER: {e}")

        # 5. Add pattern-based entity extraction
        patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "URL": r"https?://\S+",
            "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "TIME": r"\b\d{1,2}:\d{2}\b",
            "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "NUMBER": r"\b\d+(?:\.\d+)?\b",
            "PERCENTAGE": r"\b\d+(?:\.\d+)?%\b",
            "MONEY": r"\$\d+(?:\.\d+)?\b",
            "HASHTAG": r"#[A-Za-z][A-Za-z0-9_]*",
            "MENTION": r"@[A-Za-z][A-Za-z0-9_]*",
        }

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                # Check if this match overlaps with existing entities
                overlap = False
                for entity in entities:
                    if (
                        match.start() >= entity["start"]
                        and match.start() < entity["end"]
                    ) or (
                        match.end() > entity["start"] and match.end() <= entity["end"]
                    ):
                        overlap = True
                        break

                if not overlap:
                    entities.append(
                        {
                            "text": match.group(0),
                            "type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "source": "pattern",
                        }
                    )

        # 6. Add noun phrase extraction if no entities found yet or to supplement
        if SPACY_ENABLED and self.nlp is not None:
            try:
                if "doc" not in locals():  # Only parse if we haven't already
                    doc = self.nlp(text)

                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2:  # Skip very short chunks
                        # Check for overlap with existing entities
                        overlap = False
                        for entity in entities:
                            if (
                                chunk.start_char >= entity["start"]
                                and chunk.start_char < entity["end"]
                            ) or (
                                chunk.end_char > entity["start"]
                                and chunk.end_char <= entity["end"]
                            ):
                                overlap = True
                                break

                        if not overlap:
                            entities.append(
                                {
                                    "text": chunk.text,
                                    "type": "NOUN_PHRASE",
                                    "start": chunk.start_char,
                                    "end": chunk.end_char,
                                    "source": "noun_chunk",
                                }
                            )

                # Add proper nouns not already captured
                for token in doc:
                    if token.pos_ == "PROPN" and len(token.text) > 2:
                        # Check if already part of an entity
                        is_part_of_entity = False
                        for entity in entities:
                            if (
                                token.idx >= entity["start"]
                                and token.idx + len(token.text) <= entity["end"]
                            ):
                                is_part_of_entity = True
                                break

                        if not is_part_of_entity:
                            entities.append(
                                {
                                    "text": token.text,
                                    "type": "PROPER_NOUN",
                                    "start": token.idx,
                                    "end": token.idx + len(token.text),
                                    "source": "pos_tag",
                                }
                            )
            except Exception as e:
                logging.error(f"Error extracting noun phrases: {e}")

        # 7. Add domain-specific entity extraction
        try:
            # Check for domain-specific patterns based on config
            domain_entities = self._extract_domain_specific_entities(text)
            for entity in domain_entities:
                # Check for overlap
                overlap = False
                for existing_entity in entities:
                    if (
                        entity["start"] >= existing_entity["start"]
                        and entity["start"] < existing_entity["end"]
                    ) or (
                        entity["end"] > existing_entity["start"]
                        and entity["end"] <= existing_entity["end"]
                    ):
                        overlap = True
                        break

                if not overlap:
                    entity["source"] = "domain"
                    entities.append(entity)
        except Exception as e:
            logging.error(f"Error in domain-specific entity extraction: {e}")

        # 8. Perform entity linking to connect mentions to canonical entities
        linked_entities = []
        for entity in entities:
            entity_text = entity["text"]
            linked_entity = self._link_entity(entity_text)

            if linked_entity:
                # Copy original entity and add linking information
                linked_entity_data = entity.copy()
                linked_entity_data["canonical"] = (
                    linked_entity["canonical"]
                    if "canonical" in linked_entity
                    else entity_text
                )
                linked_entity_data["entity_id"] = linked_entity["id"]
                linked_entity_data["linked"] = True

                # Use canonical entity type if available
                if "type" in linked_entity and linked_entity["type"]:
                    linked_entity_data["canonical_type"] = linked_entity["type"]

                linked_entities.append(linked_entity_data)
            else:
                # No linking found, keep original entity
                entity["linked"] = False
                linked_entities.append(entity)

        return linked_entities

    def _link_entity(self, entity_text: str) -> dict[str, Any] | None:
        """
        Link entity mention to canonical entity using exact, fuzzy, and embedding-based matching.

        Args:
            entity_text: Text of the entity mention to link

        Returns:
            Dictionary with linking information or None if no match found
        """
        # First try exact match
        if entity_text in self.entity_db:
            return self.entity_db[entity_text]

        # Next try case-insensitive match
        entity_lower = entity_text.lower()
        for key in self.entity_db:
            if key.lower() == entity_lower:
                return self.entity_db[key]

        # Try fuzzy matching if available
        if FUZZY_MATCHING_ENABLED:
            try:
                # Get only canonical entities (not aliases)
                canonical_entities = [
                    key
                    for key in self.entity_db
                    if "canonical" not in self.entity_db[key]
                ]

                # Find closest match with threshold
                matches = process.extractBests(
                    entity_text,
                    canonical_entities,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=85,
                    limit=1,
                )

                if matches and len(matches) > 0:
                    match, score = matches[0]
                    result = self.entity_db[match].copy()
                    result["match_score"] = score
                    result["match_type"] = "fuzzy"
                    return result
            except Exception as e:
                logging.error(f"Error in fuzzy entity matching: {e}")

        # No match found
        return None

    def _extract_domain_specific_entities(self, text: str) -> list[dict[str, Any]]:
        """
        Extract domain-specific entities based on patterns.
        This can be extended for specialized knowledge domains.

        Args:
            text: Input text to extract domain entities from

        Returns:
            List of entity dictionaries
        """
        domain_entities = []

        # Example: Extract programming language entities
        programming_langs = [
            "Python",
            "JavaScript",
            "TypeScript",
            "Java",
            "C++",
            "C#",
            "Go",
            "Rust",
            "Swift",
            "Kotlin",
            "Ruby",
            "PHP",
            "SQL",
            "R",
            "MATLAB",
            "Scala",
            "Perl",
            "Haskell",
            "Clojure",
            "Erlang",
            "Elixir",
            "Julia",
        ]

        # Look for programming languages
        for lang in programming_langs:
            for match in re.finditer(r"\b" + re.escape(lang) + r"\b", text):
                domain_entities.append(
                    {
                        "text": match.group(0),
                        "type": "PROGRAMMING_LANGUAGE",
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        # Example: Extract ML/AI techniques
        ml_techniques = [
            "Neural Network",
            "Deep Learning",
            "Machine Learning",
            "Natural Language Processing",
            "Computer Vision",
            "Reinforcement Learning",
            "Transformer",
            "BERT",
            "GPT",
            "CNN",
            "RNN",
            "LSTM",
            "GAN",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "K-means",
            "PCA",
            "t-SNE",
            "XGBoost",
        ]

        # Look for ML/AI terms
        for technique in ml_techniques:
            pattern = r"\b" + re.escape(technique) + r"\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                domain_entities.append(
                    {
                        "text": match.group(0),
                        "type": "AI_ML_TERM",
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        return domain_entities

    def extract_relations(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract subject-predicate-object triples from text using semantic role labeling
        and enhanced dependency parsing techniques.

        NOTE: The RelationExtractor class also provides relation extraction. This method
        is the GraphStore-level implementation that includes coreference-based extraction
        and fallback heuristics. The RelationExtractor instance is available as
        self.relation_extractor for standalone relation extraction needs.

        Args:
            text: Input text to extract relations from

        Returns:
            List of (subject, predicate, object) tuples
        """
        relations = []

        if SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)

                # 1. First attempt with semantic role labeling if available
                try:
                    srl_relations = self._extract_with_semantic_roles(text)
                    if srl_relations:
                        relations.extend(srl_relations)
                except Exception as e:
                    logging.warning(f"SRL extraction not available: {e}")

                # 2. Process each sentence separately with enhanced dependency parsing
                for sent in doc.sents:
                    # Extract all subject-verb-object patterns
                    extracted_svo = self._extract_svo_from_dependency(sent)
                    relations.extend(extracted_svo)

                    # Extract prep relations (X in Y, X on Y, etc.)
                    extracted_preps = self._extract_prep_relations(sent)
                    relations.extend(extracted_preps)

                # 3. Add coreference-based relations if available
                try:
                    coref_relations = self._extract_with_coreference(text)
                    if coref_relations:
                        relations.extend(coref_relations)
                except Exception as e:
                    logging.warning(
                        f"Coreference relation extraction not available: {e}"
                    )

                # 4. If no relations found with advanced methods, fall back to simpler approaches
                if not relations:
                    entities = self.extract_entities(text)
                    if len(entities) >= 2:
                        # Create relations between consecutive entities with heuristics
                        for i in range(len(entities) - 1):
                            entity1 = entities[i]
                            entity2 = entities[i + 1]

                            # Get text between entities
                            between_start = entity1["end"]
                            between_end = entity2["start"]

                            if between_end > between_start:
                                between_text = text[between_start:between_end].strip()

                                # If there's text between, use it as predicate
                                if between_text:
                                    # Clean up the predicate
                                    predicate = between_text.strip()
                                    # Remove common stopwords
                                    for stopword in [
                                        " the ",
                                        " a ",
                                        " an ",
                                        " and ",
                                        " or ",
                                        " but ",
                                        " of ",
                                    ]:
                                        predicate = predicate.replace(stopword, " ")
                                    predicate = predicate.strip()

                                    if predicate:
                                        relations.append(
                                            (
                                                entity1["text"],
                                                predicate,
                                                entity2["text"],
                                            )
                                        )
                                else:
                                    # If no text between, use generic relation
                                    relations.append(
                                        (entity1["text"], "related_to", entity2["text"])
                                    )

            except Exception as e:
                logging.error(f"Error extracting relations: {e}")
                import traceback

                logging.error(f"Traceback: {traceback.format_exc()}")

        return relations

    def _extract_svo_from_dependency(self, sent) -> list[tuple[str, str, str]]:
        """
        Extract Subject-Verb-Object triples from a sentence using dependency parsing.

        Delegates to the shared ``_extract_svo_triples_from_sentence`` helper.

        Args:
            sent: spaCy Span representing a sentence

        Returns:
            List of (subject, predicate, object) tuples
        """
        return _extract_svo_triples_from_sentence(sent, self._get_span_text)

    def _extract_prep_relations(self, sent) -> list[tuple[str, str, str]]:
        """
        Extract relations based on prepositional phrases like "X in Y".

        Args:
            sent: spaCy Span representing a sentence

        Returns:
            List of (entity1, preposition, entity2) tuples
        """
        prep_relations = []

        for token in sent:
            if token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Get the head (the first entity)
                head_span = self._get_span_text(token.head)

                # Get the object of the preposition (the second entity)
                for child in token.children:
                    if child.dep_ == "pobj":
                        object_span = self._get_span_text(child)
                        # Create relation with the preposition as predicate
                        prep_relations.append((head_span, token.text, object_span))

        return prep_relations

    def _extract_with_semantic_roles(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract relations using semantic role labeling (SRL) if available.
        Requires AllenNLP SRL model.

        Args:
            text: Input text

        Returns:
            List of (subject, predicate, object) tuples
        """
        try:
            from allennlp.predictors.predictor import Predictor

            # Initialize SRL predictor if not already done
            if not hasattr(self, "srl_predictor"):
                self.srl_predictor = Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
                )
                logging.info("SRL model loaded successfully")

            srl_output = self.srl_predictor.predict(sentence=text)

            # Extract relations from SRL output
            relations = []
            for verb_data in srl_output.get("verbs", []):
                predicate = verb_data["verb"]
                args = {}

                # Extract arguments from tags
                for tag, words in zip(verb_data["tags"], text.split()):
                    if tag.startswith("B-ARG0"):
                        args["ARG0"] = words
                    elif tag.startswith("B-ARG1"):
                        args["ARG1"] = words
                    elif tag.startswith("B-ARG2"):
                        args["ARG2"] = words
                    elif tag.startswith("I-ARG0") and "ARG0" in args:
                        args["ARG0"] += " " + words
                    elif tag.startswith("I-ARG1") and "ARG1" in args:
                        args["ARG1"] += " " + words
                    elif tag.startswith("I-ARG2") and "ARG2" in args:
                        args["ARG2"] += " " + words

                # Create relations from arguments
                if "ARG0" in args and "ARG1" in args:
                    relations.append((args["ARG0"], predicate, args["ARG1"]))
                if "ARG0" in args and "ARG2" in args:
                    relations.append((args["ARG0"], predicate + " to", args["ARG2"]))

            return relations

        except ImportError:
            logging.warning(
                "AllenNLP SRL not available. Semantic role extraction is disabled."
            )
            return []

    def _extract_with_coreference(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract relations with coreference resolution to connect entities.

        NOTE: neuralcoref was removed (deprecated since 2020, incompatible with modern spaCy).
        This method is a stub that returns an empty list until a replacement coreference
        resolution library is integrated (e.g., coreferee, spacy-experimental coref, or fastcoref).

        Args:
            text: Input text

        Returns:
            Empty list (coreference resolution is currently disabled)
        """
        logging.debug(
            "Coreference resolution is disabled (neuralcoref was removed as deprecated)."
        )
        return []

    def _get_span_text(self, token) -> str:
        """
        Get the full text span for a token, including its children.

        Args:
            token: The token to get span text for

        Returns:
            Text of the span
        """
        # If token has no children, just return its text
        if not list(token.children):
            return token.text

        # Otherwise, get the leftmost and rightmost token in the subtree
        leftmost = token
        rightmost = token

        # Find leftmost token
        for descendant in token.subtree:
            if descendant.i < leftmost.i:
                leftmost = descendant

        # Find rightmost token
        for descendant in token.subtree:
            if descendant.i > rightmost.i:
                rightmost = descendant

        # Return the span text
        return token.doc[leftmost.i : rightmost.i + 1].text

    def add_entity(
        self,
        entity: str,
        entity_type: str = None,
        metadata: dict[str, Any] = None,
        provenance: str = None,
        confidence: float = 0.8,
        temporal_start: str = None,
        temporal_end: str = None,
        extraction_method: str = None,
        changed_by: str = None,
    ) -> int:
        """
        Add an entity to the graph with enhanced metadata and versioning.

        Args:
            entity: Entity text
            entity_type: Type of entity (e.g., PERSON, LOCATION)
            metadata: Additional entity metadata
            provenance: Source of the entity information
            confidence: Confidence score for this entity (0.0 to 1.0)
            temporal_start: Start time/date for temporal validity
            temporal_end: End time/date for temporal validity
            extraction_method: Method used to extract this entity
            changed_by: Identifier of who/what made this change

        Returns:
            ID of the created entity
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        timestamp = time.time()

        try:
            # Check if entity already exists
            cursor.execute(
                "SELECT id, version FROM graph_entities WHERE entity = ?", (entity,)
            )
            existing = cursor.fetchone()

            if existing:
                # Get the current entity data for version history
                cursor.execute(
                    """
                    SELECT entity, entity_type, metadata, provenance, confidence,
                    temporal_start, temporal_end, extraction_method
                    FROM graph_entities WHERE id = ?
                """,
                    (existing[0],),
                )
                current_data = cursor.fetchone()

                # Store the current version in version history
                new_version = existing[1] + 1
                cursor.execute(
                    """
                    INSERT INTO entity_versions
                    (entity_id, entity, entity_type, metadata, provenance, confidence,
                     temporal_start, temporal_end, extraction_method, version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        existing[0],
                        current_data[0],
                        current_data[1],
                        current_data[2],
                        current_data[3],
                        current_data[4],
                        current_data[5],
                        current_data[6],
                        current_data[7],
                        existing[1],
                        timestamp,
                        "UPDATE",
                        changed_by,
                    ),
                )

                # Update the entity with new data
                cursor.execute(
                    """
                    UPDATE graph_entities
                    SET entity_type = COALESCE(?, entity_type),
                        metadata = COALESCE(?, metadata),
                        provenance = COALESCE(?, provenance),
                        confidence = COALESCE(?, confidence),
                        temporal_start = COALESCE(?, temporal_start),
                        temporal_end = COALESCE(?, temporal_end),
                        extraction_method = COALESCE(?, extraction_method),
                        version = ?,
                        last_updated = ?
                    WHERE id = ?
                """,
                    (
                        entity_type,
                        json.dumps(metadata) if metadata else None,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        new_version,
                        timestamp,
                        existing[0],
                    ),
                )

                entity_id = existing[0]
            else:
                # Insert new entity
                cursor.execute(
                    """
                    INSERT INTO graph_entities
                    (entity, entity_type, metadata, timestamp, provenance, confidence,
                     temporal_start, temporal_end, extraction_method, version, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entity,
                        entity_type,
                        json.dumps(metadata) if metadata else None,
                        timestamp,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp,
                    ),
                )

                entity_id = cursor.lastrowid

                # Add to version history
                cursor.execute(
                    """
                    INSERT INTO entity_versions
                    (entity_id, entity, entity_type, metadata, provenance, confidence,
                     temporal_start, temporal_end, extraction_method, version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entity_id,
                        entity,
                        entity_type,
                        json.dumps(metadata) if metadata else None,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp,
                        "CREATE",
                        changed_by,
                    ),
                )

            conn.commit()

            # Add to NetworkX graph if enabled
            if NETWORKX_ENABLED and self.graph is not None:
                node_attrs = {
                    "name": entity,
                    "entity_type": entity_type,
                    "provenance": provenance,
                    "confidence": confidence,
                    "temporal_start": temporal_start,
                    "temporal_end": temporal_end,
                    "extraction_method": extraction_method,
                    "version": 1 if existing is None else existing[1] + 1,
                    "timestamp": timestamp,
                }

                if metadata:
                    node_attrs.update(metadata)

                self.graph.add_node(entity_id, **node_attrs)

            return entity_id

        except Exception as e:
            logging.error(f"Error adding entity: {e}")
            conn.rollback()
            return -1

        finally:
            if self.conn is None and conn:
                conn.close()

    def add_entity_alias(
        self, entity_id: int, alias: str, confidence: float = 0.8
    ) -> bool:
        """
        Add an alias to an existing entity for entity linking.

        Args:
            entity_id: ID of the existing entity
            alias: Alternative name or reference to the entity
            confidence: Confidence score for this alias (0.0-1.0)

        Returns:
            True if successful, False otherwise
        """
        # Verify entity exists
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Get entity information
            cursor.execute(
                "SELECT entity, metadata FROM graph_entities WHERE id = ?", (entity_id,)
            )
            result = cursor.fetchone()

            if not result:
                logging.error(f"Entity with ID {entity_id} not found")
                return False

            canonical_name = result["entity"]
            metadata_str = result["metadata"]

            # Update metadata to include the new alias
            if metadata_str:
                metadata = json.loads(metadata_str)
            else:
                metadata = {}

            if "aliases" not in metadata:
                metadata["aliases"] = []

            # Add alias if not already present
            if alias not in metadata["aliases"]:
                metadata["aliases"].append(alias)
                metadata["alias_confidence"] = metadata.get("alias_confidence", {})
                metadata["alias_confidence"][alias] = confidence

                # Update database
                cursor.execute(
                    "UPDATE graph_entities SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), entity_id),
                )

                # Update entity linking database
                self.entity_db[canonical_name] = {
                    "id": entity_id,
                    "metadata": metadata,
                    "aliases": metadata["aliases"],
                }

                self.entity_db[alias] = {"canonical": canonical_name, "id": entity_id}

                conn.commit()
                return True
            else:
                # Alias already exists
                return True

        except Exception as e:
            logging.error(f"Error adding entity alias: {e}")
            conn.rollback()
            return False
        finally:
            if self.conn is None:
                conn.close()

    def add_relation(
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
        changed_by: str = None,
    ) -> bool:
        """
        Add a relation between two entities with enhanced metadata and versioning.

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
            changed_by: Identifier of who/what made this change

        Returns:
            True if relation was added successfully
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        timestamp = time.time()

        try:
            # Verify relation type exists
            cursor.execute(
                "SELECT id, symmetric, transitive, inverse_relation FROM relation_types WHERE name = ?",
                (relation_type,),
            )
            relation_type_data = cursor.fetchone()

            if not relation_type_data:
                # Add the relation type dynamically
                logging.info(f"Adding new relation type: {relation_type}")
                cursor.execute(
                    """
                    INSERT INTO relation_types (name, parent_type, description, metadata)
                    VALUES (?, 'related_to', ?, ?)
                """,
                    (
                        relation_type,
                        f"Dynamically added relation type: {relation_type}",
                        json.dumps({"automatic": True, "added_at": timestamp}),
                    ),
                )

                # Get symmetry and transitivity for new relation type
                is_symmetric = False
                is_transitive = False
                inverse_relation = None
            else:
                is_symmetric = bool(relation_type_data[1])
                is_transitive = bool(relation_type_data[2])
                inverse_relation = relation_type_data[3]

            # Get source entity ID
            cursor.execute(
                "SELECT id FROM graph_entities WHERE entity = ?", (source_entity,)
            )
            source = cursor.fetchone()

            # Get target entity ID
            cursor.execute(
                "SELECT id FROM graph_entities WHERE entity = ?", (target_entity,)
            )
            target = cursor.fetchone()

            # If either entity doesn't exist, create them
            if not source:
                source_id = self.add_entity(
                    source_entity,
                    provenance=provenance,
                    confidence=confidence,
                    extraction_method=extraction_method,
                    changed_by=changed_by,
                )
            else:
                source_id = source[0]

            if not target:
                target_id = self.add_entity(
                    target_entity,
                    provenance=provenance,
                    confidence=confidence,
                    extraction_method=extraction_method,
                    changed_by=changed_by,
                )
            else:
                target_id = target[0]

            # Check if relation already exists
            cursor.execute(
                """
                SELECT id, version FROM graph_relationships
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
            """,
                (source_id, target_id, relation_type),
            )
            existing = cursor.fetchone()

            if existing:
                # Get the current relation data for version history
                cursor.execute(
                    """
                    SELECT source_id, target_id, relation_type, weight, metadata,
                    provenance, confidence, temporal_start, temporal_end, extraction_method
                    FROM graph_relationships WHERE id = ?
                """,
                    (existing[0],),
                )
                current_data = cursor.fetchone()

                # Store the current version in version history
                new_version = existing[1] + 1
                cursor.execute(
                    """
                    INSERT INTO relationship_versions
                    (relationship_id, source_id, target_id, relation_type, weight, metadata,
                     provenance, confidence, temporal_start, temporal_end, extraction_method,
                     version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        existing[0],
                        current_data[0],
                        current_data[1],
                        current_data[2],
                        current_data[3],
                        current_data[4],
                        current_data[5],
                        current_data[6],
                        current_data[7],
                        current_data[8],
                        current_data[9],
                        existing[1],
                        timestamp,
                        "UPDATE",
                        changed_by,
                    ),
                )

                # Update existing relation
                cursor.execute(
                    """
                    UPDATE graph_relationships
                    SET weight = COALESCE(?, weight),
                        metadata = COALESCE(?, metadata),
                        provenance = COALESCE(?, provenance),
                        confidence = COALESCE(?, confidence),
                        temporal_start = COALESCE(?, temporal_start),
                        temporal_end = COALESCE(?, temporal_end),
                        extraction_method = COALESCE(?, extraction_method),
                        version = ?,
                        last_updated = ?
                    WHERE id = ?
                """,
                    (
                        weight,
                        json.dumps(metadata) if metadata else None,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        new_version,
                        timestamp,
                        existing[0],
                    ),
                )

                relation_id = existing[0]
            else:
                # Insert new relation
                cursor.execute(
                    """
                    INSERT INTO graph_relationships
                    (source_id, target_id, relation_type, weight, metadata, timestamp,
                     provenance, confidence, temporal_start, temporal_end, extraction_method,
                     version, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        source_id,
                        target_id,
                        relation_type,
                        weight,
                        json.dumps(metadata) if metadata else None,
                        timestamp,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp,
                    ),
                )

                relation_id = cursor.lastrowid

                # Add to version history
                cursor.execute(
                    """
                    INSERT INTO relationship_versions
                    (relationship_id, source_id, target_id, relation_type, weight, metadata,
                     provenance, confidence, temporal_start, temporal_end, extraction_method,
                     version, timestamp, change_type, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        relation_id,
                        source_id,
                        target_id,
                        relation_type,
                        weight,
                        json.dumps(metadata) if metadata else None,
                        provenance,
                        confidence,
                        temporal_start,
                        temporal_end,
                        extraction_method,
                        1,  # Initial version
                        timestamp,
                        "CREATE",
                        changed_by,
                    ),
                )

                # If relation is symmetric, add the reverse relation
                if is_symmetric and source_id != target_id:
                    cursor.execute(
                        """
                        INSERT INTO graph_relationships
                        (source_id, target_id, relation_type, weight, metadata, timestamp,
                         provenance, confidence, temporal_start, temporal_end, extraction_method,
                         version, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            target_id,
                            source_id,
                            relation_type,
                            weight,
                            json.dumps(
                                {**(metadata or {}), "symmetric_of": relation_id}
                            ),
                            timestamp,
                            provenance,
                            confidence,
                            temporal_start,
                            temporal_end,
                            extraction_method,
                            1,  # Initial version
                            timestamp,
                        ),
                    )

                    symmetric_id = cursor.lastrowid

                    # Add symmetric relation to version history
                    cursor.execute(
                        """
                        INSERT INTO relationship_versions
                        (relationship_id, source_id, target_id, relation_type, weight, metadata,
                         provenance, confidence, temporal_start, temporal_end, extraction_method,
                         version, timestamp, change_type, changed_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symmetric_id,
                            target_id,
                            source_id,
                            relation_type,
                            weight,
                            json.dumps(
                                {**(metadata or {}), "symmetric_of": relation_id}
                            ),
                            provenance,
                            confidence,
                            temporal_start,
                            temporal_end,
                            extraction_method,
                            1,  # Initial version
                            timestamp,
                            "CREATE_SYMMETRIC",
                            changed_by,
                        ),
                    )

                # If inverse relation is defined, add the inverse relation
                if inverse_relation and source_id != target_id:
                    cursor.execute(
                        """
                        INSERT INTO graph_relationships
                        (source_id, target_id, relation_type, weight, metadata, timestamp,
                         provenance, confidence, temporal_start, temporal_end, extraction_method,
                         version, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            target_id,
                            source_id,
                            inverse_relation,
                            weight,
                            json.dumps({**(metadata or {}), "inverse_of": relation_id}),
                            timestamp,
                            provenance,
                            confidence,
                            temporal_start,
                            temporal_end,
                            extraction_method,
                            1,  # Initial version
                            timestamp,
                        ),
                    )

                    inverse_id = cursor.lastrowid

                    # Add inverse relation to version history
                    cursor.execute(
                        """
                        INSERT INTO relationship_versions
                        (relationship_id, source_id, target_id, relation_type, weight, metadata,
                         provenance, confidence, temporal_start, temporal_end, extraction_method,
                         version, timestamp, change_type, changed_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            inverse_id,
                            target_id,
                            source_id,
                            inverse_relation,
                            weight,
                            json.dumps({**(metadata or {}), "inverse_of": relation_id}),
                            provenance,
                            confidence,
                            temporal_start,
                            temporal_end,
                            extraction_method,
                            1,  # Initial version
                            timestamp,
                            "CREATE_INVERSE",
                            changed_by,
                        ),
                    )

                # If relation is transitive, check for implied transitive relations
                if is_transitive:
                    self._add_transitive_relations(
                        source_id,
                        target_id,
                        relation_type,
                        weight,
                        provenance,
                        confidence,
                        extraction_method,
                        changed_by,
                    )

            conn.commit()

            # Add to NetworkX graph if enabled
            if NETWORKX_ENABLED and self.graph is not None:
                edge_attrs = {
                    "relation": relation_type,
                    "weight": weight,
                    "provenance": provenance,
                    "confidence": confidence,
                    "temporal_start": temporal_start,
                    "temporal_end": temporal_end,
                    "extraction_method": extraction_method,
                    "timestamp": timestamp,
                    "version": 1 if existing is None else existing[1] + 1,
                }

                if metadata:
                    edge_attrs.update(metadata)

                self.graph.add_edge(source_id, target_id, **edge_attrs)

                # Add symmetric and inverse edges to NetworkX if applicable
                if is_symmetric and source_id != target_id:
                    self.graph.add_edge(
                        target_id,
                        source_id,
                        **{**edge_attrs, "symmetric_of": relation_id},
                    )

                if inverse_relation and source_id != target_id:
                    self.graph.add_edge(
                        target_id,
                        source_id,
                        **{
                            **edge_attrs,
                            "relation": inverse_relation,
                            "inverse_of": relation_id,
                        },
                    )

            return True

        except Exception as e:
            logging.error(f"Error adding relation: {e}")
            import traceback

            logging.error(f"Traceback: {traceback.format_exc()}")
            conn.rollback()
            return False

        finally:
            if self.conn is None:
                conn.close()

    def _add_transitive_relations(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        weight: float,
        provenance: str,
        confidence: float,
        extraction_method: str,
        changed_by: str,
    ) -> None:
        """
        Add implied transitive relations.

        For example, if A related_to B and B related_to C, then A related_to C
        for transitive relations like "is_a" or "part_of".

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relation
            weight: Weight of the relation
            provenance: Source of the relation information
            confidence: Confidence score for this relation
            extraction_method: Method used to extract this relation
            changed_by: Identifier of who/what made this change
        """
        if self.conn is None:
            return

        cursor = self.conn.cursor()
        timestamp = time.time()

        try:
            # Find all entities that the source is related to with this relation
            cursor.execute(
                """
                SELECT target_id FROM graph_relationships
                WHERE source_id = ? AND relation_type = ?
            """,
                (source_id, relation_type),
            )
            sources_targets = cursor.fetchall()

            # Find all entities that have this relation to the target
            cursor.execute(
                """
                SELECT source_id FROM graph_relationships
                WHERE target_id = ? AND relation_type = ?
            """,
                (target_id, relation_type),
            )
            targets_sources = cursor.fetchall()

            # Add transitive relations: source -> target's sources
            for row in targets_sources:
                third_id = row[0]
                if third_id != source_id:  # Avoid self-relations
                    # Check if this relation already exists
                    cursor.execute(
                        """
                        SELECT id FROM graph_relationships
                        WHERE source_id = ? AND target_id = ? AND relation_type = ?
                    """,
                        (source_id, third_id, relation_type),
                    )
                    if not cursor.fetchone():
                        # Add the transitive relation with reduced confidence
                        new_confidence = (
                            confidence * 0.9
                        )  # Reduce confidence for transitive inference

                        # Add the relation
                        transitive_metadata = {
                            "transitive": True,
                            "via_entity": target_id,
                            "inference_type": "transitive_relation",
                        }

                        cursor.execute(
                            """
                            INSERT INTO graph_relationships
                            (source_id, target_id, relation_type, weight, metadata, timestamp,
                             provenance, confidence, extraction_method, version, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                source_id,
                                third_id,
                                relation_type,
                                weight * 0.8,  # Reduce weight for transitive relations
                                json.dumps(transitive_metadata),
                                timestamp,
                                provenance,
                                new_confidence,
                                "transitive_inference",
                                1,
                                timestamp,
                            ),
                        )

                        transitive_id = cursor.lastrowid

                        # Add to version history
                        cursor.execute(
                            """
                            INSERT INTO relationship_versions
                            (relationship_id, source_id, target_id, relation_type, weight, metadata,
                             provenance, confidence, extraction_method, version, timestamp, change_type, changed_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                transitive_id,
                                source_id,
                                third_id,
                                relation_type,
                                weight * 0.8,
                                json.dumps(transitive_metadata),
                                provenance,
                                new_confidence,
                                "transitive_inference",
                                1,
                                timestamp,
                                "CREATE_TRANSITIVE",
                                changed_by,
                            ),
                        )

            # Add transitive relations: source's sources -> target
            for row in sources_targets:
                third_id = row[0]
                if third_id != target_id:  # Avoid self-relations
                    # Check if this relation already exists
                    cursor.execute(
                        """
                        SELECT id FROM graph_relationships
                        WHERE source_id = ? AND target_id = ? AND relation_type = ?
                    """,
                        (third_id, target_id, relation_type),
                    )
                    if not cursor.fetchone():
                        # Add the transitive relation with reduced confidence
                        new_confidence = (
                            confidence * 0.9
                        )  # Reduce confidence for transitive inference

                        # Add the relation
                        transitive_metadata = {
                            "transitive": True,
                            "via_entity": source_id,
                            "inference_type": "transitive_relation",
                        }

                        cursor.execute(
                            """
                            INSERT INTO graph_relationships
                            (source_id, target_id, relation_type, weight, metadata, timestamp,
                             provenance, confidence, extraction_method, version, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                third_id,
                                target_id,
                                relation_type,
                                weight * 0.8,  # Reduce weight for transitive relations
                                json.dumps(transitive_metadata),
                                timestamp,
                                provenance,
                                new_confidence,
                                "transitive_inference",
                                1,
                                timestamp,
                            ),
                        )

                        transitive_id = cursor.lastrowid

                        # Add to version history
                        cursor.execute(
                            """
                            INSERT INTO relationship_versions
                            (relationship_id, source_id, target_id, relation_type, weight, metadata,
                             provenance, confidence, extraction_method, version, timestamp, change_type, changed_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                transitive_id,
                                third_id,
                                target_id,
                                relation_type,
                                weight * 0.8,
                                json.dumps(transitive_metadata),
                                provenance,
                                new_confidence,
                                "transitive_inference",
                                1,
                                timestamp,
                                "CREATE_TRANSITIVE",
                                changed_by,
                            ),
                        )

        except Exception as e:
            logging.error(f"Error adding transitive relations: {e}")
            import traceback

            logging.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise - this is a best-effort operation

    def add_nary_relation(
        self,
        relation_type: str,
        participants: dict[str, str],
        metadata: dict[str, Any] = None,
        provenance: str = None,
        confidence: float = 0.5,
    ) -> int:
        """
        Add an n-ary relation involving multiple entities with different roles.

        Args:
            relation_type: Type of the n-ary relation
            participants: Dictionary mapping role names to entity names
            metadata: Additional relation metadata
            provenance: Source of the relation information
            confidence: Confidence score for this relation (0.0 to 1.0)

        Returns:
            ID of the created n-ary relation or -1 if failed
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        cursor = conn.cursor()
        timestamp = time.time()

        try:
            # If ontology is enabled, check the relation type in the ontology
            if ONTOLOGY_ENABLED and self.ontology:
                if not self.ontology.get_relation_type(relation_type):
                    # Add a basic relation type if needed
                    from cortexflow.ontology import RelationType

                    self.ontology.add_relation_type(
                        RelationType(
                            name=relation_type,
                            parent_types=["related_to"],
                            metadata={
                                "n_ary": True,
                                "automatic": True,
                                "confidence": 0.7,
                            },
                        )
                    )
                    logging.info(
                        f"Added n-ary relation type to ontology: {relation_type}"
                    )

            # Insert the n-ary relation
            cursor.execute(
                """
                INSERT INTO nary_relationships
                (relation_type, metadata, provenance, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    relation_type,
                    json.dumps(metadata) if metadata else None,
                    provenance,
                    confidence,
                    timestamp,
                ),
            )

            relation_id = cursor.lastrowid

            # Add all participants
            for role, entity_name in participants.items():
                # Get or create the entity
                cursor.execute(
                    "SELECT id FROM graph_entities WHERE entity = ?", (entity_name,)
                )
                entity_row = cursor.fetchone()

                if not entity_row:
                    entity_id = self.add_entity(
                        entity=entity_name, provenance=provenance, confidence=confidence
                    )
                else:
                    entity_id = entity_row[0]

                # Add the participant
                cursor.execute(
                    """
                    INSERT INTO nary_participants
                    (relationship_id, entity_id, role, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (relation_id, entity_id, role, timestamp),
                )

            conn.commit()

            # Add to NetworkX graph if enabled (using a hypergraph-like representation)
            if NETWORKX_ENABLED and self.graph is not None:
                # Create a special node for the n-ary relation
                relation_node_id = f"nary_{relation_id}"
                self.graph.add_node(
                    relation_node_id,
                    relation_type=relation_type,
                    n_ary=True,
                    provenance=provenance,
                    confidence=confidence,
                    timestamp=timestamp,
                    **(metadata or {}),
                )

                # Connect all participants to the relation node
                for role, entity_name in participants.items():
                    cursor.execute(
                        "SELECT id FROM graph_entities WHERE entity = ?", (entity_name,)
                    )
                    entity_row = cursor.fetchone()
                    if entity_row:
                        entity_id = entity_row[0]
                        self.graph.add_edge(
                            entity_id,
                            relation_node_id,
                            role=role,
                            weight=1.0,
                            timestamp=timestamp,
                        )

            return relation_id

        except Exception as e:
            logging.error(f"Error adding n-ary relation: {e}")
            conn.rollback()
            return -1

        finally:
            if self.conn is None:
                conn.close()

    def get_nary_relation(self, relation_id: int) -> dict[str, Any] | None:
        """
        Get details of an n-ary relation by ID.

        Args:
            relation_id: ID of the n-ary relation

        Returns:
            Dictionary with relation details or None if not found
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Get the relation details
            cursor.execute(
                """
                SELECT relation_type, metadata, provenance, confidence, timestamp
                FROM nary_relationships
                WHERE id = ?
            """,
                (relation_id,),
            )

            relation_row = cursor.fetchone()

            if not relation_row:
                return None

            # Convert row to dictionary
            relation = dict(relation_row)

            # Parse metadata JSON
            if relation["metadata"]:
                relation["metadata"] = json.loads(relation["metadata"])
            else:
                relation["metadata"] = {}

            # Get all participants
            cursor.execute(
                """
                SELECT np.role, ge.entity, ge.entity_type, ge.id as entity_id
                FROM nary_participants np
                JOIN graph_entities ge ON np.entity_id = ge.id
                WHERE np.relationship_id = ?
            """,
                (relation_id,),
            )

            participants = {}
            for row in cursor.fetchall():
                participants[row["role"]] = {
                    "entity": row["entity"],
                    "entity_type": row["entity_type"],
                    "entity_id": row["entity_id"],
                }

            relation["participants"] = participants
            relation["id"] = relation_id

            return relation

        except Exception as e:
            logging.error(f"Error getting n-ary relation: {e}")
            return None

        finally:
            if self.conn is None:
                conn.close()

    def query_nary_relations(
        self,
        relation_type: str = None,
        participant_entity: str = None,
        participant_role: str = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query n-ary relations with optional filters.

        Args:
            relation_type: Type of relation to filter by (optional)
            participant_entity: Filter by entity name participating in the relation (optional)
            participant_role: Filter by role in the relation (optional)
            limit: Maximum number of results

        Returns:
            List of n-ary relation dictionaries
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            params = []

            # Start with base query
            query = """
                SELECT DISTINCT nr.id as relation_id
                FROM nary_relationships nr
            """

            # Add join conditions if needed
            if participant_entity or participant_role:
                query += """
                    JOIN nary_participants np ON nr.id = np.relationship_id
                    JOIN graph_entities ge ON np.entity_id = ge.id
                """

            # Add WHERE clause
            where_clauses = []

            if relation_type:
                where_clauses.append("nr.relation_type = ?")
                params.append(relation_type)

            if participant_entity:
                where_clauses.append("ge.entity = ?")
                params.append(participant_entity)

            if participant_role:
                where_clauses.append("np.role = ?")
                params.append(participant_role)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            # Add order and limit
            query += " ORDER BY nr.timestamp DESC LIMIT ?"
            params.append(limit)

            # Execute query to get relation IDs
            cursor.execute(query, params)
            relation_ids = [row["relation_id"] for row in cursor.fetchall()]

            # Get full details for each relation
            results = []
            for relation_id in relation_ids:
                relation = self.get_nary_relation(relation_id)
                if relation:
                    results.append(relation)

            return results

        except Exception as e:
            logging.error(f"Error querying n-ary relations: {e}")
            return []

        finally:
            if self.conn is None:
                conn.close()

    def get_entity_metadata(self, entity_id: int) -> dict[str, Any]:
        """
        Get full metadata for an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            Dictionary with entity metadata
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT entity, entity_type, metadata, provenance, confidence,
                       temporal_start, temporal_end, timestamp
                FROM graph_entities
                WHERE id = ?
            """,
                (entity_id,),
            )

            row = cursor.fetchone()

            if not row:
                return {}

            result = dict(row)

            # Parse metadata JSON
            if result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])
            else:
                result["metadata"] = {}

            result["id"] = entity_id

            return result

        except Exception as e:
            logging.error(f"Error getting entity metadata: {e}")
            return {}

        finally:
            if self.conn is None:
                conn.close()

    def get_relation_metadata(self, relation_id: int) -> dict[str, Any]:
        """
        Get full metadata for a relation.

        Args:
            relation_id: ID of the relation

        Returns:
            Dictionary with relation metadata
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT source_id, target_id, relation_type, weight, metadata,
                       provenance, confidence, temporal_start, temporal_end, timestamp
                FROM graph_relationships
                WHERE id = ?
            """,
                (relation_id,),
            )

            row = cursor.fetchone()

            if not row:
                return {}

            result = dict(row)

            # Parse metadata JSON
            if result["metadata"]:
                result["metadata"] = json.loads(result["metadata"])
            else:
                result["metadata"] = {}

            # Get source and target entity details
            source = self.get_entity_metadata(result["source_id"])
            target = self.get_entity_metadata(result["target_id"])

            result["source"] = source
            result["target"] = target
            result["id"] = relation_id

            return result

        except Exception as e:
            logging.error(f"Error getting relation metadata: {e}")
            return {}

        finally:
            if self.conn is None:
                conn.close()

    def process_text_to_graph(self, text: str, source: str = None) -> int:
        """
        Process text to extract entities and relations and add them to the graph.
        Enhanced to support n-ary relationships and metadata tracking.

        Args:
            text: The text to process
            source: The source of the text (for provenance)

        Returns:
            Number of relations added to the graph
        """
        relations_added = 0

        try:
            # Extract entities
            self.extract_entities(text)

            # Extract binary relations using existing methods
            binary_relations = self.extract_relations(text)

            # Add binary relations to graph
            for subj, pred, obj in binary_relations:
                success = self.add_relation(
                    source_entity=subj,
                    relation_type=pred,
                    target_entity=obj,
                    provenance=source,
                    confidence=0.7,  # Default confidence for extracted relations
                )

                if success:
                    relations_added += 1

            # Try to extract n-ary relations with roles
            nary_relations = self._extract_complex_events(text)

            # Add n-ary relations to graph
            for relation in nary_relations:
                relation_type = relation.get("type", "event")
                participants = relation.get("participants", {})

                if participants:
                    nary_id = self.add_nary_relation(
                        relation_type=relation_type,
                        participants=participants,
                        provenance=source,
                        confidence=0.6,  # Default confidence for n-ary relations
                    )

                    if nary_id > 0:
                        relations_added += 1

            return relations_added

        except Exception as e:
            logging.error(f"Error processing text to graph: {e}")
            return 0

    def _extract_complex_events(self, text: str) -> list[dict[str, Any]]:
        """
        Extract complex events and scenarios as n-ary relations with roles.

        Args:
            text: Text to process

        Returns:
            List of n-ary relation dictionaries
        """
        complex_events = []

        # Basic implementation using existing NLP capabilities
        if not SPACY_ENABLED or not self.nlp:
            return complex_events

        try:
            doc = self.nlp(text)

            # Look for sentences with multiple entities or complex structure
            for sent in doc.sents:
                # Check if this is a complex sentence (multiple verbs or entities)
                entities = [e for e in sent.ents]
                verbs = [token for token in sent if token.pos_ == "VERB"]

                # Skip simple sentences
                if len(entities) < 2 or len(verbs) < 1:
                    continue

                # For each main verb, try to identify an event
                for verb in verbs:
                    # Skip auxiliary verbs
                    if verb.dep_ in ("aux", "auxpass"):
                        continue

                    # Get the verb lemma as event type
                    event_type = verb.lemma_

                    # Collect participants by role
                    participants = {}

                    # Check for subject
                    subjects = [
                        token
                        for token in verb.children
                        if token.dep_ in ("nsubj", "nsubjpass")
                    ]
                    for subject in subjects:
                        # Extend to noun phrases
                        subj_span = self._get_span_text(subject)
                        participants["agent"] = subj_span

                    # Check for object
                    objects = [
                        token
                        for token in verb.children
                        if token.dep_ in ("dobj", "pobj")
                    ]
                    for obj in objects:
                        # Extend to noun phrases
                        obj_span = self._get_span_text(obj)
                        participants["theme"] = obj_span

                    # Check for indirect object
                    ind_objects = [
                        token for token in verb.children if token.dep_ == "iobj"
                    ]
                    for ind_obj in ind_objects:
                        # Extend to noun phrases
                        ind_obj_span = self._get_span_text(ind_obj)
                        participants["recipient"] = ind_obj_span

                    # Check for time expressions
                    time_preps = [
                        token
                        for token in verb.children
                        if token.dep_ == "prep" and token.text in ("at", "on", "in")
                    ]
                    for prep in time_preps:
                        for child in prep.children:
                            if child.dep_ == "pobj":
                                time_span = self._get_span_text(child)
                                participants["time"] = time_span

                    # Check for location
                    loc_preps = [
                        token
                        for token in verb.children
                        if token.dep_ == "prep"
                        and token.text in ("at", "in", "on", "near", "by")
                    ]
                    for prep in loc_preps:
                        for child in prep.children:
                            if child.dep_ == "pobj":
                                # Check if this is actually a location
                                if child.ent_type_ in ("LOC", "GPE"):
                                    loc_span = self._get_span_text(child)
                                    participants["location"] = loc_span

                    # If we have at least two participants, add the event
                    if len(participants) >= 2:
                        complex_events.append(
                            {
                                "type": event_type,
                                "participants": participants,
                                "sentence": sent.text,
                            }
                        )

        except Exception as e:
            logging.error(f"Error extracting complex events: {e}")

        return complex_events

    def get_entity_neighbors(
        self,
        entity: str,
        direction: str = "both",
        relation_type: str = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get neighbors of an entity in the graph.

        Args:
            entity: Entity name to find neighbors for
            direction: Direction of relations: "outgoing", "incoming", or "both"
            relation_type: Filter by specific relation type (optional)
            limit: Maximum number of neighbors to return

        Returns:
            List of neighbor dictionaries with relation details
        """
        if self.conn is not None:
            conn = self.conn
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        neighbors = []

        try:
            # Get entity ID
            cursor.execute("SELECT id FROM graph_entities WHERE entity = ?", (entity,))
            entity_row = cursor.fetchone()

            if not entity_row:
                return []

            entity_id = entity_row["id"]

            # Get outgoing relations
            if direction in ["outgoing", "both"]:
                query = """
                    SELECT r.id, r.relation_type, r.weight, r.provenance, r.confidence,
                           e.id as target_id, e.entity, e.entity_type
                    FROM graph_relationships r
                    JOIN graph_entities e ON r.target_id = e.id
                    WHERE r.source_id = ?
                """

                params = [entity_id]

                if relation_type:
                    query += " AND r.relation_type = ?"
                    params.append(relation_type)

                query += " ORDER BY r.confidence DESC, r.weight DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)

                for row in cursor.fetchall():
                    neighbor = dict(row)
                    neighbor["direction"] = "outgoing"
                    neighbor["relation"] = neighbor["relation_type"]
                    neighbors.append(neighbor)

            # Get incoming relations
            if direction in ["incoming", "both"]:
                remaining_limit = limit - len(neighbors)

                if remaining_limit > 0:
                    query = """
                        SELECT r.id, r.relation_type, r.weight, r.provenance, r.confidence,
                               e.id as source_id, e.entity, e.entity_type
                        FROM graph_relationships r
                        JOIN graph_entities e ON r.source_id = e.id
                        WHERE r.target_id = ?
                    """

                    params = [entity_id]

                    if relation_type:
                        query += " AND r.relation_type = ?"
                        params.append(relation_type)

                    query += " ORDER BY r.confidence DESC, r.weight DESC LIMIT ?"
                    params.append(remaining_limit)

                    cursor.execute(query, params)

                    for row in cursor.fetchall():
                        neighbor = dict(row)
                        neighbor["direction"] = "incoming"
                        neighbor["relation"] = neighbor["relation_type"]
                        neighbors.append(neighbor)

            # Sort by weight and confidence
            neighbors = sorted(
                neighbors,
                key=lambda x: (x.get("confidence", 0.0), x.get("weight", 0.0)),
                reverse=True,
            )

            return neighbors[:limit]

        except Exception as e:
            logging.error(f"Error getting entity neighbors: {e}")
            return []

        finally:
            if self.conn is None:
                conn.close()

    def build_knowledge_subgraph(
        self, query: str, max_nodes: int = 20
    ) -> dict[str, Any]:
        """
        Build a knowledge subgraph relevant to a query.

        Args:
            query: Query text to extract entities from
            max_nodes: Maximum number of nodes to include in the subgraph

        Returns:
            Dictionary with nodes and edges of the subgraph
        """
        subgraph = {"nodes": [], "edges": []}

        try:
            # Extract potential entities from the query
            query_entities = []

            # Use NLP to extract entities if available
            if SPACY_ENABLED and self.nlp:
                doc = self.nlp(query)
                for ent in doc.ents:
                    query_entities.append(ent.text)

                # Also add noun chunks as potential entities
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2 and chunk.text not in query_entities:
                        query_entities.append(chunk.text)

            # If no entities found with NLP, use simple word extraction
            if not query_entities:
                for word in query.split():
                    if len(word) > 3 and (word[0].isupper() or word in query.lower()):
                        query_entities.append(word)

            # Check for common entity patterns in queries
            connection_pattern = r"(?:connection|relationship|relation)(?:\s+between\s+)([^,]+?)(?:\s+and\s+)([^?\.]+)"
            match = re.search(connection_pattern, query, re.IGNORECASE)
            if match:
                start_entity = match.group(1).strip()
                end_entity = match.group(2).strip()
                if start_entity not in query_entities:
                    query_entities.append(start_entity)
                if end_entity not in query_entities:
                    query_entities.append(end_entity)

            # Track nodes and edges to avoid duplicates
            node_ids = set()
            edge_ids = set()

            # For each entity in the query, get neighbors
            for entity_text in query_entities:
                neighbors = self.get_entity_neighbors(
                    entity=entity_text, direction="both", limit=5
                )

                if not neighbors:
                    continue

                # Safely get source entity ID
                source_id = None
                for neighbor in neighbors:
                    if "source_id" in neighbor and neighbor["direction"] == "outgoing":
                        source_id = neighbor["source_id"]
                        break
                    elif (
                        "target_id" in neighbor and neighbor["direction"] == "incoming"
                    ):
                        source_id = neighbor["target_id"]
                        break

                # If we can't find source ID, try a direct lookup
                if source_id is None:
                    entity_info = self.get_entity_id(entity_text)
                    if entity_info:
                        source_id = entity_info.get("id")

                # Add the entity to nodes if we have an ID and it's not already added
                if source_id and source_id not in node_ids:
                    try:
                        # Get entity details
                        entity_details = self.get_entity_metadata(source_id) or {}

                        # Create node
                        node = {
                            "id": source_id,
                            "label": entity_text,
                            "type": entity_details.get("entity_type", "unknown"),
                            "confidence": entity_details.get("confidence", 0.5),
                        }

                        subgraph["nodes"].append(node)
                        node_ids.add(source_id)
                    except Exception as e:
                        logging.error(f"Error adding source node {source_id}: {e}")

                # Process each neighbor
                for neighbor in neighbors:
                    try:
                        # Skip neighbors without required fields
                        if "entity" not in neighbor:
                            logging.warning(
                                f"Skipping neighbor without entity field: {neighbor}"
                            )
                            continue

                        neighbor_entity = neighbor["entity"]
                        relation = neighbor.get("relation", "related_to")

                        # Safely get neighbor ID
                        neighbor_id = None
                        if (
                            neighbor["direction"] == "outgoing"
                            and "target_id" in neighbor
                        ):
                            neighbor_id = neighbor["target_id"]
                        elif (
                            neighbor["direction"] == "incoming"
                            and "source_id" in neighbor
                        ):
                            neighbor_id = neighbor["source_id"]

                        # If we can't get neighbor ID, try direct lookup
                        if neighbor_id is None:
                            nb_info = self.get_entity_id(neighbor_entity)
                            if nb_info:
                                neighbor_id = nb_info.get("id")

                        # Skip if we still don't have required IDs
                        if not source_id or not neighbor_id:
                            logging.warning(
                                f"Missing ID for {entity_text} or {neighbor_entity}"
                            )
                            continue

                        # Add neighbor node if new
                        if neighbor_id not in node_ids:
                            try:
                                # Get entity details
                                entity_details = (
                                    self.get_entity_metadata(neighbor_id) or {}
                                )

                                # Create node
                                node = {
                                    "id": neighbor_id,
                                    "label": neighbor_entity,
                                    "type": entity_details.get(
                                        "entity_type", "unknown"
                                    ),
                                    "confidence": entity_details.get("confidence", 0.5),
                                }

                                subgraph["nodes"].append(node)
                                node_ids.add(neighbor_id)
                            except Exception as e:
                                logging.error(
                                    f"Error adding neighbor node {neighbor_id}: {e}"
                                )
                                continue

                        # Add edge
                        edge_id = (
                            f"{source_id}_{neighbor_id}_{relation}"
                            if neighbor["direction"] == "outgoing"
                            else f"{neighbor_id}_{source_id}_{relation}"
                        )

                        if edge_id not in edge_ids:
                            edge = {
                                "source": source_id
                                if neighbor["direction"] == "outgoing"
                                else neighbor_id,
                                "target": neighbor_id
                                if neighbor["direction"] == "outgoing"
                                else source_id,
                                "label": relation,
                                "weight": neighbor.get("weight", 1.0),
                                "confidence": neighbor.get("confidence", 0.5),
                            }

                            subgraph["edges"].append(edge)
                            edge_ids.add(edge_id)

                    except Exception as e:
                        logging.error(f"Error processing neighbor: {e}")

                # Check if we reached max nodes
                if len(node_ids) >= max_nodes:
                    break

            return subgraph

        except Exception as e:
            logging.error(f"Error building knowledge subgraph: {e}")
            return {"nodes": [], "edges": []}

    def get_entity_id(self, entity_text: str) -> dict[str, Any] | None:
        """
        Get entity ID from text.

        Args:
            entity_text: Entity text to look up

        Returns:
            Dictionary with entity information or None if not found
        """
        try:
            if self.conn is not None:
                conn = self.conn
            else:
                conn = sqlite3.connect(self.db_path)

            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Try exact match first
            cursor.execute(
                "SELECT id, entity, entity_type FROM graph_entities WHERE entity = ?",
                (entity_text,),
            )
            entity_row = cursor.fetchone()

            if not entity_row:
                # Try case-insensitive matching
                cursor.execute(
                    "SELECT id, entity, entity_type FROM graph_entities WHERE LOWER(entity) = LOWER(?)",
                    (entity_text,),
                )
                entity_row = cursor.fetchone()

            if not entity_row:
                # Try fuzzy matching as last resort
                cursor.execute(
                    "SELECT id, entity, entity_type FROM graph_entities WHERE entity LIKE ? LIMIT 1",
                    (f"%{entity_text}%",),
                )
                entity_row = cursor.fetchone()

            if self.conn is None:
                conn.close()

            if entity_row:
                return dict(entity_row)

            return None

        except Exception as e:
            logging.error(f"Error getting entity ID: {e}")
            return None

    def close(self):
        """Clean up resources."""
        if self.conn is not None:
            self.conn.close()

    def __del__(self):
        """Destructor to clean up resources."""
        self.close()
