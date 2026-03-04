"""
RelationExtractor class and shared SVO extraction helpers.

Provides advanced relation extraction capabilities using dependency parsing,
semantic role labeling, and relation classification.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ._deps import SPACY_ENABLED, spacy

# ---------------------------------------------------------------------------
# Shared SVO extraction helpers (used by both RelationExtractor and GraphStore)
# ---------------------------------------------------------------------------


def _extract_svo_triples_from_sentence(
    sent, get_span_text
) -> list[tuple[str, str, str]]:
    """Extract Subject-Verb-Object triples from a single spaCy sentence span.

    Args:
        sent: A spaCy Span representing one sentence.
        get_span_text: Callable(token) -> str that expands a token into its
            full noun-phrase text.  Both RelationExtractor and GraphStore
            supply their own ``_get_span_text`` implementations.

    Returns:
        List of (subject, predicate, object) tuples.
    """
    triples: list[tuple[str, str, str]] = []

    verbs = [token for token in sent if token.pos_ == "VERB"]
    if not verbs:
        return triples

    for verb in verbs:
        subjects = []
        objects: list[str | tuple[str, str]] = []

        for token in sent:
            # Subjects
            if (
                token.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "agent")
                and token.head == verb
            ):
                subjects.append(get_span_text(token))
            # Direct / indirect objects
            elif token.dep_ in ("dobj", "pobj", "iobj", "obj") and token.head == verb:
                objects.append(get_span_text(token))
            # Prepositional phrases attached to the verb
            elif token.dep_ == "prep" and token.head == verb:
                for child in token.children:
                    if child.dep_ == "pobj":
                        pred = f"{verb.lemma_} {token.text}"
                        objects.append((pred, get_span_text(child)))

        for subject in subjects:
            for obj in objects:
                if isinstance(obj, tuple):
                    pred, obj_text = obj
                    triples.append((subject, pred, obj_text))
                else:
                    triples.append((subject, verb.lemma_, obj))

    return triples


class RelationExtractor:
    """
    Dedicated relation extraction class for CortexFlow knowledge graph.
    Provides advanced relation extraction capabilities using dependency parsing,
    semantic role labeling, and relation classification.
    """

    def __init__(self, nlp=None):
        """
        Initialize relation extractor.

        Args:
            nlp: spaCy language model, or None to create a new one
        """
        # Initialize spaCy model if needed
        self.nlp = nlp
        if SPACY_ENABLED and not self.nlp:
            try:
                # Use a model with dependency parsing for relation extraction
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("Relation Extractor: spaCy model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading spaCy model for relation extraction: {e}")
                self.nlp = None

        # Initialize SRL (Semantic Role Labeling) components
        self.srl_predictor = None
        try:
            from allennlp.predictors.predictor import Predictor

            self.srl_predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
            )
            logging.info("SRL model loaded successfully")
        except ImportError:
            logging.warning(
                "AllenNLP SRL not available. Semantic role labeling is disabled."
            )
        except Exception as e:
            logging.warning(
                f"AllenNLP SRL model failed to load. Semantic role labeling is disabled: {e}"
            )

        # Define relation patterns and templates
        self.relation_patterns = self._init_relation_patterns()

    def _init_relation_patterns(self) -> dict[str, list[dict[str, Any]]]:
        """
        Initialize common relation patterns for rule-based extraction.

        Returns:
            Dictionary of relation patterns
        """
        patterns = {
            "is_a": [
                {"pattern": r"([^\s]+) is (?:a|an) ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) are ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) classified as ([^\s]+)", "groups": (1, 2)},
            ],
            "part_of": [
                {"pattern": r"([^\s]+) is part of ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) belongs to ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) contains ([^\s]+)", "groups": (2, 1)},
            ],
            "located_in": [
                {"pattern": r"([^\s]+) is (?:in|at|on) ([^\s]+)", "groups": (1, 2)},
                {
                    "pattern": r"([^\s]+) is located (?:in|at|on) ([^\s]+)",
                    "groups": (1, 2),
                },
            ],
            "has_property": [
                {"pattern": r"([^\s]+) has (?:a|an)? ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) with (?:a|an)? ([^\s]+)", "groups": (1, 2)},
            ],
            "causes": [
                {"pattern": r"([^\s]+) causes ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) leads to ([^\s]+)", "groups": (1, 2)},
                {"pattern": r"([^\s]+) results in ([^\s]+)", "groups": (1, 2)},
            ],
        }
        return patterns

    def extract_relations(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract subject-predicate-object triples from text using multiple techniques.

        Args:
            text: Input text to extract relations from

        Returns:
            List of (subject, predicate, object) tuples
        """
        relations = []

        if SPACY_ENABLED and self.nlp is not None:
            try:
                doc = self.nlp(text)

                # 1. Extract relations using dependency parsing
                dep_relations = self.extract_svo_from_dependency(doc)
                relations.extend(dep_relations)

                # 2. Extract prepositional relations
                prep_relations = self.extract_prep_relations(doc)
                relations.extend(prep_relations)

                # 3. Extract relations using semantic role labeling
                if self.srl_predictor:
                    srl_relations = self.extract_with_semantic_roles(text)
                    relations.extend(srl_relations)

                # 4. Extract relations using pattern matching
                pattern_relations = self.extract_with_patterns(text)
                relations.extend(pattern_relations)

            except Exception as e:
                logging.error(f"Error in relation extraction: {e}")
                import traceback

                logging.error(f"Traceback: {traceback.format_exc()}")

        # Deduplicate relations
        unique_relations = []
        relation_strings = set()

        for subj, pred, obj in relations:
            rel_str = f"{subj.lower()}|{pred.lower()}|{obj.lower()}"
            if rel_str not in relation_strings:
                relation_strings.add(rel_str)
                unique_relations.append((subj, pred, obj))

        return unique_relations

    def extract_svo_from_dependency(self, doc) -> list[tuple[str, str, str]]:
        """
        Extract Subject-Verb-Object triples from a document using dependency parsing.

        Args:
            doc: spaCy document

        Returns:
            List of (subject, predicate, object) tuples
        """
        triples = []
        for sent in doc.sents:
            triples.extend(
                _extract_svo_triples_from_sentence(sent, self._get_span_text)
            )
        return triples

    def extract_prep_relations(self, doc) -> list[tuple[str, str, str]]:
        """
        Extract relations based on prepositional phrases like "X in Y".

        Args:
            doc: spaCy document

        Returns:
            List of (entity1, preposition, entity2) tuples
        """
        prep_relations = []

        for sent in doc.sents:
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

    def extract_with_semantic_roles(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract relations using semantic role labeling (SRL).

        Args:
            text: Input text

        Returns:
            List of (subject, predicate, object) tuples
        """
        if not self.srl_predictor:
            return []

        try:
            srl_output = self.srl_predictor.predict(sentence=text)

            # Extract relations from SRL output
            relations = []
            for verb_data in srl_output.get("verbs", []):
                predicate = verb_data["verb"]

                # Process tagged spans to extract arguments
                arg0 = None
                arg1 = None
                arg2 = None
                loc = None
                tmp = None

                # Extract arguments from tags
                tagged_string = verb_data["description"]
                current_arg = None
                current_text = ""

                for part in tagged_string.split():
                    if part.startswith("["):
                        # Start of new argument
                        if current_arg and current_text:
                            if current_arg == "ARG0":
                                arg0 = current_text.strip()
                            elif current_arg == "ARG1":
                                arg1 = current_text.strip()
                            elif current_arg == "ARG2":
                                arg2 = current_text.strip()
                            elif current_arg.startswith("ARGM-LOC"):
                                loc = current_text.strip()
                            elif current_arg.startswith("ARGM-TMP"):
                                tmp = current_text.strip()

                        # Set new current argument
                        if "*" in part:
                            label_end = part.find("*")
                            current_arg = part[1:label_end]
                            current_text = part[label_end + 1 :]
                            if part.endswith("]"):
                                current_text = current_text[:-1]
                    elif part.endswith("]"):
                        # End of current argument
                        current_text += " " + part[:-1]

                        if current_arg == "ARG0":
                            arg0 = current_text.strip()
                        elif current_arg == "ARG1":
                            arg1 = current_text.strip()
                        elif current_arg == "ARG2":
                            arg2 = current_text.strip()
                        elif current_arg.startswith("ARGM-LOC"):
                            loc = current_text.strip()
                        elif current_arg.startswith("ARGM-TMP"):
                            tmp = current_text.strip()

                        current_arg = None
                        current_text = ""
                    elif current_arg:
                        # Continue current argument
                        current_text += " " + part

                # Create relations from arguments
                if arg0 and arg1:
                    relations.append((arg0, predicate, arg1))
                if arg0 and arg2:
                    relations.append((arg0, predicate + " to", arg2))
                if arg1 and loc:
                    relations.append((arg1, "located in", loc))
                if arg0 and loc:
                    relations.append((arg0, "located in", loc))
                if arg0 and tmp:
                    relations.append((arg0, "at time", tmp))

            return relations

        except Exception as e:
            logging.error(f"Error in semantic role labeling: {e}")
            return []

    def extract_with_patterns(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract relations using pattern-based rules.

        Args:
            text: Input text

        Returns:
            List of (subject, relation, object) tuples
        """
        relations = []

        # Apply each relation pattern
        for relation_type, patterns in self.relation_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                group_indices = pattern_info["groups"]

                # Find matches in text
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    try:
                        # Extract subject and object from specified groups
                        subject = match.group(group_indices[0]).strip()
                        obj = match.group(group_indices[1]).strip()

                        # Add to relations
                        relations.append((subject, relation_type, obj))
                    except IndexError:
                        continue

        return relations

    def _get_span_text(self, token) -> str:
        """
        Get the text of the full noun phrase for a token.

        Args:
            token: spaCy token

        Returns:
            Text of the full noun phrase
        """
        # If token is part of a compound, get the full compound
        if token.pos_ in ["NOUN", "PROPN"]:
            # Start with the token itself
            start = token
            # Traverse left children to find compound parts
            lefts = list(token.lefts)
            for left in lefts:
                if left.dep_ in ["compound", "amod", "det", "nummod"]:
                    if left.i < start.i:
                        start = left

            # Traverse right children to find the end of the noun phrase
            end = token
            rights = list(token.rights)
            for right in rights:
                if right.dep_ in ["compound", "amod"]:
                    if right.i > end.i:
                        end = right

            # Get the full span text
            span_tokens = [t for t in start.doc if start.i <= t.i <= end.i]
            return " ".join([t.text for t in span_tokens])

        return token.text
