"""
CortexFlow Personal Fact Detector.

Lightweight regex-based detector for personal facts in conversation messages.
No external dependencies required (uses stdlib re). Optional spaCy NER boost
when available.
"""

from __future__ import annotations

import re


class PersonalFactDetector:
    """Detects personal facts in text using regex patterns.

    Recognizes common personal disclosure patterns such as names, occupations,
    locations, preferences, possessions, and age statements.

    Args:
        use_spacy: If True, attempt to load spaCy for NER boost. Default False
                   for speed (~5-20 microseconds per call with regex only).
    """

    # Pattern name -> list of compiled regexes
    # Each regex should capture the *value* in group 1
    _PATTERNS = {
        "name": [
            re.compile(r"\bmy name is\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
            re.compile(r"\bi(?:'m| am)\s+called\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
            re.compile(r"\bcall me\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
        ],
        "occupation": [
            re.compile(
                r"\bi work (?:at|for)\s+(.+?)(?:\s+as\b|\.|,|!|\?|$)", re.IGNORECASE
            ),
            re.compile(r"\bi(?:'m| am) an?\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
            re.compile(r"\bmy job is\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
            re.compile(r"\bi work as an?\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
        ],
        "location": [
            re.compile(r"\bi live in\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
            re.compile(r"\bi(?:'m| am) from\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
            re.compile(r"\bi was born in\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
            re.compile(r"\bi grew up in\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE),
        ],
        "preference": [
            re.compile(
                r"\bmy favo(?:u?)rite\s+(\w+)\s+is\s+(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bi (?:really )?(?:love|like|prefer|enjoy)\s+(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
        ],
        "possession": [
            re.compile(
                r"\bmy\s+(\w+(?:'s)?)\s+name is\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE
            ),
            re.compile(
                r"\bi have an?\s+(.+?)\s+named\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE
            ),
            re.compile(
                r"\bi have an?\s+(.+?)\s+called\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE
            ),
        ],
        "age": [
            re.compile(r"\bi(?:'m| am)\s+(\d+)\s+years?\s+old", re.IGNORECASE),
            re.compile(r"\bmy age is\s+(\d+)", re.IGNORECASE),
        ],
        # Companion-specific patterns (Phase 2B)
        "relationship_status": [
            re.compile(
                r"\bi(?:'m| am)\s+(single|married|divorced|engaged|in a relationship|dating|widowed)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bi have a\s+(boyfriend|girlfriend|partner|wife|husband|spouse)",
                re.IGNORECASE,
            ),
        ],
        "emotional_state": [
            re.compile(
                r"\bi(?:'m| am) (?:feeling |)(happy|sad|anxious|depressed|lonely|stressed|excited|scared|angry|grateful|hopeful|overwhelmed)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bi(?:'ve| have) been (?:feeling |)(happy|sad|anxious|depressed|lonely|stressed|excited|scared|angry|grateful|hopeful|overwhelmed)\b",
                re.IGNORECASE,
            ),
        ],
        "routine": [
            re.compile(
                r"\bevery (?:day|morning|evening|night|week)\s+i\s+(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bi (?:usually|always|typically)\s+(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
        ],
        "boundary": [
            re.compile(
                r"\bi(?:'m| am) not comfortable (?:with |talking about |discussing )(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bplease (?:don't|do not) (?:talk about |mention |bring up )(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bi(?:'d| would) (?:rather|prefer) not (?:talk about |discuss )(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
        ],
        "interest": [
            re.compile(
                r"\bi(?:'m| am) (?:really )?interested in\s+(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bi(?:'m| am) (?:really )?into\s+(.+?)(?:\.|,|!|\?|$)", re.IGNORECASE
            ),
            re.compile(
                r"\bmy hobbi(?:es|y)\s+(?:is|are|include)\s+(.+?)(?:\.|,|!|\?|$)",
                re.IGNORECASE,
            ),
        ],
    }

    # Quick-check pattern: any of these substrings hint at a personal fact.
    # Used by contains_personal_fact() for the fast path.
    _QUICK_HINTS = re.compile(
        r"\bmy name\b|\bcall me\b|\bi work\b|\bmy job\b"
        r"|\bi live\b|\bi(?:'m| am) from\b|\bi was born\b"
        r"|\bmy favo(?:u?)rite\b|\bi have a\b|\bmy \w+'s name\b"
        r"|\bi(?:'m| am) \d+ years?\b|\bmy age\b"
        r"|\bi(?:'m| am) an?\b|\bi grew up\b"
        # Companion-specific hints
        r"|\bi(?:'m| am) (?:single|married|divorced|engaged|dating)"
        r"|\bi(?:'m| am) (?:feeling )?(?:happy|sad|anxious|depressed|lonely|stressed)"
        r"|\bevery (?:day|morning|evening|night|week) i\b"
        r"|\bi(?:'m| am) not comfortable\b|\bplease (?:don't|do not)\b"
        r"|\bi(?:'m| am) (?:really )?interested in\b|\bmy hobbi",
        re.IGNORECASE,
    )

    # Maximum input length to protect against regex DoS
    MAX_INPUT_LENGTH = 10_000

    def __init__(self, use_spacy: bool = False):
        self._nlp = None
        if use_spacy:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                import logging

                logging.getLogger("cortexflow").debug(
                    "spaCy not available for fact detection, using regex only"
                )

    def contains_personal_fact(self, text: str) -> bool:
        """Fast boolean check for personal facts.

        Uses a single compiled regex with alternation for speed. Suitable
        for the importance-scoring hot path.

        Args:
            text: Input text to check.

        Returns:
            True if the text likely contains a personal fact.
        """
        if len(text) > self.MAX_INPUT_LENGTH:
            text = text[: self.MAX_INPUT_LENGTH]
        return bool(self._QUICK_HINTS.search(text))

    def detect_facts(self, text: str) -> list[dict]:
        """Extract personal facts from text.

        Args:
            text: Input text to analyze.

        Returns:
            List of dicts with keys: fact_text, fact_type, confidence, value.
        """
        if len(text) > self.MAX_INPUT_LENGTH:
            text = text[: self.MAX_INPUT_LENGTH]
        facts: list[dict] = []
        seen_values: set = set()

        for fact_type, patterns in self._PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    groups = match.groups()
                    # For multi-group patterns (preference/possession),
                    # combine groups into the value
                    if len(groups) >= 2:
                        value = " ".join(g.strip() for g in groups if g)
                    else:
                        value = groups[0].strip() if groups[0] else ""

                    if not value or value.lower() in seen_values:
                        continue
                    seen_values.add(value.lower())

                    # Use the full matched span as fact_text
                    fact_text = match.group(0).strip().rstrip(".,!?")

                    confidence = 0.85
                    # Boost confidence if spaCy NER confirms an entity
                    if self._nlp and fact_type in ("name", "location", "occupation"):
                        doc = self._nlp(value)
                        if any(
                            ent.label_ in ("PERSON", "GPE", "ORG", "LOC")
                            for ent in doc.ents
                        ):
                            confidence = 0.95

                    facts.append(
                        {
                            "fact_text": fact_text,
                            "fact_type": fact_type,
                            "confidence": confidence,
                            "value": value,
                        }
                    )

        return facts
