"""Safety pipeline for CortexFlow -- content filtering and boundary enforcement."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("cortexflow")


class SafetyLevel(Enum):
    """Safety check result levels."""

    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class SafetyResult:
    """Result of a safety check."""

    level: SafetyLevel
    reason: str = ""
    original_content: str = ""
    filtered_content: str | None = None  # None means use original
    triggered_rules: list[str] = field(default_factory=list)


@dataclass
class SafetyRule:
    """A configurable safety rule."""

    name: str
    description: str
    pattern: str | None = None  # regex pattern
    check_fn: Callable[[str], bool] | None = None  # custom check function
    level: SafetyLevel = SafetyLevel.WARNING
    replacement: str | None = None  # replacement text if blocked


class SafetyPipeline:
    """Configurable content safety pipeline with rule-based filtering."""

    def __init__(
        self,
        rules: list[SafetyRule] | None = None,
        enable_pii_detection: bool = True,
        enable_boundary_enforcement: bool = True,
        custom_blocked_patterns: list[str] | None = None,
    ):
        self._rules: list[SafetyRule] = rules or []
        self._compiled_patterns: dict[str, re.Pattern] = {}

        if enable_pii_detection:
            self._add_pii_rules()
        if enable_boundary_enforcement:
            self._add_boundary_rules()
        if custom_blocked_patterns:
            for i, pat in enumerate(custom_blocked_patterns):
                self._rules.append(
                    SafetyRule(
                        name=f"custom_{i}",
                        description=f"Custom blocked pattern: {pat[:30]}",
                        pattern=pat,
                        level=SafetyLevel.BLOCKED,
                    )
                )

        # Compile all regex patterns
        for rule in self._rules:
            if rule.pattern:
                try:
                    self._compiled_patterns[rule.name] = re.compile(
                        rule.pattern,
                        re.IGNORECASE,
                    )
                except re.error as e:
                    logger.warning(f"Invalid regex in safety rule '{rule.name}': {e}")

    # ------------------------------------------------------------------
    # Built-in rule sets
    # ------------------------------------------------------------------

    def _add_pii_rules(self) -> None:
        """Add built-in PII detection rules."""
        self._rules.extend(
            [
                SafetyRule(
                    name="email",
                    description="Email address detection",
                    pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    level=SafetyLevel.WARNING,
                ),
                SafetyRule(
                    name="phone",
                    description="Phone number detection",
                    pattern=r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                    level=SafetyLevel.WARNING,
                ),
                SafetyRule(
                    name="ssn",
                    description="SSN detection",
                    pattern=r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
                    level=SafetyLevel.BLOCKED,
                    replacement="[REDACTED-SSN]",
                ),
                SafetyRule(
                    name="credit_card",
                    description="Credit card number detection",
                    pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                    level=SafetyLevel.BLOCKED,
                    replacement="[REDACTED-CC]",
                ),
            ]
        )

    def _add_boundary_rules(self) -> None:
        """Add boundary enforcement rules for companion AI."""
        self._rules.extend(
            [
                SafetyRule(
                    name="impersonation_request",
                    description="Request to impersonate real person",
                    pattern=r"\b(?:pretend|act|roleplay)\s+(?:to be|as|like)\s+(?:a real|an? actual)\b",
                    level=SafetyLevel.WARNING,
                ),
                SafetyRule(
                    name="medical_advice",
                    description="Request for medical diagnosis",
                    pattern=r"\b(?:diagnose|prescription|what medicine|what drug)\s+(?:should I|do I|for my)\b",
                    level=SafetyLevel.WARNING,
                ),
                SafetyRule(
                    name="legal_advice",
                    description="Request for legal counsel",
                    pattern=r"\b(?:legal advice|am I guilty|sue them|file a lawsuit)\b",
                    level=SafetyLevel.WARNING,
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def check(self, content: str) -> SafetyResult:
        """Run all safety rules against content. Returns the highest-severity result."""
        triggered: list[str] = []
        highest_level = SafetyLevel.SAFE
        filtered = content
        reasons: list[str] = []

        for rule in self._rules:
            matched = False

            if rule.name in self._compiled_patterns:
                pattern = self._compiled_patterns[rule.name]
                if pattern.search(content):
                    matched = True

            if rule.check_fn and rule.check_fn(content):
                matched = True

            if matched:
                triggered.append(rule.name)
                reasons.append(rule.description)

                if self._level_severity(rule.level) > self._level_severity(
                    highest_level
                ):
                    highest_level = rule.level

                if (
                    rule.replacement
                    and rule.pattern
                    and rule.name in self._compiled_patterns
                ):
                    filtered = self._compiled_patterns[rule.name].sub(
                        rule.replacement,
                        filtered,
                    )

        return SafetyResult(
            level=highest_level,
            reason="; ".join(reasons) if reasons else "",
            original_content=content,
            filtered_content=filtered if filtered != content else None,
            triggered_rules=triggered,
        )

    def add_rule(self, rule: SafetyRule) -> None:
        """Add a custom safety rule."""
        self._rules.append(rule)
        if rule.pattern:
            try:
                self._compiled_patterns[rule.name] = re.compile(
                    rule.pattern,
                    re.IGNORECASE,
                )
            except re.error as e:
                logger.warning(f"Invalid regex in safety rule '{rule.name}': {e}")

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        self._compiled_patterns.pop(name, None)
        return len(self._rules) < original_len

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _level_severity(level: SafetyLevel) -> int:
        """Return numeric severity for comparison."""
        return {"safe": 0, "warning": 1, "blocked": 2}[level.value]
