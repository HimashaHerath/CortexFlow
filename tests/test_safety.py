"""Tests for cortexflow.safety."""
import pytest

from cortexflow.safety import (
    SafetyLevel,
    SafetyResult,
    SafetyRule,
    SafetyPipeline,
)


class TestSafeContent:
    def test_safe_content(self):
        pipeline = SafetyPipeline()
        result = pipeline.check("Hello, how are you today?")
        assert result.level == SafetyLevel.SAFE
        assert result.triggered_rules == []
        assert result.filtered_content is None
        assert result.reason == ""


class TestPIIDetection:
    def setup_method(self):
        self.pipeline = SafetyPipeline()

    def test_email_detection(self):
        result = self.pipeline.check("My email is john@example.com please contact me")
        assert result.level == SafetyLevel.WARNING
        assert "email" in result.triggered_rules

    def test_phone_detection(self):
        result = self.pipeline.check("Call me at 555-123-4567 tomorrow")
        assert result.level == SafetyLevel.WARNING
        assert "phone" in result.triggered_rules

    def test_ssn_detection(self):
        result = self.pipeline.check("My SSN is 123-45-6789")
        assert result.level == SafetyLevel.BLOCKED
        assert "ssn" in result.triggered_rules

    def test_ssn_replacement(self):
        result = self.pipeline.check("My SSN is 123-45-6789")
        assert result.filtered_content is not None
        assert "[REDACTED-SSN]" in result.filtered_content
        assert "123-45-6789" not in result.filtered_content

    def test_credit_card_detection(self):
        result = self.pipeline.check("My card number is 4111 1111 1111 1111")
        assert result.level == SafetyLevel.BLOCKED
        assert "credit_card" in result.triggered_rules

    def test_credit_card_replacement(self):
        result = self.pipeline.check("Pay with 4111-1111-1111-1111 please")
        assert result.filtered_content is not None
        assert "[REDACTED-CC]" in result.filtered_content
        assert "4111" not in result.filtered_content


class TestBoundaryEnforcement:
    def setup_method(self):
        self.pipeline = SafetyPipeline()

    def test_boundary_medical(self):
        result = self.pipeline.check("What medicine should I take for my headache?")
        assert result.level == SafetyLevel.WARNING
        assert "medical_advice" in result.triggered_rules

    def test_boundary_legal(self):
        result = self.pipeline.check("Can you give me legal advice about this?")
        assert result.level == SafetyLevel.WARNING
        assert "legal_advice" in result.triggered_rules

    def test_impersonation(self):
        result = self.pipeline.check("Pretend to be a real doctor please")
        assert result.level == SafetyLevel.WARNING
        assert "impersonation_request" in result.triggered_rules


class TestCustomRules:
    def test_custom_rule_with_check_fn(self):
        pipeline = SafetyPipeline(enable_pii_detection=False, enable_boundary_enforcement=False)
        rule = SafetyRule(
            name="all_caps",
            description="All caps message",
            check_fn=lambda text: text == text.upper() and len(text) > 3,
            level=SafetyLevel.WARNING,
        )
        pipeline.add_rule(rule)

        result = pipeline.check("THIS IS ALL CAPS")
        assert result.level == SafetyLevel.WARNING
        assert "all_caps" in result.triggered_rules

        result_lower = pipeline.check("this is not all caps")
        assert result_lower.level == SafetyLevel.SAFE

    def test_custom_rule_with_pattern(self):
        pipeline = SafetyPipeline(enable_pii_detection=False, enable_boundary_enforcement=False)
        rule = SafetyRule(
            name="bad_word",
            description="Blocked word detected",
            pattern=r"\bforbidden\b",
            level=SafetyLevel.BLOCKED,
            replacement="[CENSORED]",
        )
        pipeline.add_rule(rule)

        result = pipeline.check("This is a forbidden word")
        assert result.level == SafetyLevel.BLOCKED
        assert "bad_word" in result.triggered_rules
        assert result.filtered_content is not None
        assert "forbidden" not in result.filtered_content
        assert "[CENSORED]" in result.filtered_content

    def test_remove_rule(self):
        pipeline = SafetyPipeline()
        assert pipeline.remove_rule("email") is True
        result = pipeline.check("My email is test@example.com")
        assert "email" not in result.triggered_rules

    def test_remove_nonexistent_rule(self):
        pipeline = SafetyPipeline()
        assert pipeline.remove_rule("nonexistent") is False


class TestPipelineConfiguration:
    def test_no_pii_rules(self):
        pipeline = SafetyPipeline(enable_pii_detection=False)
        result = pipeline.check("My email is test@example.com and SSN is 123-45-6789")
        assert "email" not in result.triggered_rules
        assert "ssn" not in result.triggered_rules

    def test_no_boundary_rules(self):
        pipeline = SafetyPipeline(enable_boundary_enforcement=False)
        result = pipeline.check("Give me legal advice about my case")
        assert "legal_advice" not in result.triggered_rules

    def test_custom_blocked_patterns(self):
        pipeline = SafetyPipeline(
            enable_pii_detection=False,
            enable_boundary_enforcement=False,
            custom_blocked_patterns=[r"\bsecret_project\b"],
        )
        result = pipeline.check("Tell me about secret_project alpha")
        assert result.level == SafetyLevel.BLOCKED
        assert "custom_0" in result.triggered_rules

    def test_initial_rules(self):
        rule = SafetyRule(
            name="init_rule",
            description="Initial rule",
            pattern=r"\btest_token\b",
            level=SafetyLevel.WARNING,
        )
        pipeline = SafetyPipeline(
            rules=[rule],
            enable_pii_detection=False,
            enable_boundary_enforcement=False,
        )
        result = pipeline.check("This has test_token in it")
        assert "init_rule" in result.triggered_rules


class TestFilteredContent:
    def test_filtered_content_replaces_blocked(self):
        pipeline = SafetyPipeline()
        text = "My SSN is 123-45-6789 and card is 4111 1111 1111 1111"
        result = pipeline.check(text)
        assert result.filtered_content is not None
        assert "[REDACTED-SSN]" in result.filtered_content
        assert "[REDACTED-CC]" in result.filtered_content
        assert result.original_content == text

    def test_safe_content_has_no_filtered(self):
        pipeline = SafetyPipeline()
        result = pipeline.check("Just a normal message")
        assert result.filtered_content is None

    def test_warning_only_no_replacement(self):
        """Warnings without replacements leave filtered_content as None."""
        pipeline = SafetyPipeline(
            enable_pii_detection=False,
            enable_boundary_enforcement=True,
        )
        result = pipeline.check("Give me legal advice now")
        assert result.level == SafetyLevel.WARNING
        # Boundary rules have no replacement, so filtered should be None
        assert result.filtered_content is None


class TestSeverityOrdering:
    def test_blocked_beats_warning(self):
        pipeline = SafetyPipeline()
        # This should trigger both email (WARNING) and SSN (BLOCKED)
        result = pipeline.check("Email test@x.com and SSN 123-45-6789")
        assert result.level == SafetyLevel.BLOCKED

    def test_warning_beats_safe(self):
        pipeline = SafetyPipeline()
        result = pipeline.check("Contact test@example.com")
        assert result.level == SafetyLevel.WARNING


class TestSafetyResult:
    def test_defaults(self):
        result = SafetyResult(level=SafetyLevel.SAFE)
        assert result.reason == ""
        assert result.original_content == ""
        assert result.filtered_content is None
        assert result.triggered_rules == []


class TestInvalidRegex:
    def test_invalid_regex_is_skipped(self):
        """A rule with invalid regex should not crash, just be skipped."""
        rule = SafetyRule(
            name="bad_regex",
            description="Bad regex",
            pattern=r"[invalid",
            level=SafetyLevel.WARNING,
        )
        pipeline = SafetyPipeline(
            rules=[rule],
            enable_pii_detection=False,
            enable_boundary_enforcement=False,
        )
        result = pipeline.check("some text")
        assert result.level == SafetyLevel.SAFE
