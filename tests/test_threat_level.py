"""Tests for threat_level validation and defaulting."""
import pytest

from tg_compiler.analyzer import PostAnalysis


def _base(**kwargs) -> dict:
    return {
        "summary": "Test event.",
        "importance_score": 3,
        "urgency_score": 3,
        "credibility_score": 3,
        "relevance_score": 3,
        "category": "Analysis",
        **kwargs,
    }


@pytest.mark.parametrize("level", ["CRITICAL", "HIGH", "MODERATE", "LOW"])
def test_valid_threat_levels_accepted(level):
    pa = PostAnalysis.model_validate(_base(threat_level=level))
    assert pa.threat_level == level


@pytest.mark.parametrize("bad", ["EXTREME", "medium", "", "5", "None", "null"])
def test_invalid_threat_level_defaults_to_moderate(bad):
    pa = PostAnalysis.model_validate(_base(threat_level=bad))
    assert pa.threat_level == "MODERATE"


def test_threat_level_case_normalised():
    pa = PostAnalysis.model_validate(_base(threat_level="critical"))
    assert pa.threat_level == "CRITICAL"


def test_threat_level_missing_defaults_to_moderate():
    pa = PostAnalysis.model_validate(_base())
    assert pa.threat_level == "MODERATE"
