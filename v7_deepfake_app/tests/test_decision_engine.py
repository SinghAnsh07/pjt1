"""
Automated unit tests for Conservative Decision Engine.
"""

import pytest
import math

from core.decision_engine import DecisionEngine
from core.schemas import DecisionLabel, ScoreDirection, ThresholdStatus
from core.exceptions import ThresholdConfigurationError, InferenceError


def test_decision_above_threshold():
    """Verifies that a score strictly above the threshold is classified as REAL."""
    engine = DecisionEngine(threshold=0.4820)
    decision = engine.decide(0.6500)

    assert decision.prediction == DecisionLabel.REAL
    assert decision.raw_score == 0.6500
    assert decision.threshold == 0.4820
    assert decision.margin_to_threshold > 0
    assert decision.score_direction == ScoreDirection.HIGHER_IS_REAL
    assert "REAL (Bona Fide)" in decision.explanation


def test_decision_equal_to_threshold():
    """Verifies that a score exactly equal to the threshold is classified as REAL (boundary condition)."""
    engine = DecisionEngine(threshold=0.4820)
    decision = engine.decide(0.4820)

    assert decision.prediction == DecisionLabel.REAL
    assert decision.margin_to_threshold == 0.0


def test_decision_below_threshold():
    """Verifies that a score strictly below the threshold is classified as FAKE."""
    engine = DecisionEngine(threshold=0.4820)
    decision = engine.decide(-0.8500)

    assert decision.prediction == DecisionLabel.FAKE
    assert decision.raw_score == -0.8500
    assert decision.margin_to_threshold < 0
    assert "FAKE (Synthetic / Spoof)" in decision.explanation


def test_no_arbitrary_percentage_confidences():
    """Verifies that the decision engine does not manufacture fake probability or confidence percentages."""
    engine = DecisionEngine()
    decision = engine.decide(0.8500)
    decision_dict = decision.to_dict()

    # Ensure no keys such as 'confidence_percent', 'probability', 'prob_real', etc.
    assert "confidence_percentage" not in decision_dict
    assert "probability" not in decision_dict
    assert "prob_real" not in decision_dict
    assert "prob_fake" not in decision_dict


def test_invalid_threshold_rejection():
    """Verifies that out-of-domain or non-finite thresholds are rejected."""
    with pytest.raises(ThresholdConfigurationError):
        DecisionEngine(threshold=1.5)  # Outside [-1.0, 1.0]
    with pytest.raises(ThresholdConfigurationError):
        DecisionEngine(threshold=-1.5)
    with pytest.raises(ThresholdConfigurationError):
        DecisionEngine(threshold=float("nan"))


def test_non_finite_score_rejection():
    """Verifies that NaN or Inf model scores are rejected."""
    engine = DecisionEngine(threshold=0.4820)
    with pytest.raises(InferenceError):
        engine.decide(float("nan"))
    with pytest.raises(InferenceError):
        engine.decide(float("inf"))


def test_candidate_threshold_status_warning():
    """Verifies that candidate threshold status adds appropriate transparency warning."""
    engine = DecisionEngine(threshold=0.4820, threshold_status=ThresholdStatus.CANDIDATE_UNVALIDATED)
    decision = engine.decide(0.5000)

    assert decision.threshold_status == ThresholdStatus.CANDIDATE_UNVALIDATED
    assert len(decision.warnings) > 0
    assert "ASVspoof 2019" in decision.warnings[0]
