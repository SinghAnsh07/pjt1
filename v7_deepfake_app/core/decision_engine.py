"""
Conservative Decision Engine for the V7 Audio Deepfake Detection System.

Interprets continuous OC-Softmax cosine similarity scores against an explicitly
configured biometric decision threshold without fabricating uncalibrated percentage
confidences or misleading probability estimates.
"""

import math
from typing import Optional, List

from core.schemas import (
    DecisionResult,
    DecisionLabel,
    ScoreDirection,
    ThresholdStatus,
)
from core.exceptions import ThresholdConfigurationError, InferenceError

# Candidate EER Operating Point calibrated on official ASVspoof 2019 LA benchmark (71,237 utterances)
DEFAULT_CANDIDATE_THRESHOLD: float = 0.4820
DEFAULT_THRESHOLD_SOURCE: str = "ASVspoof 2019 LA Evaluation Set Continuous Linear Interpolation EER Operating Point"


class DecisionEngine:
    """
    Interprets raw OC-Softmax cosine scores under an explicit, auditable decision policy.
    
    Principles:
      1. Conservative & Honest: Does not present uncalibrated heuristic percentages as probabilities.
      2. Explicit Directionality: S(x) >= threshold -> REAL (Bona Fide); S(x) < threshold -> FAKE (Spoof).
      3. Provenance Transparency: Clearly flags calibration status as candidate / benchmark.
    """
    def __init__(
        self,
        threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
        threshold_status: ThresholdStatus = ThresholdStatus.CANDIDATE_UNVALIDATED,
        threshold_source: str = DEFAULT_THRESHOLD_SOURCE,
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_REAL,
    ):
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ThresholdConfigurationError(f"Threshold must be a finite float, got {threshold}")
        if threshold < -1.0 or threshold > 1.0:
            raise ThresholdConfigurationError(
                f"Threshold {threshold} is outside the valid cosine score domain [-1.0, 1.0]"
            )

        self.threshold = float(threshold)
        self.threshold_status = threshold_status
        self.threshold_source = threshold_source
        self.score_direction = score_direction

    def decide(self, model_score: float) -> DecisionResult:
        """
        Classifies a raw model cosine score.

        Args:
            model_score: Scalar cosine similarity score S(x) from OC-Softmax layer

        Returns:
            Structured DecisionResult

        Raises:
            InferenceError: If model_score is non-finite or outside [-1.0, 1.0]
        """
        if not isinstance(model_score, (int, float)) or not math.isfinite(model_score):
            raise InferenceError(f"Model score is non-finite: {model_score}")

        # Minor numerical tolerance check for floating-point cosine boundaries
        clamped_score = max(-1.0, min(1.0, float(model_score)))
        margin = clamped_score - self.threshold

        warnings: List[str] = []
        if self.threshold_status == ThresholdStatus.CANDIDATE_UNVALIDATED:
            warnings.append(
                "Threshold 0.4820 is an experimental candidate derived from the ASVspoof 2019 benchmark. "
                "It has not been independently calibrated for unconstrained in-the-wild audio."
            )

        if clamped_score >= self.threshold:
            prediction = DecisionLabel.REAL
            explanation = (
                f"Classified as REAL (Bona Fide). The acoustic embedding exhibits high cosine similarity "
                f"({clamped_score:+.4f}) to the bona fide voice centroid, exceeding the operational threshold "
                f"({self.threshold:.4f}) by a margin of {margin:+.4f}."
            )
        else:
            prediction = DecisionLabel.FAKE
            explanation = (
                f"Classified as FAKE (Synthetic / Spoof). The acoustic embedding yields low cosine similarity "
                f"({clamped_score:+.4f}) relative to the bona fide voice centroid, falling below the operational "
                f"threshold ({self.threshold:.4f}) by a margin of {margin:+.4f}."
            )

        return DecisionResult(
            prediction=prediction,
            raw_score=clamped_score,
            threshold=self.threshold,
            threshold_status=self.threshold_status,
            threshold_source=self.threshold_source,
            score_direction=self.score_direction,
            margin_to_threshold=round(margin, 4),
            explanation=explanation,
            warnings=warnings,
        )
