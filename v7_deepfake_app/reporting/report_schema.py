"""
Canonical report payload schemas and validation for the V7 Audio Authenticity Analyzer.

Defines the single source of truth for both JSON export and PDF document generation,
ensuring strict agreement on all metadata, metrics, scores, and disclaimers.
"""

import os
import re
import math
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union

from core.schemas import (
    LoadedAudio,
    InferenceResult,
    LongRecordingInferenceResult,
    DecisionLabel,
    ModelMetadata,
)
from core.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_DURATION_SECONDS,
    DEFAULT_HOP_SECONDS,
)
from core.decision_engine import (
    DEFAULT_CANDIDATE_THRESHOLD,
    DEFAULT_THRESHOLD_SOURCE,
)


def sanitize_filename(filename: Optional[str]) -> str:
    """
    Sanitizes untrusted filenames to prevent path traversal or document formatting breakage.
    Strips directory separators and limits character set to safe alphanumeric/punctuation.
    """
    if not filename or not isinstance(filename, str):
        return "unnamed_audio_file"

    # Strip full filesystem path
    base = os.path.basename(filename).strip()
    # Remove potentially dangerous characters
    cleaned = re.sub(r'[^a-zA-Z0-9_\-\.\s\(\)]', '_', base)
    # Truncate overly long filenames
    if len(cleaned) > 80:
        cleaned = cleaned[:76] + "..."
    return cleaned or "unnamed_audio_file"


@dataclass
class ReportMetadata:
    """Metadata identifying the generation context and scientific governance."""
    report_id: str
    schema_version: str = "1.0.0"
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    application_name: str = "V7 Audio Authenticity Analyzer"
    model_name: str = "2D-FrFT Wavelet Light CNN (LCNN-2D)"
    model_version: str = "v7_2d_frft_wavelet_lcnn"
    checkpoint_sha256: str = "b356dacc24dfc1ff463b1f1c5f4488cf374330f6c2edae6c5dd5abe79b76397b"
    classification_scope: str = "Automated Biometric Screening Tool (Not a Legal Forensic Certification)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportMediaInfo:
    """Sanitized metadata describing the ingested audio container."""
    filename: str
    media_type: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    format_name: str
    subtype: str
    is_resampled: bool
    is_downmixed: bool
    is_padded: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportAnalysisConfig:
    """Model operational parameters and scoring thresholds."""
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE
    window_duration_seconds: float = WINDOW_DURATION_SECONDS
    hop_duration_seconds: Optional[float] = DEFAULT_HOP_SECONDS
    scoring_metric: str = "One-Class Softmax Cosine Similarity S(x)"
    score_domain: str = "[-1.0, +1.0] (Higher scores indicate greater similarity to bona fide centroid)"
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD
    threshold_status: str = "CANDIDATE_UNVALIDATED (Empirically derived from ASVspoof 2019 LA benchmark)"
    threshold_source: str = DEFAULT_THRESHOLD_SOURCE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportWindowItem:
    """Individual sliding window observation across time."""
    window_index: int
    start_seconds: float
    end_seconds: float
    score: float
    prediction: str
    margin: float
    is_padded: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportResultSummary:
    """Structured forensic findings across single-window or multi-window evaluation."""
    analysis_mode: str  # "single_window" or "multi_window_timeline"
    overall_screening_finding: str
    primary_score: float
    decision_label: Optional[str]
    total_windows: int
    likely_genuine_windows: int
    potential_fake_windows: int
    flagged_window_percentage: float
    min_window_score: Optional[float]
    max_window_score: Optional[float]
    mean_window_score: Optional[float]
    windows: List[ReportWindowItem] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["windows"] = [w.to_dict() for w in self.windows]
        return res


@dataclass
class ReportPayload:
    """Complete, self-contained report payload for JSON export and PDF generation."""
    metadata: ReportMetadata
    media: ReportMediaInfo
    config: ReportAnalysisConfig
    result: ReportResultSummary
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "media": self.media.to_dict(),
            "config": self.config.to_dict(),
            "result": self.result.to_dict(),
            "limitations": self.limitations,
        }


def build_report_payload(
    loaded: LoadedAudio,
    result: Union[InferenceResult, LongRecordingInferenceResult],
    engine_metadata: Optional[ModelMetadata] = None,
    candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
) -> ReportPayload:
    """
    Constructs a validated, privacy-hardened ReportPayload from loaded audio and inference results.
    Does NOT rerun model inference or access disk media.

    Args:
        loaded: LoadedAudio instance containing container metadata
        result: InferenceResult or LongRecordingInferenceResult
        engine_metadata: Optional ModelMetadata from loaded V7 engine
        candidate_threshold: Operational decision threshold

    Returns:
        ReportPayload: Validated, serializable report data structure
    """
    report_id = f"v7_{uuid.uuid4().hex[:10]}"
    sha = engine_metadata.checkpoint_sha256 if engine_metadata else "b356dacc24dfc1ff463b1f1c5f4488cf374330f6c2edae6c5dd5abe79b76397b"

    metadata = ReportMetadata(
        report_id=report_id,
        checkpoint_sha256=sha,
    )

    is_long = isinstance(result, LongRecordingInferenceResult)
    safe_filename = sanitize_filename(getattr(result, "filename", None) or getattr(loaded, "format_name", "audio"))

    # Determine preprocessing transformations from available metadata
    is_resampled = (loaded.sample_rate != TARGET_SAMPLE_RATE)
    is_downmixed = (loaded.channels > 1)
    is_padded = False

    if not is_long and hasattr(result, "preprocessing"):
        is_padded = result.preprocessing.is_padded
        is_resampled = result.preprocessing.is_resampled
        is_downmixed = result.preprocessing.is_multichannel

    media_info = ReportMediaInfo(
        filename=safe_filename,
        media_type="audio",
        duration_seconds=round(float(loaded.duration_seconds), 3),
        sample_rate_hz=int(loaded.sample_rate),
        channels=int(loaded.channels),
        format_name=str(loaded.format_name),
        subtype=str(loaded.subtype),
        is_resampled=bool(is_resampled),
        is_downmixed=bool(is_downmixed),
        is_padded=bool(is_padded),
    )

    analysis_config = ReportAnalysisConfig(
        target_sample_rate_hz=TARGET_SAMPLE_RATE,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
        hop_duration_seconds=DEFAULT_HOP_SECONDS if is_long else None,
        candidate_threshold=candidate_threshold,
    )

    # Compile result summary
    if not is_long:
        # Single window result
        pred_label = result.decision.prediction.value
        finding = "Likely Genuine Audio" if result.decision.prediction == DecisionLabel.REAL else "Potential AI-Generated or Manipulated Audio"
        score = float(result.model_score) if math.isfinite(result.model_score) else 0.0

        res_summary = ReportResultSummary(
            analysis_mode="single_window",
            overall_screening_finding=finding,
            primary_score=round(score, 4),
            decision_label=pred_label,
            total_windows=1,
            likely_genuine_windows=1 if result.decision.prediction == DecisionLabel.REAL else 0,
            potential_fake_windows=0 if result.decision.prediction == DecisionLabel.REAL else 1,
            flagged_window_percentage=0.0 if result.decision.prediction == DecisionLabel.REAL else 100.0,
            min_window_score=round(score, 4),
            max_window_score=round(score, 4),
            mean_window_score=round(score, 4),
            windows=[
                ReportWindowItem(
                    window_index=0,
                    start_seconds=0.0,
                    end_seconds=round(float(loaded.duration_seconds), 3),
                    score=round(score, 4),
                    prediction=pred_label,
                    margin=round(float(result.decision.margin_to_threshold), 4),
                    is_padded=bool(is_padded),
                )
            ],
            total_processing_time_ms=round(float(result.timing.total_ms), 1),
            warnings=list(result.system_warnings),
        )
    else:
        # Multi-window timeline result
        real_count = result.summary_real_windows
        fake_count = result.summary_fake_windows
        tot_windows = result.total_windows or len(result.windows)
        flagged_pct = round((fake_count / tot_windows) * 100.0, 1) if tot_windows > 0 else 0.0

        if fake_count == 0:
            finding = "All Analyzed Windows Meet Bona Fide Screening Threshold"
        elif real_count == 0:
            finding = "All Analyzed Windows Diverge from Bona Fide Centroid"
        else:
            finding = f"Mixed Window Classifications ({real_count} Genuine, {fake_count} Potential Spoof)"

        window_items = []
        for w in result.windows:
            w_score = float(w.model_score) if math.isfinite(w.model_score) else 0.0
            window_items.append(
                ReportWindowItem(
                    window_index=int(w.window.window_index),
                    start_seconds=round(float(w.window.start_time_seconds), 3),
                    end_seconds=round(float(w.window.end_time_seconds), 3),
                    score=round(w_score, 4),
                    prediction=str(w.decision.prediction.value),
                    margin=round(float(w.decision.margin_to_threshold), 4),
                    is_padded=bool(w.window.is_padded),
                )
            )

        mean_s = float(result.mean_score) if math.isfinite(result.mean_score) else 0.0
        min_s = float(result.min_score) if math.isfinite(result.min_score) else 0.0
        max_s = float(result.max_score) if math.isfinite(result.max_score) else 0.0

        res_summary = ReportResultSummary(
            analysis_mode="multi_window_timeline",
            overall_screening_finding=finding,
            primary_score=round(mean_s, 4),
            decision_label=None,
            total_windows=tot_windows,
            likely_genuine_windows=real_count,
            potential_fake_windows=fake_count,
            flagged_window_percentage=flagged_pct,
            min_window_score=round(min_s, 4),
            max_window_score=round(max_s, 4),
            mean_window_score=round(mean_s, 4),
            windows=window_items,
            total_processing_time_ms=round(float(result.total_processing_time_ms), 1),
            warnings=list(result.system_warnings),
        )

    limitations = [
        "Screening Nature: The V7 system is an automated machine-learning screening tool designed for research evaluation. Its output does not constitute definitive legal or forensic certification.",
        "Candidate Decision Threshold: The decision threshold (tau* = 0.4820) was empirically derived on the ASVspoof 2019 Logical Access evaluation benchmark. It has not been independently calibrated on noisy, compressed in-the-wild audio.",
        "Cosine Score Interpretation: Model scores represent continuous One-Class Softmax cosine similarity S(x) ∈ [-1.0, +1.0] relative to the bona fide speech centroid. They are geometric distances, not calibrated posterior probabilities.",
        "Overlapping Windows: Because 4.0-second sliding windows overlap by 75% (1.0s hop), the percentage of flagged windows reflects temporal window distribution, not a percentage of unique audio duration.",
        "Descriptive Visualizations: Acoustic amplitude and spectral plots illustrate energy distributions over time and frequency. They do not independently prove or disprove voice cloning or synthetic fabrication.",
    ]

    return ReportPayload(
        metadata=metadata,
        media=media_info,
        config=analysis_config,
        result=res_summary,
        limitations=limitations,
    )
