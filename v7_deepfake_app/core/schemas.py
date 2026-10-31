"""
Typed data structures and schemas for the V7 Audio Deepfake Detection System.

All schemas are standard Python dataclasses for lightweight serialization,
type safety, and clean separation between backend inference and frontend UI.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Tuple, Any, Dict


class ScoreDirection(str, Enum):
    """Directionality semantics of the OC-Softmax cosine similarity score."""
    HIGHER_IS_REAL = "HIGHER_IS_REAL"  # S(x) >= threshold -> Real; S(x) < threshold -> Fake


class DecisionLabel(str, Enum):
    """Binary biometric decision classification."""
    REAL = "REAL"  # Bona fide authentic speech
    FAKE = "FAKE"  # Synthetic deepfake / voice conversion / spoofed speech


class ThresholdStatus(str, Enum):
    """Validation and calibration status of the operational decision threshold."""
    CANDIDATE_UNVALIDATED = "CANDIDATE_UNVALIDATED"  # EER operating point on ASVspoof 2019 LA; uncalibrated on out-of-domain audio
    OFFICIAL_BENCHMARK_EER = "OFFICIAL_BENCHMARK_EER"  # Exactly tau* = 0.4820 from official research evaluation
    CUSTOM_USER_CONFIGURED = "CUSTOM_USER_CONFIGURED"  # Manually supplied operating threshold


@dataclass(frozen=True)
class ModelMetadata:
    """Verified architectural and cryptographic metadata of the loaded model checkpoint."""
    checkpoint_path: str
    checkpoint_sha256: str
    saved_epoch: Optional[int]
    saved_dev_eer: Optional[float]
    model_class: str
    device: str
    total_parameters: int
    trainable_parameters: int
    buffer_elements: int
    strict_load_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessingResult:
    """Detailed record of waveform conditioning, resampling, and windowing."""
    original_sample_rate: int
    target_sample_rate: int
    original_samples: int
    target_samples: int
    original_duration_seconds: float
    analyzed_duration_seconds: float
    is_multichannel: bool
    is_resampled: bool
    is_padded: bool
    is_truncated: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionResult:
    """Structured decision output produced by the conservative decision engine."""
    prediction: DecisionLabel
    raw_score: float
    threshold: float
    threshold_status: ThresholdStatus
    threshold_source: str
    score_direction: ScoreDirection
    margin_to_threshold: float
    explanation: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res['prediction'] = self.prediction.value
        res['threshold_status'] = self.threshold_status.value
        res['score_direction'] = self.score_direction.value
        return res


@dataclass
class TimingBreakdown:
    """Wall-clock timing measurements across inference pipeline stages in milliseconds."""
    preprocessing_ms: float
    feature_extraction_ms: float
    neural_forward_ms: float
    decision_ms: float
    total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    """Complete, end-to-end structured inference response."""
    status: str  # "SUCCESS" or "ERROR"
    filename: Optional[str]
    model_score: float
    decision: DecisionResult
    preprocessing: PreprocessingResult
    model_metadata: ModelMetadata
    timing: TimingBreakdown
    feature_tensor_shape: Tuple[int, ...]
    embedding_shape: Tuple[int, ...]
    system_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "filename": self.filename,
            "model_score": self.model_score,
            "decision": self.decision.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "model_metadata": self.model_metadata.to_dict(),
            "timing": self.timing.to_dict(),
            "feature_tensor_shape": list(self.feature_tensor_shape),
            "embedding_shape": list(self.embedding_shape),
            "system_warnings": self.system_warnings,
        }


# ============================================================================
# Phase 3 Media Ingestion, Decoding, and Windowing Schemas
# ============================================================================

@dataclass
class LoadedAudio:
    """Decoded in-memory audio signal and source container metadata."""
    waveform: Any  # np.ndarray shaped (frames,) or (frames, channels)
    sample_rate: int
    channels: int
    duration_seconds: float
    frames: int
    format_name: str
    subtype: str
    decoder: str
    warnings: List[str] = field(default_factory=list)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Returns metadata dictionary without embedding the large raw waveform array."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": round(self.duration_seconds, 4),
            "frames": self.frames,
            "format_name": self.format_name,
            "subtype": self.subtype,
            "decoder": self.decoder,
            "warnings": self.warnings,
        }


@dataclass
class VideoExtractionResult:
    """Diagnostic outcome of demuxing an audio track from a video container."""
    extracted_audio_path: str
    video_duration_seconds: float
    has_audio_stream: bool
    temp_dir: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedMedia:
    """Fully conditioned media container ready for single or multi-window inference."""
    waveform: Any  # torch.Tensor [1, 64000] for single-window or 1D array for windowing
    original_filename: str
    media_type: str  # "audio" or "video"
    original_sample_rate: int
    original_channels: int
    original_duration_seconds: float
    analyzed_duration_seconds: float
    target_sample_rate: int
    target_samples: int
    is_video: bool
    decoder_info: str
    warnings: List[str] = field(default_factory=list)
    temp_dir_to_clean: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_filename": self.original_filename,
            "media_type": self.media_type,
            "original_sample_rate": self.original_sample_rate,
            "original_channels": self.original_channels,
            "original_duration_seconds": round(self.original_duration_seconds, 4),
            "analyzed_duration_seconds": round(self.analyzed_duration_seconds, 4),
            "target_sample_rate": self.target_sample_rate,
            "target_samples": self.target_samples,
            "is_video": self.is_video,
            "decoder_info": self.decoder_info,
            "warnings": self.warnings,
        }


@dataclass
class WindowMetadata:
    """Positional temporal boundaries and padding state of an individual window."""
    window_index: int
    start_time_seconds: float
    end_time_seconds: float
    start_sample: int
    end_sample: int
    is_padded: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WindowInferenceResult:
    """Inference diagnostic result for a single temporal window."""
    window: WindowMetadata
    model_score: float
    decision: DecisionResult
    timing_ms: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window.to_dict(),
            "model_score": round(self.model_score, 4),
            "decision": self.decision.to_dict(),
            "timing_ms": round(self.timing_ms, 2),
            "warnings": self.warnings,
        }


@dataclass
class LongRecordingInferenceResult:
    """Comprehensive multi-window timeline diagnostic result for extended recordings."""
    filename: Optional[str]
    total_windows: int
    total_audio_duration_seconds: float
    covered_duration_seconds: float
    unanalyzed_duration_seconds: float
    summary_real_windows: int
    summary_fake_windows: int
    min_score: float
    max_score: float
    mean_score: float
    windows: List[WindowInferenceResult]
    total_processing_time_ms: float
    system_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "total_windows": self.total_windows,
            "total_audio_duration_seconds": round(self.total_audio_duration_seconds, 4),
            "covered_duration_seconds": round(self.covered_duration_seconds, 4),
            "unanalyzed_duration_seconds": round(self.unanalyzed_duration_seconds, 4),
            "summary_real_windows": self.summary_real_windows,
            "summary_fake_windows": self.summary_fake_windows,
            "min_score": round(self.min_score, 4),
            "max_score": round(self.max_score, 4),
            "mean_score": round(self.mean_score, 4),
            "windows": [w.to_dict() for w in self.windows],
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "system_warnings": self.system_warnings,
        }

