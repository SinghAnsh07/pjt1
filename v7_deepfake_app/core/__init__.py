"""
V7 Audio Deepfake Detection Core Package.
"""

from core.model import (
    EndToEndLCNN2D,
    LCNN2D,
    FrFTWavelet2DExtractor,
    OCSoftmax,
)
from core.checkpoint_loader import load_model
from core.preprocessor import preprocess_waveform
from core.feature_extractor import FeatureExtractor
from core.decision_engine import DecisionEngine
from core.inference import V7InferenceEngine, predict_waveform
from core.audio_loader import load_audio
from core.video_processor import extract_audio_from_video, discover_ffmpeg
from core.media_ingestion import prepare_media, cleanup_prepared_media, media_context
from core.windowing import generate_windows, analyze_long_recording
from core.schemas import (
    InferenceResult,
    DecisionResult,
    ModelMetadata,
    PreprocessingResult,
    DecisionLabel,
    ScoreDirection,
    ThresholdStatus,
    LoadedAudio,
    VideoExtractionResult,
    PreparedMedia,
    WindowMetadata,
    WindowInferenceResult,
    LongRecordingInferenceResult,
)
from core.exceptions import DeepfakeAppError

__all__ = [
    "EndToEndLCNN2D",
    "LCNN2D",
    "FrFTWavelet2DExtractor",
    "OCSoftmax",
    "load_model",
    "preprocess_waveform",
    "FeatureExtractor",
    "DecisionEngine",
    "V7InferenceEngine",
    "predict_waveform",
    "load_audio",
    "extract_audio_from_video",
    "discover_ffmpeg",
    "prepare_media",
    "cleanup_prepared_media",
    "media_context",
    "generate_windows",
    "analyze_long_recording",
    "InferenceResult",
    "DecisionResult",
    "ModelMetadata",
    "PreprocessingResult",
    "DecisionLabel",
    "ScoreDirection",
    "ThresholdStatus",
    "LoadedAudio",
    "VideoExtractionResult",
    "PreparedMedia",
    "WindowMetadata",
    "WindowInferenceResult",
    "LongRecordingInferenceResult",
    "DeepfakeAppError",
]
