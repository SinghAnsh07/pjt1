"""
Custom exceptions for the V7 Audio Deepfake Detection System.

All exceptions inherit from DeepfakeAppError to allow granular error handling
without broad or silent failure modes.
"""

class DeepfakeAppError(Exception):
    """Base exception for all errors in the V7 deepfake detection application."""
    pass


class CheckpointError(DeepfakeAppError):
    """Base exception for checkpoint-related errors."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when the specified model checkpoint file does not exist."""
    pass


class CheckpointIncompatibleError(CheckpointError):
    """Raised when the checkpoint state dictionary fails strict architecture matching."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when the checkpoint file is corrupted or cannot be safely deserialized."""
    pass


class AudioProcessingError(DeepfakeAppError):
    """Base exception for audio processing and validation errors."""
    pass


class InvalidWaveformError(AudioProcessingError):
    """Raised when the input audio waveform is empty, malformed, non-finite (NaN/Inf), or improperly dimensioned."""
    pass


class InvalidSampleRateError(AudioProcessingError):
    """Raised when the input audio sample rate is invalid (non-positive, zero, or unparsable)."""
    pass


class PreprocessingError(AudioProcessingError):
    """Raised when audio conversion, downmixing, resampling, or length adjustment fails."""
    pass


class FeatureExtractionError(DeepfakeAppError):
    """Raised when the 2D FrFT-Wavelet feature extraction pipeline fails."""
    pass


class InferenceError(DeepfakeAppError):
    """Raised when the neural network forward pass fails or produces non-finite outputs."""
    pass


class ThresholdError(DeepfakeAppError):
    """Base exception for decision threshold configuration errors."""
    pass


class ThresholdConfigurationError(ThresholdError):
    """Raised when a decision threshold is invalid or out of the mathematical score bounds [-1.0, 1.0]."""
    pass


# ============================================================================
# Phase 3 Media Ingestion & Decoding Exceptions
# ============================================================================

class AudioLoadError(AudioProcessingError):
    """Base exception for audio file loading and decoding failures."""
    pass


class UnsupportedAudioFormatError(AudioLoadError):
    """Raised when the requested audio format or codec is not supported by available decoders."""
    pass


class CorruptAudioFileError(AudioLoadError):
    """Raised when an audio file has invalid headers, unreadable frames, or corrupted data."""
    pass


class AudioFileTooLargeError(AudioLoadError):
    """Raised when an uploaded file exceeds the configured maximum byte limit."""
    pass


class AudioTooLongError(AudioLoadError):
    """Raised when media exceeds the maximum configured duration limit."""
    pass


class EmptyAudioFileError(AudioLoadError):
    """Raised when decoded media contains 0 audio frames."""
    pass


class VideoProcessingError(DeepfakeAppError):
    """Base exception for video demuxing and audio extraction failures."""
    pass


class FFmpegNotFoundError(VideoProcessingError):
    """Raised when an operation requires FFmpeg but the executable cannot be located."""
    pass


class FFmpegExecutionError(VideoProcessingError):
    """Raised when FFmpeg process returns a non-zero exit code or times out."""
    pass


class NoAudioStreamError(VideoProcessingError):
    """Raised when a video container has no valid audio stream to extract."""
    pass


class WindowingError(DeepfakeAppError):
    """Raised when sliding window segmentation encounters invalid indexing or configuration."""
    pass

