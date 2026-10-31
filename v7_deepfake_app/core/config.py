"""
Central configuration and operational safeguards for the V7 Audio Deepfake System.

Defines resource limits, timeout thresholds, windowing parameters, and supported
media container extensions.
"""

from typing import Set

# ============================================================================
# Media Resource & Safety Limits
# ============================================================================

# Maximum allowed file size for uploaded media (100 MB)
MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024

# Maximum allowed audio duration for processing (10 minutes)
MAX_AUDIO_DURATION_SECONDS: float = 600.0

# Maximum allowed video duration for audio extraction (10 minutes)
MAX_VIDEO_DURATION_SECONDS: float = 600.0

# Maximum number of analysis windows allowed for long recordings (e.g. 120 windows)
MAX_WINDOWS_COUNT: int = 120

# Subprocess execution timeout for FFmpeg operations in seconds
FFMPEG_TIMEOUT_SECONDS: float = 60.0

# Minimum audio duration threshold for meaningful analysis (0.1 seconds)
MIN_AUDIO_DURATION_SECONDS: float = 0.1

# ============================================================================
# Acoustic & Model Windowing Specifications
# ============================================================================

# Target model sampling frequency
TARGET_SAMPLE_RATE: int = 16000

# Fixed model temporal window length in seconds
WINDOW_DURATION_SECONDS: float = 4.0

# Fixed model temporal window length in samples (4.0s @ 16 kHz)
WINDOW_SAMPLES: int = 64000

# Default sliding window hop length in seconds
DEFAULT_HOP_SECONDS: float = 1.0

# Default sliding window hop length in samples (1.0s @ 16 kHz)
DEFAULT_HOP_SAMPLES: int = 16000

# ============================================================================
# Media Format Support Classification
# ============================================================================

# Formats supported for direct audio ingestion (soundfile + PyAV / Media Foundation)
NATIVE_AUDIO_EXTENSIONS: Set[str] = {
    ".wav",
    ".flac",
    ".ogg",
    ".mp3",
    ".aiff",
    ".aif",
    ".m4a",
    ".aac",
}

# Formats requiring external FFmpeg for audio decoding
FFMPEG_REQUIRED_AUDIO_EXTENSIONS: Set[str] = {
    ".wma",
    ".opus",
}

# Video containers requiring FFmpeg for audio stream demuxing
SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}
