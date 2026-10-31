"""
Video-to-Audio Processing Module for the V7 Audio Deepfake System.

Provides secure audio extraction from video containers (MP4, MOV, MKV, AVI, WEBM)
using external FFmpeg when available. Manages temporary file lifecycle, enforces
process timeouts, and handles missing dependencies gracefully.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Tuple, Optional, List

from core.schemas import VideoExtractionResult
from core.config import (
    MAX_FILE_SIZE_BYTES,
    MAX_VIDEO_DURATION_SECONDS,
    FFMPEG_TIMEOUT_SECONDS,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from core.exceptions import (
    VideoProcessingError,
    FFmpegNotFoundError,
    FFmpegExecutionError,
    NoAudioStreamError,
    AudioFileTooLargeError,
)


def discover_ffmpeg(custom_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Reliably checks for callable FFmpeg executable.

    Checks:
      1. Explicitly configured path, if provided
      2. System PATH resolution via shutil.which('ffmpeg')
      3. Callable verification via 'ffmpeg -version' execution

    Returns:
        Tuple of (is_available: bool, resolved_path: Optional[str])
    """
    candidates = []
    if custom_path:
        candidates.append(custom_path)

    which_path = shutil.which("ffmpeg")
    if which_path:
        candidates.append(which_path)

    for path in candidates:
        if not os.path.exists(path) and not which_path:
            continue
        try:
            # Controlled non-shell execution
            res = subprocess.run(
                [path, "-version"],
                capture_output=True,
                text=True,
                timeout=5.0,
                shell=False,
            )
            if res.returncode == 0:
                return True, path
        except (subprocess.SubprocessError, OSError):
            continue

    return False, None


def extract_audio_from_video(
    video_path: str,
    ffmpeg_path: Optional[str] = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
    timeout_seconds: float = FFMPEG_TIMEOUT_SECONDS,
) -> VideoExtractionResult:
    """
    Extracts the audio track from a video file into an uncompressed 16 kHz mono WAV file.

    Args:
        video_path: Absolute or relative filesystem path to the video file
        ffmpeg_path: Optional path to FFmpeg binary override
        max_size_bytes: Configured maximum video file size
        timeout_seconds: Subprocess execution timeout

    Returns:
        VideoExtractionResult: Contains path to temporary WAV file and temp directory

    Raises:
        VideoProcessingError: If video file does not exist or has invalid extension
        FFmpegNotFoundError: If FFmpeg is not installed or available on PATH
        AudioFileTooLargeError: If video file exceeds size limit
        NoAudioStreamError: If video container contains no audio tracks
        FFmpegExecutionError: If FFmpeg process crashes, fails, or times out
    """
    # 1. Existence and Format Validation
    if not os.path.exists(video_path):
        raise VideoProcessingError(f"Video file not found at: {video_path}")
    if not os.path.isfile(video_path):
        raise VideoProcessingError(f"Video path is not a regular file: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoProcessingError(
            f"Unsupported video container '{ext}'. Supported formats: {sorted(list(SUPPORTED_VIDEO_EXTENSIONS))}"
        )

    file_size = os.path.getsize(video_path)
    if file_size > max_size_bytes:
        raise AudioFileTooLargeError(
            f"Video file size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit "
            f"({max_size_bytes / (1024*1024):.1f} MB)"
        )

    # 2. FFmpeg Discovery
    has_ffmpeg, resolved_ffmpeg = discover_ffmpeg(ffmpeg_path)
    if not has_ffmpeg or not resolved_ffmpeg:
        raise FFmpegNotFoundError(
            "Video analysis requires FFmpeg, which is not available in this environment. "
            "You can upload an audio file directly instead."
        )

    # 3. Create Unique Temporary Directory for Safe Extraction
    temp_dir = tempfile.mkdtemp(prefix="v7_deepfake_video_")
    output_wav = os.path.join(temp_dir, "extracted_audio.wav")

    # 4. Controlled Subprocess Invocation (No shell=True to prevent injection)
    cmd = [
        resolved_ffmpeg,
        "-y",               # Overwrite output
        "-v", "error",      # Suppress verbose banners
        "-i", video_path,   # Input file
        "-vn",              # Disable video stream processing
        "-acodec", "pcm_s16le",  # Uncompressed standard 16-bit PCM WAV
        "-ar", "16000",     # Model target sampling rate
        "-ac", "1",         # Force mono downmix
        output_wav,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired as e:
        cleanup_temp_dir(temp_dir)
        raise FFmpegExecutionError(f"FFmpeg audio extraction timed out after {timeout_seconds} seconds") from e
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise FFmpegExecutionError(f"Failed to execute FFmpeg process: {str(e)}") from e

    # 5. Diagnostic Inspection of Subprocess Outcome
    if proc.returncode != 0:
        cleanup_temp_dir(temp_dir)
        stderr_text = proc.stderr.lower() if proc.stderr else ""
        if "does not contain any stream" in stderr_text or "no audio" in stderr_text:
            raise NoAudioStreamError(f"Video file '{os.path.basename(video_path)}' contains no audio stream.")
        raise FFmpegExecutionError(f"FFmpeg extraction failed (code {proc.returncode}): {proc.stderr}")

    if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
        cleanup_temp_dir(temp_dir)
        raise NoAudioStreamError(f"No audio stream could be extracted from video '{os.path.basename(video_path)}'.")

    warnings = [
        "Audio extracted from video container. Note: Video frames were not analyzed; "
        "forensic deepfake classification applies strictly to the extracted audio track."
    ]

    return VideoExtractionResult(
        extracted_audio_path=output_wav,
        video_duration_seconds=0.0,  # Will be accurately determined upon audio decoding
        has_audio_stream=True,
        temp_dir=temp_dir,
        warnings=warnings,
    )


def cleanup_temp_dir(temp_dir: Optional[str]) -> None:
    """Safely removes an extraction temporary directory and its contents."""
    if temp_dir and os.path.exists(temp_dir) and "v7_deepfake_video_" in os.path.basename(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
