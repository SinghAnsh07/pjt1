"""
Unified Media Ingestion Service for the V7 Audio Deepfake System.

Coordinates audio loading, video audio extraction, defensive validation,
and waveform preprocessing into a single, cohesive ingestion pipeline.
"""

import os
from typing import Optional, ContextManager
from contextlib import contextmanager

from core.schemas import PreparedMedia, LoadedAudio
from core.audio_loader import load_audio
from core.video_processor import extract_audio_from_video, cleanup_temp_dir
from core.preprocessor import preprocess_waveform
from core.config import (
    SUPPORTED_VIDEO_EXTENSIONS,
    NATIVE_AUDIO_EXTENSIONS,
    FFMPEG_REQUIRED_AUDIO_EXTENSIONS,
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from core.exceptions import (
    AudioProcessingError,
    UnsupportedAudioFormatError,
)


def prepare_media(
    file_path: str,
    media_type: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    channel_layout: Optional[str] = None,
) -> PreparedMedia:
    """
    Ingests and standardizes any supported audio or video file for deepfake inference.

    Args:
        file_path: Filesystem path to audio or video file
        media_type: Optional override ('audio' or 'video')
        ffmpeg_path: Optional path to FFmpeg binary
        channel_layout: Optional channel layout override ('frames_first', 'channels_first')

    Returns:
        PreparedMedia: Unified container with normalized waveform and provenance metadata

    Raises:
        AudioLoadError / VideoProcessingError / PreprocessingError on failure
    """
    if not os.path.exists(file_path):
        raise AudioProcessingError(f"Media file does not exist: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    is_video = (media_type == "video") or (ext in SUPPORTED_VIDEO_EXTENSIONS)

    temp_dir_to_clean: Optional[str] = None
    target_audio_path = file_path
    video_warnings = []

    # 1. Video Demuxing Path (if applicable)
    if is_video:
        extraction = extract_audio_from_video(file_path, ffmpeg_path=ffmpeg_path)
        target_audio_path = extraction.extracted_audio_path
        temp_dir_to_clean = extraction.temp_dir
        video_warnings.extend(extraction.warnings)

    try:
        # 2. Audio Decoding via libsndfile
        loaded: LoadedAudio = load_audio(target_audio_path)

        # 3. Waveform Conditioning (Downmix, Polyphase Resample, Length Normalization)
        # Note: For long recordings, preprocess_waveform single-window truncates to 64,000 samples.
        tensor, prep_result = preprocess_waveform(
            loaded.waveform,
            sample_rate=loaded.sample_rate,
            target_sample_rate=TARGET_SAMPLE_RATE,
            target_samples=WINDOW_SAMPLES,
            channel_layout=channel_layout,
        )

        all_warnings = video_warnings + loaded.warnings + prep_result.warnings

        return PreparedMedia(
            waveform=tensor,
            original_filename=os.path.basename(file_path),
            media_type="video" if is_video else "audio",
            original_sample_rate=loaded.sample_rate,
            original_channels=loaded.channels,
            original_duration_seconds=loaded.duration_seconds,
            analyzed_duration_seconds=prep_result.analyzed_duration_seconds,
            target_sample_rate=TARGET_SAMPLE_RATE,
            target_samples=WINDOW_SAMPLES,
            is_video=is_video,
            decoder_info=f"{loaded.decoder} ({loaded.format_name}/{loaded.subtype})",
            warnings=all_warnings,
            temp_dir_to_clean=temp_dir_to_clean,
        )

    except Exception:
        # Guarantee cleanup on decoding or preprocessing failure
        if temp_dir_to_clean:
            cleanup_temp_dir(temp_dir_to_clean)
        raise


def cleanup_prepared_media(prepared: PreparedMedia) -> None:
    """Releases and removes any temporary extraction artifacts associated with prepared media."""
    if prepared.temp_dir_to_clean:
        cleanup_temp_dir(prepared.temp_dir_to_clean)
        prepared.temp_dir_to_clean = None


@contextmanager
def media_context(file_path: str, **kwargs):
    """Context manager ensuring temporary files are safely deleted upon exit."""
    prepared = prepare_media(file_path, **kwargs)
    try:
        yield prepared
    finally:
        cleanup_prepared_media(prepared)
