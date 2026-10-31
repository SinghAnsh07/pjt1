"""
Automated unit tests for Unified Media Ingestion Service.
"""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from core.media_ingestion import prepare_media, cleanup_prepared_media, media_context
from core.schemas import PreparedMedia
from core.exceptions import AudioProcessingError, FFmpegNotFoundError


@pytest.fixture
def temp_media_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


def test_prepare_audio_media_wav(temp_media_dir):
    """Verifies that an audio WAV file is decoded and prepared into the [1, 64000] model tensor."""
    wav_path = os.path.join(temp_media_dir, "sample.wav")
    sr = 44100
    data = (np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, sr * 4)) * 0.5).astype(np.float32)
    sf.write(wav_path, data, sr)

    prepared = prepare_media(wav_path)
    assert isinstance(prepared, PreparedMedia)
    assert prepared.is_video is False
    assert prepared.target_sample_rate == 16000
    assert prepared.target_samples == 64000
    assert isinstance(prepared.waveform, torch.Tensor)
    assert prepared.waveform.shape == (1, 64000)
    assert prepared.original_sample_rate == 44100
    assert abs(prepared.original_duration_seconds - 4.0) < 1e-4


def test_prepare_video_media_missing_ffmpeg(temp_media_dir):
    """Verifies that uploading a video without FFmpeg raises FFmpegNotFoundError."""
    video_path = os.path.join(temp_media_dir, "sample.mp4")
    with open(video_path, "wb") as f:
        f.write(b"fake video data")

    with pytest.raises(FFmpegNotFoundError):
        prepare_media(video_path)


def test_media_context_manager(temp_media_dir):
    """Verifies that media_context cleanly produces prepared media and handles cleanup."""
    wav_path = os.path.join(temp_media_dir, "context_test.wav")
    sr = 16000
    data = (np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, sr)) * 0.5).astype(np.float32)
    sf.write(wav_path, data, sr)

    with media_context(wav_path) as prep:
        assert prep.waveform.shape == (1, 64000)
        assert prep.is_video is False


def test_nonexistent_media_file_rejection():
    """Verifies that missing media files raise AudioProcessingError."""
    with pytest.raises(AudioProcessingError):
        prepare_media("C:/nonexistent_media_path_xyz.mp3")
