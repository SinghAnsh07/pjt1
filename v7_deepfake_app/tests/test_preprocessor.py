"""
Automated unit tests for Waveform Preprocessing, Resampling, and Windowing.
"""

import math
import numpy as np
import pytest
import torch

from core.preprocessor import preprocess_waveform
from core.exceptions import (
    InvalidWaveformError,
    InvalidSampleRateError,
    PreprocessingError,
)


def test_standard_four_second_16k():
    """Verifies that a standard 4.0s 16 kHz waveform produces exact [1, 64000] tensor with no pad/trunc."""
    audio = np.random.randn(64000).astype(np.float32)
    tensor, prep = preprocess_waveform(audio, 16000)

    assert tensor.shape == (1, 64000)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert prep.is_resampled is False
    assert prep.is_padded is False
    assert prep.is_truncated is False
    assert prep.is_multichannel is False
    assert prep.original_duration_seconds == 4.0
    assert prep.analyzed_duration_seconds == 4.0


def test_resampling_44100():
    """Verifies that 44.1 kHz audio resamples to exactly 64,000 samples for a 4.0s file."""
    audio = np.random.randn(44100 * 4).astype(np.float32)
    tensor, prep = preprocess_waveform(audio, 44100)

    assert tensor.shape == (1, 64000)
    assert prep.is_resampled is True
    assert prep.original_sample_rate == 44100
    assert prep.target_sample_rate == 16000


def test_resampling_48000():
    """Verifies that 48 kHz audio resamples to exactly 64,000 samples for a 4.0s file."""
    audio = np.random.randn(48000 * 4).astype(np.float32)
    tensor, prep = preprocess_waveform(audio, 48000)

    assert tensor.shape == (1, 64000)
    assert prep.is_resampled is True


def test_stereo_canonical_frames_first_downmix():
    """Verifies that canonical decoder format (frames, channels) averages across axis=1 to frames."""
    frames = 64000
    ch_left = np.ones(frames, dtype=np.float32) * 0.2
    ch_right = np.ones(frames, dtype=np.float32) * 0.8
    stereo_frames_first = np.column_stack([ch_left, ch_right])  # Shape: (64000, 2)

    tensor, prep = preprocess_waveform(stereo_frames_first, 16000)
    assert tensor.shape == (1, 64000)
    assert prep.is_multichannel is True
    # Verify exact channel average: (0.2 + 0.8) / 2 = 0.5
    assert np.allclose(tensor.numpy(), 0.5, atol=1e-5)
    # Ensure it did NOT collapse to 2 samples!
    assert tensor.shape[-1] == 64000


def test_stereo_channels_first_downmix():
    """Verifies that PyTorch/Librosa format (channels, frames) averages across axis=0 to frames."""
    frames = 64000
    ch_left = np.ones(frames, dtype=np.float32) * 0.3
    ch_right = np.ones(frames, dtype=np.float32) * 0.9
    stereo_channels_first = np.row_stack([ch_left, ch_right])  # Shape: (2, 64000)

    tensor, prep = preprocess_waveform(stereo_channels_first, 16000)
    assert tensor.shape == (1, 64000)
    assert prep.is_multichannel is True
    assert np.allclose(tensor.numpy(), 0.6, atol=1e-5)


def test_multichannel_surround_5ch_downmix():
    """Verifies that 5-channel surround audio averages across channels accurately."""
    frames = 64000
    channels = 5
    # Create 5 distinct channel levels: 0.1, 0.2, 0.3, 0.4, 0.5 -> Mean = 0.3
    multi = np.stack([np.ones(frames, dtype=np.float32) * (0.1 * i) for i in range(1, channels + 1)], axis=1)

    tensor, prep = preprocess_waveform(multi, 16000, channel_layout="frames_first")
    assert tensor.shape == (1, 64000)
    assert prep.is_multichannel is True
    assert np.allclose(tensor.numpy(), 0.3, atol=1e-5)


def test_ambiguous_2d_dimensions_rejection():
    """Verifies that ambiguous square matrices (e.g. 4x4) raise InvalidWaveformError without explicit layout."""
    ambiguous = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(InvalidWaveformError):
        preprocess_waveform(ambiguous, 16000)

    # With explicit layout, it must succeed
    tensor, prep = preprocess_waveform(ambiguous, 16000, channel_layout="frames_first")
    assert prep.is_multichannel is True


def test_stereo_downmix_with_resampling():
    """Verifies that 44.1 kHz stereo audio is downmixed to mono first, then resampled to 16 kHz."""
    frames_44k = 44100 * 4
    ch1 = np.ones(frames_44k, dtype=np.float32) * 0.4
    ch2 = np.ones(frames_44k, dtype=np.float32) * 0.6
    stereo = np.column_stack([ch1, ch2])  # Shape: (176400, 2)

    tensor, prep = preprocess_waveform(stereo, 44100)
    assert tensor.shape == (1, 64000)
    assert prep.is_multichannel is True
    assert prep.is_resampled is True
    # Verify interior samples after FIR filter transient boundary
    assert np.allclose(tensor.numpy()[:, 100:-100], 0.5, atol=1e-4)


def test_short_audio_wrap_padding():
    """Verifies that short audio (< 4.0s) uses circular wrap padding."""
    # 1.0 second audio of ramp values
    short_audio = np.arange(16000, dtype=np.float32)
    tensor, prep = preprocess_waveform(short_audio, 16000)

    assert tensor.shape == (1, 64000)
    assert prep.is_padded is True
    assert prep.is_truncated is False
    # Check wrap padding: sample at index 16000 should equal sample at index 0
    t_np = tensor.numpy().squeeze(0)
    assert t_np[16000] == t_np[0]
    assert t_np[32000] == t_np[0]
    assert t_np[48000] == t_np[0]


def test_long_audio_single_window_truncation():
    """Verifies that long audio (> 4.0s) is truncated to the first 64,000 samples with warning."""
    long_audio = np.random.randn(16000 * 8).astype(np.float32)  # 8.0 seconds
    tensor, prep = preprocess_waveform(long_audio, 16000)

    assert tensor.shape == (1, 64000)
    assert prep.is_truncated is True
    assert prep.is_padded is False
    assert prep.original_duration_seconds == 8.0
    assert prep.analyzed_duration_seconds == 4.0
    assert any("truncated to initial 4.0s" in w for w in prep.warnings)


def test_empty_waveform_rejection():
    """Verifies that empty waveforms raise InvalidWaveformError."""
    with pytest.raises(InvalidWaveformError):
        preprocess_waveform(np.array([], dtype=np.float32), 16000)


def test_nan_and_inf_waveform_rejection():
    """Verifies that NaN and Inf values in waveform raise InvalidWaveformError."""
    nan_audio = np.ones(64000, dtype=np.float32)
    nan_audio[100] = np.nan
    with pytest.raises(InvalidWaveformError):
        preprocess_waveform(nan_audio, 16000)

    inf_audio = np.ones(64000, dtype=np.float32)
    inf_audio[100] = np.inf
    with pytest.raises(InvalidWaveformError):
        preprocess_waveform(inf_audio, 16000)


def test_invalid_sample_rate_rejection():
    """Verifies that invalid sample rates raise InvalidSampleRateError."""
    audio = np.zeros(64000, dtype=np.float32)
    with pytest.raises(InvalidSampleRateError):
        preprocess_waveform(audio, 0)
    with pytest.raises(InvalidSampleRateError):
        preprocess_waveform(audio, -16000)


def test_invalid_high_dimensional_waveform():
    """Verifies that >2D waveforms raise InvalidWaveformError."""
    audio_3d = np.zeros((2, 2, 64000), dtype=np.float32)
    with pytest.raises(InvalidWaveformError):
        preprocess_waveform(audio_3d, 16000)
