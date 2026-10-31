"""
Automated unit tests for Long-Recording Window Generation and Temporal Inference.
"""

import os
import pytest
import numpy as np

from core.windowing import generate_windows, analyze_long_recording
from core.schemas import LongRecordingInferenceResult, WindowMetadata
from core.inference import V7InferenceEngine

CURRENT_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CKPT_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))


@pytest.fixture(scope="module")
def engine():
    return V7InferenceEngine(checkpoint_path=CKPT_PATH, device="cpu")


def test_generate_windows_exact_four_seconds():
    """Verifies that an exact 4.0s recording (64,000 samples) generates exactly 1 unpadded window."""
    windows, warnings = generate_windows(total_samples=64000, sample_rate=16000)
    assert len(windows) == 1
    w = windows[0]
    assert w.window_index == 0
    assert w.start_sample == 0
    assert w.end_sample == 64000
    assert w.start_time_seconds == 0.0
    assert w.end_time_seconds == 4.0
    assert w.is_padded is False


def test_generate_windows_short_recording():
    """Verifies that short audio (< 4.0s) generates a single window marked is_padded=True."""
    windows, warnings = generate_windows(total_samples=32000, sample_rate=16000)
    assert len(windows) == 1
    w = windows[0]
    assert w.is_padded is True
    assert w.end_time_seconds == 2.0
    assert any("shorter than model window" in warn for warn in warnings)


def test_generate_windows_ten_seconds():
    """Verifies sliding window count and timestamps for a 10.0s recording with 1.0s hop."""
    # 10.0 seconds at 16 kHz = 160,000 samples
    # Windows: [0, 64k], [16k, 80k], [32k, 96k], [48k, 112k], [64k, 128k], [80k, 144k], [96k, 160k]
    # Total = 7 windows
    windows, warnings = generate_windows(
        total_samples=160000,
        sample_rate=16000,
        window_samples=64000,
        hop_samples=16000,
    )
    assert len(windows) == 7
    assert windows[0].start_time_seconds == 0.0
    assert windows[0].end_time_seconds == 4.0
    assert windows[-1].end_time_seconds == 10.0
    assert windows[-1].end_sample == 160000


def test_generate_windows_tail_anchoring_coverage():
    """Verifies that non-aligned audio lengths (e.g. 7.5s) anchor a final window to ensure 100% coverage."""
    # 7.5 seconds = 120,000 samples
    # Regular windows: [0, 64k], [16k, 80k], [32k, 96k], [48k, 112k] (ends at 7.0s)
    # Tail window: [120k - 64k : 120k] = [56k, 120k] (ends at 7.5s)
    windows, warnings = generate_windows(
        total_samples=120000,
        sample_rate=16000,
        window_samples=64000,
        hop_samples=16000,
    )
    assert len(windows) == 5
    assert windows[-1].end_sample == 120000
    assert windows[-1].end_time_seconds == 7.5
    # Verify 100% coverage
    assert windows[-1].end_sample == 120000


def test_max_windows_safeguard():
    """Verifies that max_windows parameter stops generation before runaway looping."""
    windows, warnings = generate_windows(
        total_samples=160000,
        sample_rate=16000,
        window_samples=64000,
        hop_samples=16000,
        max_windows=3,
    )
    assert len(windows) == 3
    assert any("maximum configured limit" in warn for warn in warnings)


def test_analyze_long_recording_end_to_end(engine):
    """Executes multi-window deepfake analysis on an 8.0s synthetic signal with the real V7 engine."""
    # 8.0s audio at 16 kHz = 128,000 samples
    t = np.linspace(0, 8.0, 128000, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    result = analyze_long_recording(
        audio,
        engine=engine,
        hop_seconds=1.0,
        filename="long_recording_8s.wav",
    )

    assert isinstance(result, LongRecordingInferenceResult)
    assert result.filename == "long_recording_8s.wav"
    assert result.total_audio_duration_seconds == 8.0
    assert result.covered_duration_seconds == 8.0
    assert result.total_windows == 5  # [0, 4s], [1, 5s], [2, 6s], [3, 7s], [4, 8s]
    assert len(result.windows) == 5
    for w_res in result.windows:
        assert np.isfinite(w_res.model_score)
        assert -1.0 <= w_res.model_score <= 1.0
        assert w_res.decision.prediction is not None
        assert w_res.timing_ms > 0

    assert -1.0 <= result.min_score <= 1.0
    assert -1.0 <= result.max_score <= 1.0
    assert -1.0 <= result.mean_score <= 1.0
    assert result.total_processing_time_ms > 0
    assert (result.summary_real_windows + result.summary_fake_windows) == result.total_windows
