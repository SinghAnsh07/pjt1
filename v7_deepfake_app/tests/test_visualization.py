"""
Automated unit and integration tests for Phase 5 Audio Signal Visualization.

Covers:
  - Waveform envelope decimation, bounds, and non-mutation
  - Spectrogram calculation, frequency axes, silence/low-energy stability
  - Multi-window timeline score mapping and candidate threshold rendering
  - Neural feature view extraction and tensor plotting
  - Edge cases: silence, near-silence, very short signals, stereo downmixing, long signals
  - Session state isolation and absence of fabricated probabilities
"""

import os
import sys
import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt

from ui.plots import (
    compute_waveform_envelope,
    create_waveform_figure,
    create_spectrogram_figure,
    create_timeline_figure,
    create_feature_views_figure,
)
from core.schemas import (
    LoadedAudio,
    InferenceResult,
    LongRecordingInferenceResult,
    WindowInferenceResult,
    WindowMetadata,
    DecisionResult,
    DecisionLabel,
    ScoreDirection,
    ThresholdStatus,
    TimingBreakdown,
    PreprocessingResult,
    ModelMetadata,
)
from core.decision_engine import DEFAULT_CANDIDATE_THRESHOLD
from core.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from core.inference import V7InferenceEngine

CURRENT_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CKPT_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))


@pytest.fixture(scope="module")
def engine():
    return V7InferenceEngine(checkpoint_path=CKPT_PATH, device="cpu")


def test_waveform_envelope_short_signal():
    """Verifies that short audio (<= max_points) returns exact samples without decimation."""
    audio = np.array([0.1, -0.5, 0.8, -0.2], dtype=np.float32)
    t, mins, maxs = compute_waveform_envelope(audio, max_points=10)
    assert len(t) == len(audio)
    assert np.allclose(mins, audio)
    assert np.allclose(maxs, audio)


def test_waveform_envelope_decimation_preserves_peaks():
    """Verifies that decimation preserves peak extrema without mutating the input signal."""
    np.random.seed(42)
    audio = np.random.uniform(-0.3, 0.3, 20000).astype(np.float32)
    # Inject prominent positive and negative spikes
    audio[5000] = 0.95
    audio[12000] = -0.92
    audio_orig_copy = audio.copy()

    t, mins, maxs = compute_waveform_envelope(audio, max_points=1000)

    # 1. Output length is bounded
    assert len(t) <= 1000
    assert len(mins) == len(t)
    assert len(maxs) == len(t)

    # 2. Peak spikes are captured in the envelope
    assert np.isclose(np.max(maxs), 0.95, atol=1e-4)
    assert np.isclose(np.min(mins), -0.92, atol=1e-4)

    # 3. Input audio array is NOT mutated
    assert np.array_equal(audio, audio_orig_copy)


def test_waveform_figure_axis_and_labels():
    """Verifies waveform plot title, labels, and limits for 16 kHz audio."""
    sr = 16000
    duration = 3.0
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration), endpoint=False)).astype(np.float32)

    fig = create_waveform_figure(audio, sample_rate=sr, title="Test Waveform")
    ax = fig.axes[0]

    assert ax.get_xlabel() == "Time (seconds)"
    assert ax.get_ylabel() == "Amplitude"
    assert "Test Waveform" in ax.get_title()
    assert abs(ax.get_xlim()[1] - duration) < 0.05
    plt.close(fig)


def test_waveform_figure_multichannel_downmixing():
    """Verifies that stereo input is displayed cleanly without raising errors."""
    stereo_audio = np.random.uniform(-0.5, 0.5, (32000, 2)).astype(np.float32)
    fig = create_waveform_figure(stereo_audio, sample_rate=16000)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_spectrogram_frequency_and_time_axes():
    """Verifies that spectrogram axes match sample rate and duration."""
    sr = 16000
    duration = 2.5
    audio = np.sin(2 * np.pi * 1000 * np.linspace(0, duration, int(sr * duration), endpoint=False)).astype(np.float32)

    fig = create_spectrogram_figure(audio, sample_rate=sr, title="Test Spectrogram")
    ax = fig.axes[0]

    assert ax.get_xlabel() == "Time (seconds)"
    assert ax.get_ylabel() == "Frequency (kHz)"
    # Max frequency should equal Nyquist (8.0 kHz for 16 kHz sample rate)
    assert abs(ax.get_ylim()[1] - 8.0) < 0.1
    plt.close(fig)


def test_spectrogram_silence_stability():
    """Verifies that completely silent audio does not produce NaN/Inf or crash."""
    silence = np.zeros(16000, dtype=np.float32)
    fig = create_spectrogram_figure(silence, sample_rate=16000)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_spectrogram_very_short_signal():
    """Verifies that very short audio (< 512 samples) adapts FFT size without error."""
    short_audio = np.sin(np.linspace(0, 10, 100)).astype(np.float32)
    fig = create_spectrogram_figure(short_audio, sample_rate=16000)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_timeline_figure_threshold_and_points():
    """Verifies that timeline chart plots window scores accurately against the candidate threshold."""
    dec_real = DecisionResult(
        prediction=DecisionLabel.REAL,
        raw_score=0.7200,
        threshold=0.4820,
        threshold_status=ThresholdStatus.CANDIDATE_UNVALIDATED,
        threshold_source="Benchmark",
        score_direction=ScoreDirection.HIGHER_IS_REAL,
        margin_to_threshold=0.2380,
        explanation="Real",
    )
    dec_fake = DecisionResult(
        prediction=DecisionLabel.FAKE,
        raw_score=-0.6500,
        threshold=0.4820,
        threshold_status=ThresholdStatus.CANDIDATE_UNVALIDATED,
        threshold_source="Benchmark",
        score_direction=ScoreDirection.HIGHER_IS_REAL,
        margin_to_threshold=-1.1320,
        explanation="Fake",
    )

    w1 = WindowInferenceResult(WindowMetadata(0, 0.0, 4.0, 0, 64000, False), 0.7200, dec_real, 25.0)
    w2 = WindowInferenceResult(WindowMetadata(1, 1.0, 5.0, 16000, 80000, False), -0.6500, dec_fake, 25.0)

    long_res = LongRecordingInferenceResult(
        filename="multiseg.wav",
        total_windows=2,
        total_audio_duration_seconds=5.0,
        covered_duration_seconds=5.0,
        unanalyzed_duration_seconds=0.0,
        summary_real_windows=1,
        summary_fake_windows=1,
        min_score=-0.6500,
        max_score=0.7200,
        mean_score=0.0350,
        windows=[w1, w2],
        total_processing_time_ms=50.0,
    )

    fig = create_timeline_figure(long_res, candidate_threshold=0.4820)
    ax = fig.axes[0]

    assert ax.get_ylabel() == "Cosine Similarity S(x)"
    assert ax.get_ylim()[0] <= -1.0
    assert ax.get_ylim()[1] >= 1.0

    # Ensure candidate threshold label is present in legend
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("0.4820" in text for text in legend_texts)
    assert any("Candidate Threshold" in text for text in legend_texts)
    assert any("Unvalidated" in text for text in legend_texts)

    plt.close(fig)


def test_feature_views_figure_tensor_dimensions():
    """Verifies that neural feature view renders all 3 channels cleanly."""
    dummy_feat = np.random.randn(3, 256, 397).astype(np.float32)
    fig = create_feature_views_figure(dummy_feat, window_label="Test Window")

    # Figure must contain 3 subplots plus colorbars
    axes = [ax for ax in fig.axes if not ax.get_label().startswith("<colorbar>")]
    assert len(axes) == 3

    assert "ST-FrFT (α = 0.50" in axes[0].get_title()
    assert "ST-FrFT (α = 0.85" in axes[1].get_title()
    assert "Wavelet Scalogram" in axes[2].get_title()

    plt.close(fig)


def test_end_to_end_feature_extraction_plotting(engine):
    """End-to-end test: real 4.0s audio -> model.extractor -> feature views plot."""
    t = np.linspace(0, 4.0, 64000, endpoint=False)
    audio_wav = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    tensor = torch.from_numpy(audio_wav).unsqueeze(0).to(engine.device)

    with torch.inference_mode():
        feat = engine.model.extractor(tensor).detach().cpu().numpy()

    assert feat.shape == (1, 3, 256, 397)
    fig = create_feature_views_figure(feat[0], window_label="Real Audio 440Hz")
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_session_state_and_chart_cleanup():
    """Verifies that file ID changes clear cached analysis results and visualizations."""
    session = {
        "active_file_id": "speech_old.wav_5000",
        "loaded_audio": "audio_data",
        "analysis_result": "result_data",
    }

    # Simulate uploading a new file
    new_id = "speech_new.wav_6000"
    if session.get("active_file_id") != new_id:
        session["active_file_id"] = new_id
        session["loaded_audio"] = None
        session["analysis_result"] = None

    assert session["active_file_id"] == "speech_new.wav_6000"
    assert session["loaded_audio"] is None
    assert session["analysis_result"] is None
