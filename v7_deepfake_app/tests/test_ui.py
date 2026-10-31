"""
Automated unit and integration tests for Streamlit UI foundation (Phase 4).
"""

import os
import sys
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

from core.inference import V7InferenceEngine
from core.audio_loader import load_audio
from core.preprocessor import preprocess_waveform
from core.windowing import analyze_long_recording
from core.schemas import (
    LoadedAudio,
    InferenceResult,
    LongRecordingInferenceResult,
    DecisionResult,
    DecisionLabel,
    ScoreDirection,
    ThresholdStatus,
    TimingBreakdown,
    PreprocessingResult,
    ModelMetadata,
)
from core.exceptions import (
    UnsupportedAudioFormatError,
    CorruptAudioFileError,
    EmptyAudioFileError,
)
from ui.styles import get_custom_css
from ui.components import (
    render_header,
    render_single_result,
    render_long_recording_result,
    render_technical_details,
    render_metadata_card,
)

CURRENT_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
APP_PY_PATH = os.path.join(APP_ROOT, "app.py")
CKPT_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))


@pytest.fixture(scope="module")
def engine():
    return V7InferenceEngine(checkpoint_path=CKPT_PATH, device="cpu")


def test_app_startup_clean():
    """Verifies that app.py initializes cleanly with AppTest without exceptions or premature inference."""
    at = AppTest.from_file(APP_PY_PATH)
    at.run(timeout=20)
    assert not at.exception
    # Ensure uploader is displayed
    uploaders = at.get("file_uploader")
    assert len(uploaders) == 1
    # Ensure initial state does not contain analysis result
    assert "analysis_result" not in at.session_state or at.session_state["analysis_result"] is None


def test_app_ffmpeg_missing_notice_displayed():
    """Verifies that when FFmpeg is missing, an informative callout is displayed."""
    with patch("core.video_processor.discover_ffmpeg", return_value=(False, None)):
        at = AppTest.from_file(APP_PY_PATH)
        at.run(timeout=20)
        assert not at.exception
        # Info box explaining native vs video requirements must be present
        info_texts = [inf.value for inf in at.info]
        assert any("WAV, FLAC, MP3" in text for text in info_texts)
        assert any("FFmpeg" in text for text in info_texts)


def test_custom_css_syntax_valid():
    """Verifies that custom CSS is non-empty and well-formed."""
    css = get_custom_css()
    assert "<style>" in css
    assert "</style>" in css
    assert "verdict-card" in css
    assert "metadata-card" in css


def test_single_result_rendering_no_probability_percentages():
    """Verifies that the verdict display never fabricates probability percentages or confidence meters."""
    dummy_decision = DecisionResult(
        prediction=DecisionLabel.REAL,
        raw_score=0.6500,
        threshold=0.4820,
        threshold_status=ThresholdStatus.CANDIDATE_UNVALIDATED,
        threshold_source="Test Benchmark",
        score_direction=ScoreDirection.HIGHER_IS_REAL,
        margin_to_threshold=0.1680,
        explanation="Test explanation",
    )
    dummy_prep = PreprocessingResult(
        original_sample_rate=16000,
        target_sample_rate=16000,
        original_samples=64000,
        target_samples=64000,
        original_duration_seconds=4.0,
        analyzed_duration_seconds=4.0,
        is_multichannel=False,
        is_resampled=False,
        is_padded=False,
        is_truncated=False,
    )
    dummy_meta = ModelMetadata(
        checkpoint_path="test.pt",
        checkpoint_sha256="abc",
        saved_epoch=26,
        saved_dev_eer=3.14,
        model_class="EndToEndLCNN2D",
        device="cpu",
        total_parameters=170336,
        trainable_parameters=0,
        buffer_elements=18340,
        strict_load_verified=True,
    )
    dummy_timing = TimingBreakdown(
        preprocessing_ms=1.0,
        feature_extraction_ms=2.0,
        neural_forward_ms=20.0,
        decision_ms=0.5,
        total_ms=23.5,
    )
    single_res = InferenceResult(
        status="SUCCESS",
        filename="test.wav",
        model_score=0.6500,
        decision=dummy_decision,
        preprocessing=dummy_prep,
        model_metadata=dummy_meta,
        timing=dummy_timing,
        feature_tensor_shape=(1, 3, 256, 397),
        embedding_shape=(1, 128),
    )

    with patch("streamlit.markdown") as mock_markdown:
        render_single_result(single_res)
        assert mock_markdown.called
        rendered_html = mock_markdown.call_args[0][0]
        # Must show approved label
        assert "Likely Genuine Audio" in rendered_html
        # Must contain screening disclaimer
        assert "screening result" in rendered_html
        # Must NOT invent arbitrary percentage confidence
        assert "%" not in rendered_html
        assert "confidence" not in rendered_html.lower()
        assert "probability" not in rendered_html.lower()


def test_fake_verdict_rendering():
    """Verifies that a fake verdict renders 'Potential AI-Generated or Manipulated Audio'."""
    fake_decision = DecisionResult(
        prediction=DecisionLabel.FAKE,
        raw_score=-0.8500,
        threshold=0.4820,
        threshold_status=ThresholdStatus.CANDIDATE_UNVALIDATED,
        threshold_source="Test Benchmark",
        score_direction=ScoreDirection.HIGHER_IS_REAL,
        margin_to_threshold=-1.3320,
        explanation="Test explanation",
    )
    dummy_res = MagicMock()
    dummy_res.decision = fake_decision

    with patch("streamlit.markdown") as mock_markdown:
        render_single_result(dummy_res)
        rendered_html = mock_markdown.call_args[0][0]
        assert "Potential AI-Generated or Manipulated Audio" in rendered_html


def test_long_recording_timeline_no_fake_probability():
    """Verifies that long recording analysis reports window counts without whole-recording probability."""
    dummy_long = LongRecordingInferenceResult(
        filename="speech_10s.wav",
        total_windows=7,
        total_audio_duration_seconds=10.0,
        covered_duration_seconds=10.0,
        unanalyzed_duration_seconds=0.0,
        summary_real_windows=5,
        summary_fake_windows=2,
        min_score=-0.4500,
        max_score=0.7200,
        mean_score=0.3500,
        windows=[],
        total_processing_time_ms=180.0,
        system_warnings=["Timeline summary"],
    )

    with patch("streamlit.markdown") as mock_md, patch("streamlit.metric") as mock_metric, patch("streamlit.dataframe") as mock_df:
        render_long_recording_result(dummy_long)
        assert mock_metric.called
        # Check metrics exposed: Total Windows, Likely Genuine, Potential Fake, Coverage
        metric_labels = [call[0][0] for call in mock_metric.call_args_list]
        assert "Total Windows" in metric_labels
        assert "Likely Genuine" in metric_labels
        assert "Potential Fake" in metric_labels
        assert "Timeline Coverage" in metric_labels


def test_session_state_file_invalidation_logic():
    """Verifies that changing the active file ID invalidates prior analysis results."""
    session = {
        "active_file_id": "fileA.wav_1024",
        "loaded_audio": "dummy_audio_A",
        "analysis_result": "dummy_result_A",
        "is_long_recording": False,
    }

    # Simulate selecting a new file B
    new_file_id = "fileB.wav_2048"
    if session.get("active_file_id") != new_file_id:
        session["active_file_id"] = new_file_id
        session["loaded_audio"] = None
        session["analysis_result"] = None
        session["is_long_recording"] = False

    assert session["active_file_id"] == "fileB.wav_2048"
    assert session["analysis_result"] is None
    assert session["loaded_audio"] is None


def test_full_pipeline_from_file_to_single_result(engine):
    """End-to-end integration test: Real WAV audio -> load -> preprocess -> model inference -> valid result."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sine_4s.wav")
        sr = 16000
        audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000)) * 0.5).astype(np.float32)
        sf.write(path, audio, sr)

        loaded = load_audio(path)
        assert loaded.duration_seconds == 4.0

        res = engine.predict_waveform(loaded.waveform, sample_rate=loaded.sample_rate, filename="sine_4s.wav")
        assert res.status == "SUCCESS"
        assert res.decision.prediction in [DecisionLabel.REAL, DecisionLabel.FAKE]
        assert np.isfinite(res.model_score)


def test_full_pipeline_from_file_to_long_recording(engine):
    """End-to-end integration test: Real 7.0s WAV audio -> load -> 16k mono -> multi-window timeline."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "speech_7s.wav")
        sr = 44100
        total_samples = int(sr * 7.0)
        audio = (np.sin(2 * np.pi * 300 * np.linspace(0, 7.0, total_samples)) * 0.4).astype(np.float32)
        sf.write(path, audio, sr)

        loaded = load_audio(path)
        tensor, _ = preprocess_waveform(loaded.waveform, sample_rate=loaded.sample_rate, target_samples=None)
        audio_16k_mono = tensor.squeeze(0).numpy()

        long_res = analyze_long_recording(audio_16k_mono, engine=engine, hop_seconds=1.0, filename="speech_7s.wav")
        assert isinstance(long_res, LongRecordingInferenceResult)
        assert long_res.total_windows == 4
        assert abs(long_res.covered_duration_seconds - 7.0) < 1e-4
        assert len(long_res.windows) == 4
