"""
Automated unit tests for V7 Deepfake Inference Engine.
"""

import os
import pytest
import numpy as np
import torch

from core.inference import V7InferenceEngine, predict_waveform
from core.schemas import InferenceResult, DecisionLabel
from core.exceptions import InvalidWaveformError, CheckpointNotFoundError

CURRENT_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CKPT_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))


@pytest.fixture(scope="module")
def engine():
    """Initializes and reuses a single inference engine across tests."""
    return V7InferenceEngine(checkpoint_path=CKPT_PATH, device="cpu")


def test_inference_pipeline_valid_waveform(engine):
    """Verifies that a valid waveform completes the full pipeline and returns structured InferenceResult."""
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000, endpoint=False)).astype(np.float32)
    result = engine.predict_waveform(audio, 16000, filename="test_sine.wav")

    assert isinstance(result, InferenceResult)
    assert result.status == "SUCCESS"
    assert result.filename == "test_sine.wav"
    assert np.isfinite(result.model_score)
    assert -1.0 <= result.model_score <= 1.0
    assert result.feature_tensor_shape == (1, 3, 256, 397)
    assert result.embedding_shape == (1, 128)
    assert result.decision.prediction in [DecisionLabel.REAL, DecisionLabel.FAKE]
    assert result.timing.total_ms > 0


def test_inference_with_resampling_and_padding(engine):
    """Verifies inference on short 44.1 kHz audio requiring resampling and circular padding."""
    audio = np.random.randn(22050).astype(np.float32)  # 0.5 seconds at 44.1 kHz
    result = engine.predict_waveform(audio, 44100)

    assert result.status == "SUCCESS"
    assert result.preprocessing.is_resampled is True
    assert result.preprocessing.is_padded is True
    assert np.isfinite(result.model_score)


def test_inference_rejects_empty_audio(engine):
    """Verifies that empty audio raises an exception and never produces a default classification."""
    empty_audio = np.array([], dtype=np.float32)
    with pytest.raises(InvalidWaveformError):
        engine.predict_waveform(empty_audio, 16000)


def test_stateless_predict_waveform_helper():
    """Verifies that the stateless predict_waveform function works correctly."""
    audio = np.zeros(64000, dtype=np.float32)
    result = predict_waveform(audio, 16000, checkpoint_path=CKPT_PATH, device="cpu")
    assert result.status == "SUCCESS"
    assert np.isfinite(result.model_score)
