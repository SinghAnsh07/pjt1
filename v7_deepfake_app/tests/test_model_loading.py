"""
Automated unit tests for Model Loading and Checkpoint Verification.
"""

import os
import pytest
import torch

from core.checkpoint_loader import load_model, resolve_device
from core.exceptions import CheckpointNotFoundError, CheckpointCorruptedError
from core.model import EndToEndLCNN2D

# Workspace paths
CURRENT_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CKPT_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))


def test_load_valid_checkpoint():
    """Verifies that the legitimate V7 checkpoint loads with strict=True on CPU."""
    model, meta = load_model(CKPT_PATH, device="cpu")
    assert isinstance(model, EndToEndLCNN2D)
    assert meta.strict_load_verified is True
    assert meta.total_parameters == 170336
    assert meta.trainable_parameters == 0  # Parameters should be frozen
    assert meta.buffer_elements == 18340
    assert meta.saved_epoch == 26
    assert abs(meta.saved_dev_eer - 3.1397) < 1e-3
    assert not model.training  # Model must be in eval mode


def test_load_nonexistent_checkpoint():
    """Verifies that attempting to load a missing checkpoint raises CheckpointNotFoundError."""
    with pytest.raises(CheckpointNotFoundError):
        load_model("C:/nonexistent_path/checkpoint.pt", device="cpu")


def test_device_resolution():
    """Verifies safe device resolution logic."""
    cpu_dev = resolve_device("cpu")
    assert cpu_dev == torch.device("cpu")

    auto_dev = resolve_device("auto")
    assert auto_dev in [torch.device("cpu"), torch.device("cuda")]

    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_cuda_error_when_unavailable():
    """Verifies that requesting CUDA raises an error when CUDA is not present."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_parameters_frozen_in_eval_mode():
    """Verifies that all loaded parameters have requires_grad=False."""
    model, _ = load_model(CKPT_PATH, device="cpu")
    for param in model.parameters():
        assert not param.requires_grad
