"""
Secure Checkpoint Loader for the V7 Audio Deepfake Detection System.

Handles cryptographic verification, safe deserialization with weights_only=True,
strict state dictionary validation, device binding, and model instantiation.
"""

import os
import hashlib
import torch
from typing import Tuple, Union

from core.model import EndToEndLCNN2D
from core.schemas import ModelMetadata
from core.exceptions import (
    CheckpointNotFoundError,
    CheckpointIncompatibleError,
    CheckpointCorruptedError,
)


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file efficiently in 64 KB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_device(device_choice: str = "auto") -> torch.device:
    """
    Resolves execution device safely.
    Options:
      - 'auto': selects CUDA if torch reports CUDA available; otherwise CPU.
      - 'cpu': forces CPU execution.
      - 'cuda': requires CUDA execution; raises error if unavailable.
    """
    choice = device_choice.lower().strip()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif choice == "cpu":
        return torch.device("cpu")
    elif choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution was explicitly requested, but CUDA is not available in PyTorch.")
        return torch.device("cuda")
    else:
        raise ValueError(f"Unknown device specification '{device_choice}'. Supported: 'auto', 'cpu', 'cuda'.")


def load_model(
    checkpoint_path: str,
    device: str = "auto",
) -> Tuple[EndToEndLCNN2D, ModelMetadata]:
    """
    Loads and strictly validates the V7 EndToEndLCNN2D model from checkpoint.

    Args:
        checkpoint_path: Absolute or relative filesystem path to .pt checkpoint
        device: Device selection target ('auto', 'cpu', or 'cuda')

    Returns:
        Tuple of (initialized model in eval mode, verified ModelMetadata)

    Raises:
        CheckpointNotFoundError: If file does not exist or is not a file
        CheckpointCorruptedError: If file cannot be read or deserialized
        CheckpointIncompatibleError: If state_dict fails strict key matching
    """
    # 1. Path existence and readability check
    if not os.path.exists(checkpoint_path):
        raise CheckpointNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise CheckpointNotFoundError(f"Checkpoint path is not a regular file: {checkpoint_path}")

    # 2. Checkpoint SHA-256 checksum
    try:
        sha256_hash = compute_file_sha256(checkpoint_path)
    except Exception as e:
        raise CheckpointCorruptedError(f"Failed to read checkpoint file for hashing: {str(e)}") from e

    # 3. Secure deserialization with weights_only=True
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        # Fallback inspection: report exact failure cause
        raise CheckpointCorruptedError(
            f"Failed to securely deserialize checkpoint using weights_only=True: {str(e)}"
        ) from e

    # 4. Checkpoint dictionary validation
    if not isinstance(checkpoint_data, dict):
        raise CheckpointCorruptedError(
            f"Expected checkpoint to be a dictionary, got {type(checkpoint_data).__name__}"
        )
    if "state_dict" not in checkpoint_data:
        raise CheckpointCorruptedError("Checkpoint does not contain 'state_dict' key.")

    state_dict = checkpoint_data["state_dict"]
    saved_epoch = checkpoint_data.get("epoch")
    saved_dev_eer = checkpoint_data.get("dev_eer")

    # 5. Model instantiation
    try:
        model = EndToEndLCNN2D()
    except Exception as e:
        raise CheckpointIncompatibleError(f"Failed to instantiate EndToEndLCNN2D architecture: {str(e)}") from e

    # 6. Strict state dictionary loading
    try:
        load_result = model.load_state_dict(state_dict, strict=True)
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys
        if missing_keys or unexpected_keys:
            raise CheckpointIncompatibleError(
                f"Strict weight loading failed. Missing: {missing_keys}, Unexpected: {unexpected_keys}"
            )
    except Exception as e:
        if isinstance(e, CheckpointIncompatibleError):
            raise
        raise CheckpointIncompatibleError(f"State dictionary incompatible with architecture: {str(e)}") from e

    # 7. Device resolution and model binding
    target_device = resolve_device(device)
    model.to(target_device)
    model.eval()

    # 8. Freeze all parameters for inference safety
    for param in model.parameters():
        param.requires_grad = False

    # 9. Compute verified parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    buffer_elems = sum(b.numel() for b in model.buffers())

    metadata = ModelMetadata(
        checkpoint_path=os.path.abspath(checkpoint_path),
        checkpoint_sha256=sha256_hash,
        saved_epoch=saved_epoch,
        saved_dev_eer=float(saved_dev_eer) if saved_dev_eer is not None else None,
        model_class="EndToEndLCNN2D",
        device=str(target_device),
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        buffer_elements=buffer_elems,
        strict_load_verified=True,
    )

    return model, metadata
