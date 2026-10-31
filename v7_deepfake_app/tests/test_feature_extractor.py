"""
Automated unit tests for 2D FrFT-Wavelet Feature Extractor.
"""

import pytest
import torch
import numpy as np

from core.feature_extractor import FeatureExtractor
from core.exceptions import FeatureExtractionError


def test_feature_extractor_tensor_shape():
    """Verifies that the feature extractor produces the exact [1, 3, 256, 397] tensor."""
    extractor = FeatureExtractor()
    extractor.eval()

    dummy_audio = torch.randn(1, 64000)
    with torch.no_grad():
        features = extractor(dummy_audio)

    assert features.shape == (1, 3, 256, 397)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_feature_extractor_batched_input():
    """Verifies that batched audio inputs produce [B, 3, 256, 397]."""
    extractor = FeatureExtractor()
    extractor.eval()

    batch_audio = torch.randn(3, 64000)
    with torch.no_grad():
        features = extractor(batch_audio)

    assert features.shape == (3, 3, 256, 397)
    assert torch.isfinite(features).all()


def test_feature_extractor_channels_dict():
    """Verifies that individual forensic channels can be extracted separately for explainability."""
    extractor = FeatureExtractor()
    extractor.eval()

    audio = torch.randn(1, 64000)
    with torch.no_grad():
        ch_dict = extractor.extract_channels_dict(audio)

    assert "frft_050" in ch_dict
    assert "frft_085" in ch_dict
    assert "wavelet_scalo" in ch_dict
    assert "combined_tensor" in ch_dict

    assert ch_dict["frft_050"].shape == (1, 256, 397)
    assert ch_dict["frft_085"].shape == (1, 256, 397)
    assert ch_dict["wavelet_scalo"].shape == (1, 256, 397)
    assert ch_dict["combined_tensor"].shape == (1, 3, 256, 397)


def test_feature_extractor_determinism():
    """Verifies that the feature extractor is perfectly deterministic on identical inputs."""
    extractor = FeatureExtractor()
    extractor.eval()

    audio = torch.randn(1, 64000)
    with torch.no_grad():
        out1 = extractor(audio)
        out2 = extractor(audio)

    assert torch.allclose(out1, out2, atol=1e-7)


def test_feature_extractor_invalid_shape_rejection():
    """Verifies that audio tensors with invalid sample lengths raise FeatureExtractionError."""
    extractor = FeatureExtractor()
    invalid_audio = torch.randn(1, 32000)  # Wrong sample length
    with pytest.raises(FeatureExtractionError):
        extractor(invalid_audio)
