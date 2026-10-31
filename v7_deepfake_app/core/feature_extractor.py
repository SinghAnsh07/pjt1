"""
Feature extraction module for the V7 Audio Deepfake Detection System.

Provides standalone access to the multi-channel 2D spatiotemporal extraction
pipeline (ST-FrFT alpha=0.50, ST-FrFT alpha=0.85, and Continuous Wavelet Scalogram).
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from core.model import FrFTWavelet2DExtractor, VectorizedSTFrFT, VectorizedWaveletScalogram
from core.exceptions import FeatureExtractionError


class FeatureExtractor(nn.Module):
    """
    Authoritative 3-Channel Spatiotemporal Feature Extractor.
    Transforms raw 16 kHz audio waveforms [B, 64000] into the 3-channel 2D tensor [B, 3, 256, 397].
    """
    def __init__(self):
        super().__init__()
        self.extractor = FrFTWavelet2DExtractor()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Executes unified feature extraction.
        Args:
            audio: Tensor of shape (B, 64000) or (64000,)
        Returns:
            Tensor of shape (B, 3, 256, 397)
        """
        try:
            if not isinstance(audio, torch.Tensor):
                raise FeatureExtractionError(f"Expected torch.Tensor input, got {type(audio).__name__}")
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            if audio.shape[-1] != 64000:
                raise FeatureExtractionError(f"Expected 64000 audio samples, got {audio.shape[-1]}")
            if not torch.isfinite(audio).all():
                raise FeatureExtractionError("Audio tensor contains non-finite values (NaN or Inf)")

            return self.extractor(audio)
        except Exception as e:
            if isinstance(e, FeatureExtractionError):
                raise
            raise FeatureExtractionError(f"Feature extraction failed: {str(e)}") from e

    def extract_channels_dict(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts individual 2D channels separately for visualization and forensic explainability.
        Returns:
            Dictionary containing:
              - 'frft_050': ST-FrFT with alpha=0.50 [B, 256, 397]
              - 'frft_085': ST-FrFT with alpha=0.85 [B, 256, 397]
              - 'wavelet_scalo': Continuous Wavelet Scalogram [B, 256, 397]
              - 'combined_tensor': Full 3-channel tensor [B, 3, 256, 397]
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        combined = self.forward(audio)
        return {
            "frft_050": combined[:, 0, :, :],
            "frft_085": combined[:, 1, :, :],
            "wavelet_scalo": combined[:, 2, :, :],
            "combined_tensor": combined,
        }
