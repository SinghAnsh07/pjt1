"""
Isolated V7 Architecture Implementation: 2D FrFT-Wavelet LCNN with OC-Softmax.

Faithfully extracted from the verified research implementation:
Source File: C:\\Users\\ANSH SINGH\\Desktop\\capstone1\\v7\\v7_2d_frft_wavelet_lcnn.py (Lines 150-325)

This module contains exclusively the mathematical neural network layers and
feature extraction operators required for model instantiation and inference.
All training loops, dataset loading, and Kaggle-specific execution paths have
been strictly eliminated.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VectorizedSTFrFT(nn.Module):
    """
    Short-Time Fractional Fourier Transform (ST-FrFT) Spectrogram Layer.
    Computes frame-level FrFT across multiple fractional angles (alphas).
    
    Source: v7_2d_frft_wavelet_lcnn.py, lines 150-186
    """
    def __init__(self, n_fft: int = 512, hop_length: int = 160, alphas: Tuple[float, float] = (0.50, 0.85)):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alphas = list(alphas)
        self.register_buffer('window', torch.hann_window(n_fft))

        # Precompute complex chirp modulation vectors
        N = n_fft
        n = torch.arange(N, dtype=torch.float32)
        chirps = []
        for a in self.alphas:
            phi = a * np.pi / 2.0
            c = torch.exp(1j * np.pi * np.tan(phi / 2.0) * (n**2) / N)
            chirps.append(c)
        self.register_buffer('chirps', torch.stack(chirps))  # Shape: (num_alphas, N)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Floating-point tensor of shape (B, T_samples) or (T_samples,)
        Returns:
            Tensor of shape (B, len(alphas), 256, n_frames)
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        # Audio frame unfolding: (B, n_frames, n_fft)
        frames = audio.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
        frames = frames * self.window
        frames_c = frames.to(torch.complex64)

        channels = []
        for i in range(len(self.alphas)):
            modulated = frames_c * self.chirps[i]
            frft = torch.fft.fft(modulated, dim=-1) / np.sqrt(self.n_fft)
            mag = torch.abs(frft)[:, :, :self.n_fft // 2]
            log_mag = torch.log1p(mag).transpose(1, 2)  # (B, 256, n_frames)
            channels.append(log_mag)
        return torch.stack(channels, dim=1)  # (B, len(alphas), 256, n_frames)


class VectorizedWaveletScalogram(nn.Module):
    """
    Multi-Scale Wavelet Scalogram Filterbank Layer.
    Decomposes audio across 64 continuous scales via 1D convolution.
    
    Source: v7_2d_frft_wavelet_lcnn.py, lines 188-217
    """
    def __init__(self, num_scales: int = 64, hop_length: int = 160):
        super().__init__()
        self.hop_length = hop_length
        self.num_scales = num_scales
        scales = torch.logspace(np.log10(1.0), np.log10(num_scales), num_scales)
        t = torch.linspace(-4, 4, 256)
        filters = []
        for s in scales:
            ts = t / s
            w = torch.exp(-0.5 * ts**2) * torch.cos(5.0 * ts) / torch.sqrt(s)
            filters.append(w)
        filters = torch.stack(filters).unsqueeze(1)
        self.register_buffer('filters', filters)  # Shape: (64, 1, 256)

    def forward(self, audio: torch.Tensor, target_time_frames: int) -> torch.Tensor:
        """
        Args:
            audio: Tensor of shape (B, T_samples)
            target_time_frames: Number of temporal frames to match ST-FrFT (typically 397)
        Returns:
            Tensor of shape (B, 1, 256, target_time_frames)
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        x = audio.unsqueeze(1)
        out = F.conv1d(x, self.filters, stride=self.hop_length, padding=128)
        log_scalo = torch.log1p(torch.abs(out))
        scalo_interp = F.interpolate(
            log_scalo.unsqueeze(1),
            size=(256, target_time_frames),
            mode='bilinear',
            align_corners=False
        )
        return scalo_interp  # (B, 1, 256, target_time_frames)


class FrFTWavelet2DExtractor(nn.Module):
    """
    Combines ST-FrFT and Continuous Wavelet Scalogram into a unified 3-Channel 2D Tensor.
    
    Output channel mapping:
      - Channel 0: ST-FrFT (alpha = 0.50)
      - Channel 1: ST-FrFT (alpha = 0.85)
      - Channel 2: Continuous Morlet Wavelet Scalogram
      
    Source: v7_2d_frft_wavelet_lcnn.py, lines 219-232
    """
    def __init__(self):
        super().__init__()
        self.st_frft = VectorizedSTFrFT(n_fft=512, hop_length=160, alphas=(0.50, 0.85))
        self.wavelet = VectorizedWaveletScalogram(num_scales=64, hop_length=160)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Tensor of shape (B, 64000) or (64000,)
        Returns:
            Tensor of shape (B, 3, 256, 397)
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            
        frft_maps = self.st_frft(audio)  # (B, 2, 256, T)
        T_frames = frft_maps.shape[-1]
        scalo_map = self.wavelet(audio, T_frames)  # (B, 1, 256, T)
        return torch.cat([frft_maps, scalo_map], dim=1)  # (B, 3, 256, T)


class MFM2D(nn.Module):
    """
    2D Max-Feature-Map activation: halves channel dimension via competitive feature selection.
    For an input of shape (B, 2*C, H, W), returns (B, C, H, W).
    
    Source: v7_2d_frft_wavelet_lcnn.py, lines 238-243
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1] // 2
        return torch.max(x[:, :n], x[:, n:])


class LCNN2DBlock(nn.Module):
    """
    2D Convolutional LCNN Building Block:
    Conv2D(in_c -> out_c * 2) -> MFM-2D(-> out_c) -> BatchNorm2D(out_c) -> MaxPool2D(2x2)
    
    Source: v7_2d_frft_wavelet_lcnn.py, lines 245-256
    """
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, padding: int = 1, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * 2, kernel_size=kernel_size, padding=padding, bias=False)
        self.mfm = MFM2D()
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.bn(self.mfm(self.conv(x))))


class LCNN2D(nn.Module):
    """
    4-Stage 2D Convolutional Light CNN (LCNN-2D) Backbone.
    Projects a 3-channel 2D spatiotemporal tensor [B, 3, 256, 397] into a 128-dimensional embedding.
    Total trainable parameters: 170,208.
    
    Source: v7_2d_frft_wavelet_lcnn.py, lines 258-282
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.stage1 = LCNN2DBlock(in_channels, 32, kernel_size=5, padding=2, pool=True)  # 256x397 -> 128x198
        self.stage2 = LCNN2DBlock(32, 48, kernel_size=3, padding=1, pool=True)           # 128x198 -> 64x99
        self.stage3 = LCNN2DBlock(48, 64, kernel_size=3, padding=1, pool=True)           # 64x99   -> 32x49
        self.stage4 = LCNN2DBlock(64, 64, kernel_size=3, padding=1, pool=False)          # 32x49   -> 32x49
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.head(x)  # Shape: (B, 128)


class OCSoftmax(nn.Module):
    """
    One-Class Softmax (OC-Softmax) Metric Learning Head.
    Calculates cosine similarity between normalized embedding vector and learned bona fide center w.
    
    Score definition:
      S(x) = (x_norm · w_norm) ∈ [-1.0, 1.0]
      Higher score corresponds to closer proximity to the bona fide center w (more authentic).
      
    Source: v7_2d_frft_wavelet_lcnn.py, lines 285-306
    """
    def __init__(self, feat_dim: int = 128, alpha: float = 20.0, m_real: float = 0.90, m_fake: float = 0.20):
        super().__init__()
        self.alpha = alpha
        self.m_real = m_real
        self.m_fake = m_fake
        self.w = nn.Parameter(torch.randn(feat_dim))
        nn.init.normal_(self.w, std=0.01)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        w_norm = F.normalize(self.w, p=2, dim=0)
        x_norm = F.normalize(x, p=2, dim=1)
        scores = torch.mv(x_norm, w_norm)
        if y is not None:
            z = torch.where(y == 0, self.m_real - scores, scores - self.m_fake)
            loss = F.softplus(self.alpha * z).mean()
            return loss, scores
        return scores

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity against normalized bona fide center w."""
        w_norm = F.normalize(self.w, p=2, dim=0)
        x_norm = F.normalize(x, p=2, dim=1)
        return torch.mv(x_norm, w_norm)


class EndToEndLCNN2D(nn.Module):
    """
    Complete End-to-End Deepfake Detection Pipeline.
    Combines:
      1. Extractor: FrFTWavelet2DExtractor [B, 64000] -> [B, 3, 256, 397]
      2. Backbone:  LCNN2D                 [B, 3, 256, 397] -> [B, 128]
      3. Metric:    OCSoftmax              [B, 128] -> [B] (Cosine similarity scores)
      
    Total Parameters: 170,336
      - Backbone: 170,208
      - OC-Softmax center w: 128
      - Extractor: 0 trainable (17,920 registered buffer elements)
      
    Source: v7_2d_frft_wavelet_lcnn.py, lines 308-322
    """
    def __init__(self):
        super().__init__()
        self.extractor = FrFTWavelet2DExtractor()
        self.backbone = LCNN2D(in_channels=3, embed_dim=128)
        self.oc_loss = OCSoftmax(feat_dim=128)

    def forward(self, audio: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes end-to-end forward pass from raw audio waveform to cosine similarity score.
        Args:
            audio: Waveform tensor of shape (B, 64000) or (64000,)
            y: Optional ground truth labels (0=bona fide, 1=spoof) for loss computation
        Returns:
            If y is None: 1D Tensor of shape (B,) containing cosine similarity scores
            If y is provided: Tuple of (loss, scores)
        """
        spec_2d = self.extractor(audio)
        emb = self.backbone(spec_2d)
        if y is not None:
            return self.oc_loss(emb, y)
        return self.oc_loss.get_score(emb)

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extracts the 3-channel 2D spatiotemporal tensor [B, 3, 256, 397]."""
        return self.extractor(audio)

    def extract_embedding(self, spec_2d: torch.Tensor) -> torch.Tensor:
        """Extracts the 128-dimensional embedding from the spatiotemporal tensor."""
        return self.backbone(spec_2d)

    def compute_score(self, emb: torch.Tensor) -> torch.Tensor:
        """Computes cosine similarity scores from 128-dimensional embeddings."""
        return self.oc_loss.get_score(emb)
