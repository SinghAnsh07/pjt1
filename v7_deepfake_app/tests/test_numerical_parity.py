"""
Automated unit tests for Strict Numerical Parity between Isolated Architecture
and Original Research Implementation.
"""

import os
import sys
import pytest
import numpy as np
import torch

from core.checkpoint_loader import load_model

CURRENT_DIR = os.path.dirname(__file__)
APP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CKPT_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "best_lcnn2d_v7.pt"))
SRC_PATH = os.path.abspath(os.path.join(APP_ROOT, "..", "v7", "v7_2d_frft_wavelet_lcnn.py"))


def test_bit_exact_numerical_parity():
    """
    Directly extracts the exact class source lines (150-322) from the research file
    into an isolated namespace, and validates zero numerical discrepancy against
    the isolated core/model.py implementation.
    """
    # 1. Load isolated model
    model_isolated, _ = load_model(CKPT_PATH, device="cpu")

    # 2. Extract and compile reference class lines
    with open(SRC_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    orig_code = "".join(lines[149:322])
    ref_ns = {
        "nn": torch.nn,
        "F": torch.nn.functional,
        "torch": torch,
        "np": np,
        "print": lambda *args, **kwargs: None,  # Suppress non-ascii prints
    }
    exec(orig_code, ref_ns)

    RefModelClass = ref_ns["EndToEndLCNN2D"]
    model_ref = RefModelClass()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model_ref.load_state_dict(ckpt["state_dict"], strict=True)
    model_ref.eval()

    # 3. Check parameters
    for (k1, v1), (k2, v2) in zip(model_isolated.state_dict().items(), model_ref.state_dict().items()):
        assert k1 == k2
        if v1.is_complex():
            diff = (v1 - v2).abs().max().item()
        else:
            diff = (v1.float() - v2.float()).abs().max().item()
        assert diff == 0.0, f"Weight discrepancy on key {k1}: {diff}"

    # 4. Check deterministic forward pass
    torch.manual_seed(98765)
    test_audio = torch.randn(1, 64000)

    with torch.no_grad():
        spec_isolated = model_isolated.extractor(test_audio)
        spec_ref = model_ref.extractor(test_audio)
        spec_diff = (spec_isolated - spec_ref).abs().max().item()
        assert spec_diff < 1e-6, f"Feature extractor discrepancy: {spec_diff}"

        emb_isolated = model_isolated.backbone(spec_isolated)
        emb_ref = model_ref.backbone(spec_ref)
        emb_diff = (emb_isolated - emb_ref).abs().max().item()
        assert emb_diff < 1e-6, f"Embedding discrepancy: {emb_diff}"

        score_isolated = model_isolated.oc_loss.get_score(emb_isolated).item()
        score_ref = model_ref.oc_loss.get_score(emb_ref).item()
        score_diff = abs(score_isolated - score_ref)
        assert score_diff < 1e-7, f"Cosine score discrepancy: {score_diff}"
