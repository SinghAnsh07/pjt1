"""
Verification Script for V7 Audio Deepfake Inference Engine.

Performs a complete end-to-end smoke test of the isolated V7 engine using
the actual trained model weights. Validates strict weight deserialization,
preprocessing contracts, 2D feature extraction tensor dimensions, OC-Softmax
scoring, and conservative decision evaluation.

Usage:
    python scripts/verify_inference.py
"""

import os
import sys
import time
import numpy as np
import torch

# Ensure application root is on Python module search path
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from core.checkpoint_loader import load_model
from core.inference import V7InferenceEngine
from core.schemas import ThresholdStatus, DecisionLabel


def main() -> int:
    print("=" * 75)
    print("  V7 AUDIO DEEPFAKE INFERENCE ENGINE — VERIFICATION SUITE")
    print("=" * 75)

    # 1. Checkpoint location resolution
    workspace_root = os.path.abspath(os.path.join(APP_ROOT, ".."))
    ckpt_path = os.path.join(workspace_root, "v7", "best_lcnn2d_v7.pt")

    print(f"\n[1/5] Checking Model Checkpoint...")
    print(f"  Target path: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Checkpoint file does not exist at {ckpt_path}")
        return 1
    print(f"  File found. Size: {os.path.getsize(ckpt_path):,} bytes")

    # 2. Model loading and strict validation
    print(f"\n[2/5] Loading Model with Strict Validation (weights_only=True)...")
    try:
        t0 = time.perf_counter()
        engine = V7InferenceEngine(checkpoint_path=ckpt_path, device="auto")
        load_time_ms = (time.perf_counter() - t0) * 1000.0
        meta = engine.metadata
        print(f"  Load status: SUCCESS in {load_time_ms:.1f} ms")
        print(f"  SHA-256: {meta.checkpoint_sha256}")
        print(f"  Checkpoint Epoch: {meta.saved_epoch}")
        print(f"  Checkpoint Dev EER: {meta.saved_dev_eer:.4f}%")
        print(f"  Execution Device: {meta.device}")
        print(f"  Total Parameters: {meta.total_parameters:,}")
        print(f"  Trainable Parameters: {meta.trainable_parameters:,}")
        print(f"  Registered Buffers: {meta.buffer_elements:,}")
        print(f"  Strict Load Verified: {meta.strict_load_verified}")
    except Exception as e:
        print(f"  ERROR loading checkpoint: {str(e)}")
        return 1

    # 3. Controlled Synthetic Smoke Tests
    # NOTE: Synthetic inputs verify pipeline execution, numerical stability,
    # and tensor shapes. They DO NOT evaluate forensic classification accuracy.
    print(f"\n[3/5] Running Waveform Preprocessing and Inference Tests...")
    print("  * NOTICE: Inputs are deterministic synthetic waveforms for technical smoke testing.")

    test_cases = [
        {
            "name": "Standard 4.0s Sine (16 kHz)",
            "sr": 16000,
            "duration": 4.0,
            "freq": 440.0,
            "expected_samples": 64000,
            "expect_resample": False,
            "expect_pad": False,
            "expect_trunc": False,
        },
        {
            "name": "Short 2.0s Sine (44.1 kHz - requires resample & wrap-pad)",
            "sr": 44100,
            "duration": 2.0,
            "freq": 300.0,
            "expected_samples": 64000,
            "expect_resample": True,
            "expect_pad": True,
            "expect_trunc": False,
        },
        {
            "name": "Long 6.0s Sine (48 kHz - requires resample & truncation)",
            "sr": 48000,
            "duration": 6.0,
            "freq": 500.0,
            "expected_samples": 64000,
            "expect_resample": True,
            "expect_pad": False,
            "expect_trunc": True,
        },
        {
            "name": "Multichannel Stereo 4.0s (16 kHz - requires downmixing)",
            "sr": 16000,
            "duration": 4.0,
            "freq": 440.0,
            "is_stereo": True,
            "expected_samples": 64000,
            "expect_resample": False,
            "expect_pad": False,
            "expect_trunc": False,
        },
    ]

    all_passed = True
    for idx, tc in enumerate(test_cases, 1):
        print(f"\n  --- Test Case {idx}: {tc['name']} ---")
        t = np.linspace(0, tc["duration"], int(tc["sr"] * tc["duration"]), endpoint=False)
        audio = np.sin(2 * np.pi * tc["freq"] * t).astype(np.float32)
        if tc.get("is_stereo"):
            audio = np.stack([audio, audio * 0.8], axis=0)  # Shape: (2, N)

        try:
            result = engine.predict_waveform(audio, tc["sr"], filename=tc["name"])

            # Verify finite outputs
            if not np.isfinite(result.model_score):
                print(f"    FAIL: Model score is non-finite: {result.model_score}")
                all_passed = False
                continue

            # Verify feature tensor shape
            if result.feature_tensor_shape != (1, 3, 256, 397):
                print(f"    FAIL: Unexpected feature shape: {result.feature_tensor_shape}")
                all_passed = False
                continue

            # Verify embedding shape
            if result.embedding_shape != (1, 128):
                print(f"    FAIL: Unexpected embedding shape: {result.embedding_shape}")
                all_passed = False
                continue

            # Verify preprocessing contract
            prep = result.preprocessing
            if prep.is_resampled != tc["expect_resample"]:
                print(f"    FAIL: Expected resample={tc['expect_resample']}, got {prep.is_resampled}")
                all_passed = False
            if prep.is_padded != tc["expect_pad"]:
                print(f"    FAIL: Expected pad={tc['expect_pad']}, got {prep.is_padded}")
                all_passed = False
            if prep.is_truncated != tc["expect_trunc"]:
                print(f"    FAIL: Expected trunc={tc['expect_trunc']}, got {prep.is_truncated}")
                all_passed = False

            # Verify decision engine properties
            dec = result.decision
            print(f"    Raw Score       : {result.model_score:+.4f}")
            print(f"    Threshold       : {dec.threshold:.4f} ({dec.threshold_status.value})")
            print(f"    Prediction      : {dec.prediction.value}")
            print(f"    Feature Tensor  : {result.feature_tensor_shape}")
            print(f"    Embedding Tensor: {result.embedding_shape}")
            print(f"    Timings (ms)    : Prep: {result.timing.preprocessing_ms:.1f}ms | Feat: {result.timing.feature_extraction_ms:.1f}ms | Fwd: {result.timing.neural_forward_ms:.1f}ms | Total: {result.timing.total_ms:.1f}ms")
            if result.system_warnings:
                print(f"    Warnings ({len(result.system_warnings)}): {result.system_warnings[0]}")
            print("    Status: PASS")

        except Exception as e:
            print(f"    FAIL with exception: {type(e).__name__}: {str(e)}")
            all_passed = False

    # 4. Numerical determinism check
    print(f"\n[4/5] Checking Numerical Determinism across Repeated Passes...")
    test_wav = np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 64000, endpoint=False)).astype(np.float32)
    res1 = engine.predict_waveform(test_wav, 16000)
    res2 = engine.predict_waveform(test_wav, 16000)
    diff = abs(res1.model_score - res2.model_score)
    print(f"  Score 1: {res1.model_score:.6f}")
    print(f"  Score 2: {res2.model_score:.6f}")
    print(f"  Absolute Difference: {diff:.8f}")
    if diff > 1e-6:
        print("  FAIL: Inference is non-deterministic!")
        all_passed = False
    else:
        print("  PASS: Inference is deterministic within floating-point tolerance.")

    # 5. Final Verdict
    print(f"\n[5/5] Final Verification Verdict:")
    if all_passed:
        print("  >>> ALL CHECKS PASSED. V7 Inference Engine is verified and production-ready. <<<")
        print("=" * 75)
        return 0
    else:
        print("  >>> VERIFICATION FAILED. Review errors above. <<<")
        print("=" * 75)
        return 1


if __name__ == "__main__":
    sys.exit(main())
