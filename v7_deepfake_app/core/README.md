# V7 Audio Deepfake Inference Engine (`core/`)

## 1. Overview and Architecture

The `core` package provides a standalone, production-ready inference engine for detecting synthetic speech, voice clones, and audio deepfakes using the verified **Version 7 (V7)** research model.

The architecture combines:
1. **Differentiable 2D Front-End Feature Extractor (`FrFTWavelet2DExtractor`):**
   - **Channel 0:** Short-Time Fractional Fourier Transform (ST-FrFT) with fractional order $\alpha = 0.50$ ($45^\circ$).
   - **Channel 1:** ST-FrFT with fractional order $\alpha = 0.85$ ($76.5^\circ$).
   - **Channel 2:** Continuous Morlet-style Wavelet Scalogram across 64 scales.
   - **Output Tensor:** Spatiotemporal tensor of shape `[B, 3, 256, 397]`.
2. **2D Light CNN Backbone (`LCNN2D`):**
   - 4-stage convolutional backbone with 2D Max-Feature-Map (MFM-2D) activations.
   - Stage 1: Conv2D ($3 \to 64$, $5\times 5$, $p=2$) $\to$ MFM-2D ($64 \to 32$) $\to$ BN $\to$ MaxPool2D ($2\times 2$) $\to$ $[B, 32, 128, 198]$.
   - Stage 2: Conv2D ($32 \to 96$, $3\times 3$, $p=1$) $\to$ MFM-2D ($96 \to 48$) $\to$ BN $\to$ MaxPool2D ($2\times 2$) $\to$ $[B, 48, 64, 99]$.
   - Stage 3: Conv2D ($48 \to 128$, $3\times 3$, $p=1$) $\to$ MFM-2D ($128 \to 64$) $\to$ BN $\to$ MaxPool2D ($2\times 2$) $\to$ $[B, 64, 32, 49]$.
   - Stage 4: Conv2D ($64 \to 128$, $3\times 3$, $p=1$) $\to$ MFM-2D ($128 \to 64$) $\to$ BN $\to$ $[B, 64, 32, 49]$.
   - Global Average Pooling $\to$ Flatten $\to$ Dropout(0.3) $\to$ Linear ($64 \to 128$).
   - **Output Embedding:** 128-dimensional representation $\mathbf{e} \in \mathbb{R}^{128}$.
3. **One-Class Softmax Metric Head (`OCSoftmax`):**
   - Evaluates the cosine similarity against the learned bona fide hyperspherical center $\mathbf{w} \in \mathbb{R}^{128}$.
   - **Decision Score:** $S(x) = \frac{\mathbf{e} \cdot \mathbf{w}}{\|\mathbf{e}\|_2 \|\mathbf{w}\|_2} \in [-1.0, 1.0]$.
   - **Total Network Parameters:** 170,336 parameters (170,208 in LCNN-2D backbone, 128 in OC-Softmax center $\mathbf{w}$, 0 in feature extractor).
   - **Registered Buffers:** 18,340 elements (512 Hann window, 1,024 complex chirps, 16,384 wavelet filters, 420 BatchNorm tracking buffers).

---

## 2. Model Loading Procedure

Model loading is handled by `core.checkpoint_loader.load_model`:

```python
from core.checkpoint_loader import load_model

model, metadata = load_model("path/to/best_lcnn2d_v7.pt", device="auto")
```

### Safety and Security Features:
- **`weights_only=True`:** Deserializes exclusively tensor weights and standard primitives, preventing arbitrary code execution.
- **Strict Verification (`strict=True`):** All 30 state dictionary keys must match exactly. Missing or unexpected keys raise `CheckpointIncompatibleError`.
- **Parameter Freezing:** Sets `model.eval()` and disables gradient tracking (`requires_grad=False`) for all parameters.
- **Cryptographic Hashing:** Computes and verifies SHA-256 hash upon loading.

---

## 3. Waveform Preprocessing Contract

Implemented in `core.preprocessor.preprocess_waveform`:

```python
from core.preprocessor import preprocess_waveform

audio_tensor, prep_result = preprocess_waveform(raw_audio, sample_rate=44100)
```

### Rules & Normalization Pipeline:
1. **Validation:** Checks that audio is non-empty, finite (no NaN/Inf), and 1D or 2D.
2. **Channel Downmix:** Stereo or multichannel waveforms are downmixed to mono via channel averaging: `x = np.mean(x, axis=0)`.
3. **Polyphase Resampling:** If `sample_rate != 16000`, converts to 16,000 Hz using `scipy.signal.resample_poly` with rational integer factors `up` and `down`.
4. **Length Normalization (Target = 64,000 samples / 4.0 seconds):**
   - **Short audio (< 4.0s):** Circular wrap padding (`np.pad(..., mode='wrap')`) matching the ASVspoof 2019 training protocol.
   - **Long audio (> 4.0s):** Single-window truncation to the first 64,000 samples, issuing a documented warning that audio beyond 4.0s was omitted.
5. **Output Tensor:** Floating-point tensor `[1, 64000]` in range $[-1.0, 1.0]$.

---

## 4. Score Semantics and Biometric Decision Policy

Implemented in `core.decision_engine.DecisionEngine`:

- **Score Direction (`HIGHER_IS_REAL`):**
  - High positive cosine score ($S(x) \to +1.0$) indicates close proximity to the bona fide center $\mathbf{w}$ (Authentic).
  - Low or negative cosine score ($S(x) \to -1.0$) indicates orthogonal or anti-aligned projection (Synthetic/Spoof).
- **Candidate Operational Threshold:** $\tau^* = 0.4820$.
  - Calibrated on the official ASVspoof 2019 LA evaluation benchmark (71,237 utterances), achieving 9.25% EER and 90.75% balanced accuracy.
  - **Status:** Labeled `CANDIDATE_UNVALIDATED` because it has not been validated on unconstrained, in-the-wild audio.
- **Decision Rule:**
  - $S(x) \ge 0.4820 \implies$ `DecisionLabel.REAL` (Bona Fide)
  - $S(x) < 0.4820 \implies$ `DecisionLabel.FAKE` (Synthetic / Spoof)
- **No Fabricated Confidence:** The engine refuses to map raw distance to an arbitrary percentage ("95% sure"). It reports raw score, distance from threshold, and operational caveats.

---

## 5. Public Inference API

### Stateful Inference (Recommended for Repeated Calls):

```python
from core.inference import V7InferenceEngine

# Initialize engine once (keeps model in memory)
engine = V7InferenceEngine(checkpoint_path="v7/best_lcnn2d_v7.pt", device="auto")

# Run inference on in-memory waveform
result = engine.predict_waveform(waveform_array, sample_rate=16000, filename="sample.wav")

print("Status:", result.status)
print("Score:", result.model_score)
print("Prediction:", result.decision.prediction.value)
print("Explanation:", result.decision.explanation)
print("Latency:", result.timing.total_ms, "ms")
```

### Stateless Single-Call Inference:

```python
from core.inference import predict_waveform

result = predict_waveform(waveform_array, sample_rate=16000, checkpoint_path="v7/best_lcnn2d_v7.pt")
```

---

## 6. How to Run the Verification Script and Tests

### Run the Verification Smoke Test:
```bash
python v7_deepfake_app/scripts/verify_inference.py
```

### Run the Automated Test Suite (Pytest):
```bash
python -m pytest v7_deepfake_app/tests -v
```

---

## 7. Known Limitations

1. **Single-Window Policy:** Currently evaluates only the initial 4.0 seconds (64,000 samples) of long audio files. Multi-window sliding timeline analysis is reserved for subsequent phases.
2. **Video Ingestion:** System `ffmpeg` is not present on the host `PATH`. Video container demuxing (.mp4, .mkv) requires audio pre-extraction.
3. **Domain Shift:** Threshold $\tau^* = 0.4820$ was derived on ASVspoof 2019 LA (16 kHz studio recordings). Lossy compression (e.g., telephone codecs, aggressive WhatsApp MP3 encoding) may shift scores lower.
