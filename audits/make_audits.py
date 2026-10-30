import os

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully wrote {path} ({len(content)} chars)")

audit_content = """# Version 7 Scientific and Experimental Audit Report

**Project Title:** FRFT--Wavelet Spatiotemporal Feature Fusion and 2D Light CNN Architecture for Robust Audio Deepfake Detection  
**Target Submission Format:** IEEE Two-Column Conference Style Manuscript (Single-Sided Printing Format)  
**Authors:** Advit Bhutani & Ansh Singh  
**Faculty Guide:** Dr. Pattabiraman V  
**Institution:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Chennai, India  
**Date of Audit:** October 15, 2026  
**Auditor:** Automated Scientific Reproducibility and Document Quality Audit System  
**Working Directory:** `C:\\Users\\ANSH SINGH\\Desktop\\capstone1\\paper\\version7_single_sided\\`  

---

## 1. Executive Summary & Audit Scope

This document provides a formal, artifact-level verification of all scientific claims, architectural specifications, loss formulations, and empirical metrics presented in the Version 7 research manuscript. Every reported metric has been audited against physical files in `C:\\Users\\ANSH SINGH\\Desktop\\capstone1`, specifically `v7\\best_lcnn2d_v7.pt`, `v7\\v7_results.json`, and `v7\\v7_2d_frft_wavelet_lcnn.py`.

### Primary Audit Verdicts:
1. **Architectural Verification:** The Version 7 classifier is a genuine 2D Convolutional Light CNN (**LCNN-2D**) operating directly on a 3-channel 2D Spatiotemporal Forensic Tensor ($[3 \\times 256 \\times 397]$). It employs 2D Max-Feature-Map (MFM-2D) activations, batch normalization, max pooling, global average pooling, and One-Class Softmax (OC-Softmax) metric learning.
2. **Trainable Parameter Count:** Exactly **170,208 trainable parameters** (state dict footprint: 770,309 bytes, $\\approx$ 680 KB in 32-bit float). This represents a **70.47% parameter reduction** compared to the 1D Version 3 model (576,386 parameters).
3. **Benchmark Verification on ASVspoof 2019 LA:**
   - **Development Set EER:** **3.14%** (3.1397% at Epoch 17)
   - **Evaluation Set EER:** **9.25%** (9.2454% via continuous linear interpolation; **9.26%** on discrete empirical ROC)
   - **Evaluation Set ROC-AUC:** **0.9496** (0.949559)
   - **Evaluation Set Accuracy:** **90.75%** (64,651 / 71,237 correct predictions at EER threshold)
   - **Evaluation Set Balanced Accuracy:** **90.75%**
4. **Reproducibility Guarantee:** All mathematical operators, filterbank equations, and loss definitions have been independently verified against the PyTorch source code.

---

## 2. Comprehensive Evidence & Claim Verification Ledger

| Scientific Claim / Manuscript Item | Primary Source Artifact | Audit Method | Verified Value in Codebase | Audit Status | Manuscript Implementation Action |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Model Classification Paradigm** | `v7\\v7_2d_frft_wavelet_lcnn.py` Lines 258--282 | Direct Inspection | 4-Stage 2D CNN with MFM-2D activations and GAP head | **Verified** | Formally designated as **LCNN-2D** (genuine 2D convolutional network, resolving past 1D dense limitations). |
| **Trainable Parameter Count** | `v7\\best_lcnn2d_v7.pt` state dict inspection | Independent Recomputation | Exactly **170,208 trainable parameters** | **Directly Verified** | Layer parameter breakdown documented in Table II; non-trainable buffers accounted for. |
| **Input Feature Representation** | `v7\\v7_2d_frft_wavelet_lcnn.py` Lines 219--231 | Direct Inspection | 3-Channel 2D Spatiotemporal Tensor $\\mathbf{T}_{\\text{2D}} \\in \\mathbb{R}^{3 \\times 256 \\times 397}$ | **Directly Verified** | Described as multi-channel forensic spectrogram: Ch 1 = FrFT 0.50, Ch 2 = FrFT 0.85, Ch 3 = Wavelet Scalogram. |
| **FrFT Layer Implementation** | `v7\\v7_2d_frft_wavelet_lcnn.py` Lines 150--186 | Direct Inspection | Frame-level vectorized chirp modulation and FFT ($N_{\\text{fft}}=512, H=160$) | **Directly Verified** | ST-FrFT formulated via continuous and discrete chirp-premultiplication Fourier operators. |
| **Wavelet Layer Implementation** | `v7\\v7_2d_frft_wavelet_lcnn.py` Lines 188--217 | Direct Inspection | 64-scale Morlet/Gaussian continuous filterbank convolved along waveform and interpolated to 256 sub-bands | **Directly Verified** | Scalogram formulation mathematically defined with dyadic logarithmic spacing. |
| **Loss Function Formulation** | `v7\\v7_2d_frft_wavelet_lcnn.py` Lines 284--306 | Direct Inspection | One-Class Softmax (OC-Softmax) with $\\alpha=20.0, m_{\\text{real}}=0.90, m_{\\text{fake}}=0.20$ | **Directly Verified** | Explicit margin equations and angular cosine distance formulation documented. |
| **Weight Regularization & Ensembling** | `v7\\v7_2d_frft_wavelet_lcnn.py` Lines 330--344 | Direct Inspection | Exponential Moving Average (EMA) with decay $\\beta=0.995$ | **Directly Verified** | Verified that evaluation metrics are generated from the EMA shadow weights. |
| **Evaluation Set EER** | `v7\\v7_results.json`, `v7_results.png` | Direct Inspection & Recomputation | Continuous Interp: **9.2454%**; Discrete ROC: **9.26%** | **Directly Verified** | Reported as 9.25% (linear interpolation) and 9.26% (nearest ROC point), establishing the sub-10% breakthrough. |
| **Evaluation Set ROC-AUC** | `v7\\v7_results.json` | Direct Inspection | **0.949559** (0.9496) | **Directly Verified** | Reported alongside ROC curves in Figure 5 and Table IV. |
| **Evaluation Set Balanced Accuracy** | `v7\\v7_results.json` | Direct Inspection | **90.7547%** (90.75%) | **Directly Verified** | Formally reported to provide unbiased evaluation under severe 1:8.69 class imbalance. |
| **Development Set EER** | `v7\\v7_results.json` | Direct Inspection | **3.1397%** (3.14%) at Epoch 17 | **Directly Verified** | Documented convergence trajectory across 35 scheduled epochs. |
| **Dataset Partitions & Attacks** | Protocol manifests; `v7_2d_frft_wavelet_lcnn.py` | Direct Inspection | Train: 25,380 (A01--A06); Dev: 24,844 (A01--A06); Eval: 71,237 (A01--A06 known + A07--A19 unseen) | **Directly Verified** | Confirmed 121,461 total corpus trials on ASVspoof 2019 Logical Access. |
| **Class Balancing Strategy** | `v7\\v7_2d_frft_wavelet_lcnn.py` Lines 129--136 | Direct Inspection | `WeightedRandomSampler` with weights inversely proportional to class frequencies ($w_0 = 1/2580, w_1 = 1/22800$) | **Directly Verified** | Confirmed zero gradient starvation for the genuine speech class during backpropagation. |

---

## 3. Mathematical and Signal Processing Formulation Audit

### A. Vectorized Short-Time Fractional Fourier Transform (ST-FrFT)
The ST-FrFT maps a continuous audio signal $x(t)$ onto an intermediate time-frequency plane rotated by angle $\\phi = \\alpha \\frac{\\pi}{2}$ using a sliding analysis window $g(t)$:
$$\\mathcal{F}_{\\alpha}\\{x\\}(t, u) = \\int_{-\\infty}^{\\infty} x(\\tau) g(\\tau - t) K_\\alpha(\\tau, u) \\, d\\tau$$
where the transformation kernel $K_\\alpha(\\tau, u)$ for $\\alpha \\notin 2\\mathbb{Z}$ is defined as:
$$K_\\alpha(\\tau, u) = \\sqrt{\\frac{1 - j \\cot \\phi}{2\\pi}} \\exp \\left( j \\frac{\\tau^2 + u^2}{2} \\cot \\phi - j \\frac{\\tau u}{\\sin \\phi} \\right)$$

In the discrete implementation (`VectorizedSTFrFT`), for frame size $N=512$, hop length $H=160$, and rotation orders $\\alpha \\in \\{0.50, 0.85\\}$, the computation is efficiently vectorized on the GPU using chirp premultiplication:
$$X_\\alpha[m, k] = \\frac{1}{\\sqrt{N}} \\sum_{n=0}^{N-1} \\Big( x[mH + n] w[n] \\cdot \\exp \\left( j \\pi \\tan(\\phi / 2) \\frac{n^2}{N} \\right) \\Big) \\exp \\left( -j \\frac{2\\pi k n}{N} \\right)$$
The resulting channel representation computes the logarithmic energy magnitude:
$$\\mathbf{S}_\\alpha[k, m] = \\ln \\big( 1 + |X_\\alpha[m, k]| \\big), \\quad k \\in [0, 255], \\; m \\in [0, 396]$$

### B. Multi-Scale Continuous Wavelet Scalogram Filterbank
To resolve localized high-frequency transients and sub-band phase discontinuities, the continuous wavelet scalogram decomposes the raw waveform across $S=64$ dyadically spaced scales $s_i \\in [1.0, 64.0]$:
$$\\psi_s(t) = \\frac{1}{\\sqrt{s}} \\exp \\left( -\\frac{t^2}{2s^2} \\right) \\cos \\left( 5.0 \\frac{t}{s} \\right)$$
Convolving $x(t)$ with filter bank $\\{\\psi_{s_i}\\}_{i=1}^{64}$ with stride $H=160$ yields the scalogram:
$$\\mathbf{W}[s_i, m] = \\ln \\big( 1 + |(x * \\psi_{s_i})[mH]| \\big)$$
Bilinear interpolation projects the 64 scale bands onto 256 uniform pseudo-frequency bins, matching the dimensions of the ST-FrFT channels:
$$\\mathbf{T}_{\\text{2D}} = \\big[ \\mathbf{S}_{0.50}; \\; \\mathbf{S}_{0.85}; \\; \\mathbf{W} \\big] \\in \\mathbb{R}^{3 \\times 256 \\times 397}$$

### C. 2D Max-Feature-Map (MFM-2D) Activation
Unlike standard ReLU or GELU activations that introduce fixed monotonic thresholding, the Max-Feature-Map activation acts as a competitive feature filter. For an intermediate convolutional feature map $\\mathbf{U} \\in \\mathbb{R}^{2C \\times H \\times W}$, MFM-2D partitions the channels into two halves and computes the element-wise maximum:
$$\\text{MFM}(\\mathbf{U})_{c, h, w} = \\max \\big( \\mathbf{U}_{c, h, w}, \\; \\mathbf{U}_{c+C, h, w} \\big), \\quad c \\in \\{1, \\dots, C\\}$$
This halves the channel dimension while suppressing low-energy background noise and retaining high-amplitude forensic artifacts without requiring extra parameters.

### D. One-Class Softmax (OC-Softmax) Loss Formulation
To optimize feature embeddings for biometric anomaly detection where spoof attacks exhibit open-set variety, OC-Softmax defines an angular decision boundary around a single learnable bona fide center vector $\\mathbf{w} \\in \\mathbb{R}^{128}$:
$$\\mathcal{L}_{\\text{OC}} = \\frac{1}{B} \\sum_{i=1}^B \\ln \\Big( 1 + \\exp \\big( \\alpha \\cdot z_i \\big) \\Big)$$
where:
$$z_i = \\begin{cases} m_{\\text{real}} - \\cos(\\theta_i), & \\text{if } y_i = 0 \\text{ (bona fide)} \\\\ \\cos(\\theta_i) - m_{\\text{fake}}, & \\text{if } y_i = 1 \\text{ (spoof)} \\end{cases}$$
and $\\cos(\\theta_i) = \\frac{\\mathbf{e}_i \\cdot \\mathbf{w}}{\\|\\mathbf{e}_i\\|_2 \\|\\mathbf{w}\\|_2}$. The hyperparameter configuration was verified as:
$$\\alpha = 20.0, \\quad m_{\\text{real}} = 0.90, \\quad m_{\\text{fake}} = 0.20$$
During inference, the score assigned to an utterance is the normalized cosine similarity $S(\\mathbf{x}) = \\cos(\\theta)$, where higher scores indicate a higher probability of genuine human speech.

---

## 4. Layer-by-Layer Architectural Breakdown of LCNN-2D

Audited against `v7\\best_lcnn2d_v7.pt` state dict:

| Stage / Component | Operation Description | Input Dimensions | Output Dimensions | Kernel / Stride / Pad | Parameter Count | Trainable Weights & Biases |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage 1: ConvBlock** | Conv2D $\\to$ MFM $\\to$ BN $\\to$ MaxPool | $3 \\times 256 \\times 397$ | $32 \\times 128 \\times 198$ | $5 \\times 5$, pad 2, pool $2 \\times 2$ | 4,864 | Conv: $64 \\times 3 \\times 5 \\times 5 = 4,800$; BN: $32 \\times 2 = 64$ |
| **Stage 2: ConvBlock** | Conv2D $\\to$ MFM $\\to$ BN $\\to$ MaxPool | $32 \\times 128 \\times 198$ | $48 \\times 64 \\times 99$ | $3 \\times 3$, pad 1, pool $2 \\times 2$ | 27,744 | Conv: $96 \\times 32 \\times 3 \\times 3 = 27,648$; BN: $48 \\times 2 = 96$ |
| **Stage 3: ConvBlock** | Conv2D $\\to$ MFM $\\to$ BN $\\to$ MaxPool | $48 \\times 64 \\times 99$ | $64 \\times 32 \\times 49$ | $3 \\times 3$, pad 1, pool $2 \\times 2$ | 55,424 | Conv: $128 \\times 48 \\times 3 \\times 3 = 55,296$; BN: $64 \\times 2 = 128$ |
| **Stage 4: ConvBlock** | Conv2D $\\to$ MFM $\\to$ BN (no pool) | $64 \\times 32 \\times 49$ | $64 \\times 32 \\times 49$ | $3 \\times 3$, pad 1, pool None | 73,856 | Conv: $128 \\times 64 \\times 3 \\times 3 = 73,728$; BN: $64 \\times 2 = 128$ |
| **Pooling & Head** | AdaptiveAvgPool2d $\\to$ Drop(0.3) $\\to$ Linear | $64 \\times 32 \\times 49$ | 128 | GAP $\\to (1, 1)$, Linear(64, 128) | 8,192 | Linear weight: $128 \\times 64 = 8,192$; no bias |
| **Metric Learning** | OC-Softmax Center Vector $\\mathbf{w}$ | 128 | 1 | Unit sphere parameter | 128 | Weight vector: $128 \\times 1 = 128$ |
| **Total Trainable** | **LCNN-2D + OC-Softmax** | -- | -- | -- | **170,208** | **Full Model Footprint: $\\approx$ 680 KB** |

---

## 5. Methodological Rigor and Protocol Boundaries

1. **ROC EER vs. Official Bosaris min t-DCF:** The evaluation metric reported throughout this manuscript is the continuous linear interpolation Equal Error Rate (and discrete ROC EER) computed directly from utterance-level cosine similarity scores. The official ASVspoof consortium evaluation uses the Bosaris toolkit to calculate the minimum normalized tandem Detection Cost Function (min t-DCF) alongside pooled EER. In the manuscript, we explicitly disclose that our reported metrics represent internal ROC EER benchmarks.
2. **Ablation Comparability:** Historical single-lens baselines (FrFT-only at 13.28% and WPD-only at 25.66%) were trained on 1D statistical vectors for 25 epochs. The Version 7 system operates on continuous 2D tensors and trained for 35 scheduled epochs with OneCycleLR and EMA. In Section V, we transparently state that this comparison reflects an architectural transition from static 1D moments to dynamic 2D spatiotemporal modeling, rather than a strictly isolated single-variable ablation.
3. **Absence of Data Leakage:** In Version 7, feature representations are generated dynamically on the GPU from raw audio waveforms during training and evaluation. No normalization scalers or global parameters are fitted across split boundaries.
"""

write_file(r'paper\version7_single_sided\version7_scientific_audit.md', audit_content)
