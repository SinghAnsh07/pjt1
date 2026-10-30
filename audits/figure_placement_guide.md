# Figure and Table Placement Guide

**Project Title:** FRFT–Wavelet Feature Fusion for Audio Deepfake Detection Using a Lightweight Competitive Neural Network  
**Target Format:** IEEE Two-Column Conference Style Manuscript  
**Companion Files:** `research_paper.tex`, `research_paper_draft.docx`, `research_audit.md`  

This guide provides the publication mapping for every figure, table, diagram, and algorithm box across the manuscript. Each entry details its scientific purpose, target manuscript section, insertion anchor, source artifact, formatting specifications, and full academic caption.

---

## 1. Visual Figures Mapping

| Figure # | Title & Description | Source Artifact | Target Section & Location | Column Span | Full Academic Caption | Key Visual Elements |
| :---: | :--- | :--- | :--- | :---: | :--- | :--- |
| **Fig. 1** | Overall System Architecture Framework | `fig1_framework.png` (Generated at 300 DPI) | Section III-A (Methodology: Overview) | Double Column (Top of Page) | **Fig. 1:** Architectural pipeline of the proposed audio deepfake detection framework. The raw speech signal is standardized (16 kHz, 6.0 s), processed through dual signal lenses—Fractional Fourier Transform (FrFT, 448 descriptors) and Wavelet Packet Decomposition (WPD, 256 descriptors)—fused into a 704-dimensional feature vector, standardized via `StandardScaler`, and classified using an MFM-based Deep Competitive Neural Network (MFM-DNN) equipped with Max-Feature-Map competitive units, batch normalization, dropout, and a residual skip connection. | Raw audio $\to$ Preprocessing (16 kHz, 6.0 s) $\to$ Dual Branches (FrFT & WPD) $\to$ 704-D Feature Fusion $\to$ StandardScaler $\to$ 4-Block MFM-DNN (576,386 params) + Skip $\to$ Softmax Logits. |
| **Fig. 2** | Multi-Domain Signal Representations (Real vs. Spoof) | `results_v1/audio_visualization.png` | Section III-C / III-D (Lens Explanations) | Double Column (Top of Page) | **Fig. 2:** Multi-domain signal representations comparing bona fide human speech (`LA_T_1138215`, top row) and synthetic spoofed speech (`LA_T_1004644`, bottom row). Columns from left to right display the standardized time-domain waveform (6.0 s duration), the log-magnitude Fourier spectrum (0–8 kHz), and the FrFT magnitude projection at fractional order $\alpha = 0.50$. | 6-panel grid: Waveforms (left), FFT log-magnitude spectra (middle), and FrFT $\alpha=0.5$ distributions (right) highlighting micro-transient disparities. |
| **Fig. 3** | Training Dynamics and Optimization Convergence | `results_v1/training_curves.png` | Section V-A (Training & Validation Behavior) | Double Column (Top of Page) | **Fig. 3:** Optimization dynamics of the champion MFM-DNN network across 40 training epochs on the ASVspoof 2019 LA benchmark. Left: Cross-Entropy training loss (with class weighting and label smoothing $\epsilon=0.10$). Center: Development set Equal Error Rate (Dev EER, reaching optimal threshold of 6.00% at epoch 38). Right: Development set accuracy (reaching 90.29% at epoch 38). The dashed gold lines indicate the finalized evaluation benchmark operating performance (Eval EER = 10.67%, Eval Accuracy = 85.61%, Balanced Accuracy = 90.26%). | 3-panel plot showing smooth loss reduction ($0.90 \to 0.51$), Dev EER convergence ($22.03\% \to 6.00\%$), and Dev Acc trajectory ($58.29\% \to 90.29\%$). |
| **Fig. 4** | Feature-Ablation Equal Error Rate Comparison | `results_v1/ablation_results.png` (Left Panel) or `feature_comparison_chart.png` | Section V-C (Feature-Ablation Analysis) | Single Column (Column Width) | **Fig. 4:** Comparative Equal Error Rate (EER, %) across single-lens and fused feature representations on the ASVspoof 2019 LA evaluation set. Lower EER denotes superior forensic discrimination. Fusing FrFT and WPD yields an absolute EER reduction of 2.61% points (19.65% relative improvement) over the strongest standalone baseline. | Clear bar chart comparing WPD only (25.66%), LFCC baseline (16.70%), FrFT only (13.28%), and Fused FrFT+WPD (10.67%). |
| **Fig. 5** | Angular FrFT Magnitude Distributions Across Fractional Orders | `results_v1/ablation_results.png` (Right Panel) | Section V-D (Signal Analysis & Discussion) | Single Column (Column Width) | **Fig. 5:** Magnitude spectrum of the Fractional Fourier Transform across fractional orders $\alpha \in \{0.10, 0.50, 0.90\}$ for bona fide (blue curves) and spoofed (orange dashed curves) speech segments. Subtle phase and chirp modulation discrepancies become visibly distinct along intermediate oblique axes ($\alpha = 0.50, 0.90$) that remain suppressed under orthogonal Fourier analysis. | Multi-curve line plot showing distinct separation between genuine and synthetic voice trajectories at non-orthogonal angles. |

---

## 2. Experimental Tables Mapping

| Table # | Table Title & Purpose | Source Artifact | Target Section | Column Span | Table Column Schema |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **TABLE I** | ASVspoof 2019 Logical Access Benchmark Composition | `capproject_v1.ipynb` (Cells 4, 5, 8), Protocol manifests | Section IV-A (Dataset & Splits) | Single Column | `Split`, `Utterances`, `Bona fide`, `Spoof`, `Prevalence Ratio`, `Attack Algorithms` (Train: A01–A06, Dev: A01–A06, Eval: A01–A06 known + A07–A19 unseen). |
| **TABLE II** | MFM-DNN Architectural Specifications & Layer Dimensions | `capproject_v1.ipynb` (Cell 10), PyTorch state dict | Section IV-C (Model Implementation) | Single Column | `Block / Layer`, `Input Dimension`, `Output Dimension`, `Activation / Regularization`, `Trainable Parameters` (Total: 576,386). |
| **TABLE III** | Comprehensive Feature Ablation Performance Benchmarks | `capproject_v1.ipynb` (Cell 13 output) | Section V-C (Ablation Analysis) | Single Column | `Feature Configuration`, `Input Dimensions`, `Eval EER (%)`, `Eval Accuracy (%)`, `Training Epochs`, `EER Delta vs. Fused` (FrFT+WPD: 10.67%, FrFT: 13.28%, LFCC: 16.70%, WPD: 25.66%). |
| **TABLE IV** | Experimental Progression Across System Versions (v1 $\to$ v2 $\to$ v3) | `capstoneproject2.ipynb`, `capproject.ipynb`, `capproject_v1.ipynb` | Section V-E (Version Analysis) | Double Column | `Version`, `Feature Vector`, `Backbone Architecture`, `Objective Function`, `Regularization`, `Dev EER (%)`, `Eval EER (%)`, `Eval Acc (%)`, `Primary Empirical Finding`. |
| **TABLE V** | Detailed Classification Performance on ASVspoof 2019 LA Evaluation Set | Independent recomputation script, `best_lcnn_v3.pt` | Section V-B (Evaluation Performance) | Single Column | `Class`, `Precision`, `Recall (Sensitivity / Specificity)`, `F1-Score`, `Support` (Spoof: 0.9947 / 0.8440 / 0.9132 / 63,882; Bona fide: 0.4151 / 0.9613 / 0.5798 / 7,355; Overall Acc: 85.61%; Balanced Acc: 90.26%). |
| **TABLE VI** | Structured Summary of Reviewed Literature in Audio Deepfake Detection | Section II primary sources & state of the art | Section II-F (Related Work Summary) | Double Column (Full Page Width) | `Ref`, `Author(s) & Year`, `Feature Representation / Method`, `Classifier Architecture`, `Reported Benchmark / EER`, `Key Finding or Scientific Gap Addressed`. |

---

## 3. Algorithm Box Mapping

| Algorithm # | Algorithm Title | Target Section | Purpose & Content |
| :---: | :--- | :--- | :--- |
| **Algorithm 1** | Dual-Lens Feature Extraction and MFM-DNN Optimization Pipeline | Section III-H (Methodology: Training Procedure) | Formal pseudocode detailing temporal framing (6.0 s), FrFT multi-angle chirp projection (7 orders), WPD level-5 sub-band slicing (32 sub-bands), `stats8` computation, StandardScaler normalization, batch sampling with inverse frequency weights, MFM competitive forward pass, and early stopping on Dev EER. |

---

## 4. Manuscript Formatting and Placement Rules

1. **Two-Column Balancing:**
   - Double-column floats (`figure*` and `table*` in LaTeX) are positioned strictly at the top of pages (`[t!]`) to prevent mid-page fragmentation.
   - Single-column figures (`figure` and `table`) are scaled to `\columnwidth` (3.45 inches / 88 mm) with legible typography (minimum 8 pt after scaling).
2. **First In-Text Reference Precedence:**
   - No figure or table appears before its initial reference in the text. Every figure is explicitly cited (`Fig. 1`, `Fig. 2`, etc.) and interpreted in the immediately preceding or adjacent paragraph.
3. **High-DPI Embeddings in Word (DOCX):**
   - High-resolution versions of `fig1_framework.png`, `training_curves.png`, `ablation_results.png`, and `audio_visualization.png` are embedded directly into `research_paper_draft.docx` with centered captions and formal IEEE table borders.
