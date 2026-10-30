# Originality Audit and Attribution Verification Report

**Project Title:** FRFT–Wavelet Feature Fusion for Audio Deepfake Detection Using a Lightweight CNN  
**Authors:** Advit Bhutani, Ansh Singh  
**Faculty Guide:** Dr. Pattabiraman V  
**Institution:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Chennai, India  
**Date of Audit:** October 10, 2026  
**Audit Status:** Completed  
**External Similarity Metric:** **NOT VERIFIED** (No institutional plagiarism/similarity software endpoint available in environment)

---

## 1. Executive Summary & Verification Disclaimer

### 1.1 Similarity Score Disclosure
> [!IMPORTANT]
> **SIMILARITY CHECKER STATUS: NOT VERIFIED**  
> In compliance with strict academic integrity standards, no unauthorized, synthetic, or heuristic similarity score is fabricated. The local development environment lacks an integrated institutional plagiarism/similarity detection tool (such as Turnitin, iThenticate, or Urkund). Consequently, while extensive linguistic restructuring, conceptual rebuilding, and literature deduplication have been executed across all sections of the manuscript, the official percentage similarity score must be formally determined by uploading `research_paper_final.pdf` or `research_paper_draft.docx` to the university's licensed similarity checking platform.

### 1.2 Objective of the Originality Audit
The purpose of this audit is to systematically review and eliminate common vectors of textual similarity and academic unoriginality:
1. **Elimination of Survey Clichés:** Removing stock introductory phrases, generic societal threat statements, and boilerplate deepfake summaries commonly flagged in automated cross-paper similarity databases.
2. **First-Principles Conceptual Rebuilding:** Replacing generic textbook definitions of mathematical transforms (FrFT, WPD, MFM) with domain-specific explanations grounded in speech production physics and neural vocoder synthesis mechanics.
3. **De-duplication of Internal Phrasing:** Streamlining repetitive verbatim sentences (such as recurring architectural parameter counts and class imbalance ratios) across different manuscript sections.
4. **Attribution and Bibliographic Integrity:** Verifying that all 25 cited literature sources correspond to genuine peer-reviewed publications with accurate metadata and relevant contextual integration.
5. **Preservation of Empirical Evidence:** Ensuring that every experimental metric, feature dimension, hyperparameter value, and methodological caveat established during the codebase audit is preserved with 100% precision.

---

## 2. Section-by-Section Originality & Attribution Audit

### 2.1 Abstract
* **Initial Problematic Text:**  
  *"The rapid proliferation of deep generative acoustic models---including diffusion vocoders, neural autoregressive architectures, and few-shot voice conversion algorithms---has escalated the threat of synthetic voice fraud, corporate impersonation, and biometric authentication bypass."*  
  *(Flagged as generic academic cliché widely found in modern speech synthesis introductions).*
* **Revised Original Prose:**  
  *"Algorithmic speech synthesis has advanced to where neural vocoders reconstruct acoustic waveforms that match human vocal timbre and conversational prosody, presenting substantial security challenges to biometric voice verification. While these synthesized waveforms achieve perceptual realism, their mathematical reconstruction via discrete upsampling operations and neural codebook mappings introduces micro-acoustic phase irregularities and anomalous spectral dispersion across upper harmonic bands. Conventional forensic pipelines relying on orthogonal Short-Time Fourier Transform (STFT) magnitudes or Mel-Frequency Cepstral Coefficients (MFCCs) struggle to expose these cues because of rigid time-frequency uncertainty trade-offs and aggressive high-frequency psychoacoustic compression."*
* **Underlying Rationale:** Reframes the motivation from abstract societal alarmism to the precise physical and mathematical mechanisms of vocoder artifact generation versus biological human phonation.
* **Empirical Integrity Check:** Preserves all verified project metrics: 448 FrFT descriptors, 256 WPD descriptors, 704 fused dimensions, standard scaling, MFM-DNN with 576,386 parameters (~2.3 MB), 121,461 utterances on ASVspoof 2019 LA, 71,237 evaluation utterances spanning 13 zero-day attacks, 10.67% EER at $\tau^* = 0.8362$ on $P(\text{bonafide})$, 85.61% accuracy, 90.26% Balanced Accuracy, 96.13% bona fide recall, 84.40% spoof recall, and ablation results (13.28% FrFT alone, 16.70% LFCC, 25.66% WPD alone).

---

### 2.2 Section I: Introduction
* **Initial Problematic Text:**  
  *"Recent breakthroughs in deep neural speech synthesis and voice conversion (VC) have democratized the generation of high-fidelity artificial voices... audio deepfakes have evolved from a theoretical curiosity into an acute security threat. Malicious adversaries exploit synthetic voice replicas to orchestrate telephone-based social engineering (vishing)..."*  
  *(Flagged for standard buzzwords: "democratized the generation", "theoretical curiosity into an acute security threat", "orchestrate telephone-based social engineering").*
* **Revised Original Prose:**  
  Rebuilt from the physics of acoustic phonation:
  - Contrasts continuous lung-pressure aerodynamic vocal fold vibrations and continuous vocal-tract resonance cavities against discrete algorithmic operations (transposed convolutional upsampling, periodic source filters, discrete codebooks).
  - Explicitly formalizes the two core signal-processing bottlenecks:
    1. *The Heisenberg-Gabor Uncertainty Constraint:* Rigid orthogonal projection ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$) enforcing a temporal-spectral resolution trade-off that smears fast non-stationary vocoder chirp modulations across frames.
    2. *Psychoacoustic High-Frequency Compression:* Non-linear filterbanks (e.g., MFCCs) designed for human hearing compress frequencies above 4 kHz, discarding the exact octaves where neural aliasing and harmonic mismatches reside.
  - Rephrases the 5 primary contributions into structured, technically rigorous bullet points without promotional hyperbole.

---

### 2.3 Section II: Related Work & Literature Summary
* **Initial Problematic Text:**  
  Standard boilerplate survey paragraphs summarizing ASVspoof 2019 LA, LFCC, FrFT, WPD, and LCNN with repetitive phrasing.
* **Revised Original Prose:**  
  - Subsections A through F rewritten with analytical depth:
    - Analyzes how ASVspoof 2019 LA evaluates zero-shot generalization by withholding 13 distinct generative attack algorithms (A07–A19).
    - Contrasts LFCC with STFT phase limitations, noting that linear filterbanks still discard phase dynamics.
    - Contextualizes FrFT as an oblique coordinate rotation in the Wigner-Ville distribution, exposing linear chirp trajectories that cancel out under standard 90° Fourier projections.
    - Formulates WPD as a balanced dyadic tree decomposing both approximation and detail subspaces, capturing high-frequency sub-band entropy.
    - Analyzes MFM activation as a dynamic competitive feature selector that suppresses acoustic noise without inflating parameter counts.
    - Contrasts our compact representation against heavy foundation models (>300M parameters, e.g., WavLM/HuBERT) and complex graph networks (AASIST, RawNet2).
  - **Table I (Literature Summary):** Completely updated with crisp, differentiated analytical columns (`Key Finding or Scientific Gap Addressed`), directly highlighting how each referenced paper relates to the proposed system.

---

### 2.4 Section III: Proposed Methodology
* **Mathematical Integrity:**
  - Preserved continuous FrFT integral (Eq. 2) and discrete Bailey-Swarztrauber chirp-modulated FFT algorithm (Eq. 4).
  - Preserved continuous transformation kernel (Eq. 3), reformatted using compact exponent notation to ensure zero overfull box margin violations in IEEE two-column format.
  - Preserved WPD dyadic convolution formulas for approximation and detail nodes (Eqs. 7–8).
  - Preserved exact mathematical definitions for all 8 statistical descriptors (`stats8`): Mean, Standard Deviation, Max, Min, Q1, Q3, Sub-band Energy, and Spectral Entropy (Eqs. 11–16).
  - Preserved MFM activation formula (Eq. 18) and identity residual skip connection: $\mathbf{h}_2 = \text{Drop}_2(\text{BN}_2(\text{MFM}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2))) + \mathbf{h}_1$.
  - Preserved class-weighted Cross-Entropy objective with label smoothing $\epsilon=0.10$ and inverse class weights ($w_0 = 0.557, w_1 = 4.919$).
* **Originality Improvements:**
  - Surrounding expository text rebuilt from scratch to explain the signal-processing rationale behind each discrete implementation decision (e.g., why magnitude distributions $|X_\alpha[k]|$ were analyzed across 8 chunks instead of noisy raw phase angles, and why natural dyadic tree ordering was preserved).
  - Algorithm 1 redesigned with clean, unique pseudocode layout.

---

### 2.5 Section IV: Experimental Setup
* **Dataset Composition:** Preserved exact splits (Train: 25,380; Dev: 24,844; Eval: 71,237; Total: 121,461) and attack algorithm assignments (A01–A06 known vs A07–A19 unseen).
* **Metric Clarification:** Distinguishes standard continuous ROC Equal Error Rate on $P(\text{bonafide})$ (10.67% at $\tau^* = 0.8362$) from the official ASVspoof Bosaris toolkit minimum tandem Detection Cost Function ($\min \text{t-DCF}$).
* **Class Imbalance Rigor:** Explains the 1:8.69 class imbalance in the evaluation set, demonstrating that a trivial "spoof-all" baseline yields 89.68% accuracy with 0% sensitivity, justifying the use of Balanced Accuracy (90.26%).
* **Reproducibility Details:** Confirms software environment (Python 3.10, PyTorch 2.1, PyWavelets 1.4, librosa 0.10, scikit-learn 1.2), serialized `.npy` feature matrices (~462 MB), and saved checkpoint `best_lcnn_v3.pt` (~2.3 MB).

---

### 2.6 Section V: Results and Discussion
* **Training Dynamics:** Documents convergence from Epoch 1 (loss 0.9018, Dev EER 22.03%) to optimal Epoch 38 (Dev EER 6.00%, Dev Accuracy 90.29%).
* **Evaluation Performance:** Validates per-class classification report, precision, recall, and F1-scores.
* **Confusion Matrix Analysis:** Confirms 53,919 true negatives (84.40% spoof recall), 9,963 false positives (15.60% FAR), 285 false negatives (3.87% FRR), and 7,070 true positives (96.13% bona fide recall). Explains operational security calibration from $\tau=0.50$ to $\tau^*=0.8362$.
* **Component Ablation Benchmarks:**
  - WPD alone (256 dims, 25 epochs): 25.66% EER, 41.52% Acc
  - LFCC baseline (240 dims, 25 epochs): 16.70% EER, 48.61% Acc
  - FrFT alone (448 dims, 25 epochs): 13.28% EER, 84.27% Acc
  - Fused champion (704 dims, 40 epochs): 10.67% EER, 85.61% Acc
  - *Honest Disclosure:* Notes that because the champion model trained for 40 epochs while single-lens baselines trained for 25 epochs, this reflects positive empirical synergy rather than strictly isolated causal proof.
* **Diagnostic of Version 2 Overfitting:** Rebuilt the diagnostic analysis into a case study on deepfake feature engineering:
  1. *Acoustic Fingerprint Memorization:* Concatenating LFCC and spectral contrast (996 dimensions) memorized vocoder-specific spectral envelopes of training attacks (A01–A06), driving apparent Dev EER to 0.05% but causing test EER to collapse to 13.55% (40.27% accuracy).
  2. *Mixup Boundary Distortion:* Linear convex combinations ($\tilde{\mathbf{x}} = \lambda \mathbf{x}_1 + (1-\lambda)\mathbf{x}_2$) blurred subtle phase anomalies.
  3. *Focal Loss Skew:* Concentrated gradient updates on ambiguous training outliers.
* **Version 3 Correction Disclosure:** Explicitly discloses that slicing the precomputed matrices to columns `0:704` was decided post-hoc based on diagnosing Version 2's failure on the evaluation set.

---

### 2.7 Section VI: Limitations and Threats to Validity
The seven limitations were rephrased with enhanced academic precision:
1. *Single-Corpus Scope:* Exclusive reliance on ASVspoof 2019 LA; requirement for cross-dataset validation under lossy codecs and acoustic reverberation.
2. *Fixed Transform Parameters:* Empirically fixed $\alpha$ angles and $L=5$ decomposition depth rather than end-to-end differentiable learning.
3. *Ablation Training Duration Differences:* Asymmetric training budgets (25 vs 40 epochs).
4. *Operational FAR at Default Threshold:* 15.60% false acceptance rate at $\tau=0.50$ requiring calibration for high-security applications.
5. *Iterative Post-Hoc Feature Slicing:* Feature selection informed by evaluation diagnostics.
6. *Dense Linear Classifier:* Operating on 1D statistical summary vectors without 2D spatial convolution kernels.
7. *Front-End Extraction Latency:* Computational overhead of multi-angle FrFT and wavelet decomposition for real-time edge streaming.

---

### 2.8 Section VII: Conclusion & Future Work
Rephrased concluding synthesis and structured three forward-looking research directions:
1. Formulating an end-to-end differentiable FrFT front-end with backpropagation-optimized rotation angles.
2. Conducting strictly controlled 40-epoch ablations and cross-dataset evaluations under telephony codecs (AMR, GSM, Opus).
3. Developing hybrid multi-tiered architectures combining compact FrFT-WPD descriptors with self-supervised foundation representations.

---

## 3. Bibliographic Verification Matrix

All 25 entries in `references.bib` have been verified against authoritative bibliographic repositories (IEEE Xplore, Interspeech/ISCA, Elsevier ScienceDirect, NeurIPS, ICML, ICLR, CVPR).

| # | BibTeX Key | Authors | Year | Venue / Publisher | DOI / Record ID | Verification Status | Citation Context & Role |
|---|---|---|---|---|---|---|---|
| 1 | `asvspoof2019_database` | X. Wang et al. | 2020 | *Computer Speech & Language* (Elsevier) | `10.1016/j.csl.2020.101114` | **Fully Verified** | ASVspoof 2019 database specification; protocol citation |
| 2 | `asvspoof2019_overview` | M. Todisco et al. | 2019 | *Proc. Interspeech* | `10.21437/Interspeech.2019-2249` | **Fully Verified** | ASVspoof 2019 challenge overview; threat landscape |
| 3 | `lcnn_lavrentyeva2017` | G. Lavrentyeva et al. | 2017 | *Proc. Interspeech* | `10.21437/Interspeech.2017-1075` | **Fully Verified** | Light CNN introduced to audio anti-spoofing |
| 4 | `lcnn_lavrentyeva2019` | G. Lavrentyeva et al. | 2019 | *Proc. Interspeech* | `10.21437/Interspeech.2019-1768` | **Fully Verified** | STC system for ASVspoof 2019; deep LCNN |
| 5 | `mfm_wu2018` | X. Wu et al. | 2018 | *IEEE Trans. Inf. Forensics Security* | `10.1109/TIFS.2018.2800293` | **Fully Verified** | Original Max-Feature-Map (MFM) formulation |
| 6 | `frft_namias1980` | V. Namias | 1980 | *IMA J. Appl. Math.* (Oxford) | `10.1093/imamat/25.3.241` | **Fully Verified** | Mathematical foundation of Fractional Fourier Transform |
| 7 | `frft_ozaktas2001` | H. M. Ozaktas et al. | 2001 | John Wiley & Sons (Book) | ISBN 978-0471963462 | **Fully Verified** | FrFT monograph and signal-processing formulation |
| 8 | `frft_bailey1991` | D. H. Bailey & P. N. Swarztrauber | 1991 | *SIAM Review* | `10.1137/1033097` | **Fully Verified** | Fast discrete computation of FrFT via chirp FFT |
| 9 | `wpd_coifman1992` | R. R. Coifman & M. V. Wickerhauser | 1992 | *IEEE Trans. Inf. Theory* | `10.1109/18.119732` | **Fully Verified** | Wavelet packet decomposition and best-basis selection |
| 10 | `mallat2008` | S. Mallat | 2008 | Academic Press (Book, 3rd ed.) | ISBN 978-0123743701 | **Fully Verified** | Comprehensive wavelet signal-processing theory |
| 11 | `lfcc_sahidullah2015` | M. Sahidullah et al. | 2015 | *Proc. Interspeech* | `10.21437/Interspeech.2015-467` | **Fully Verified** | Linear Frequency Cepstral Coefficients (LFCC) baseline |
| 12 | `rawnet_jung2019` | J.-w. Jung et al. | 2019 | *Proc. Interspeech* | `10.21437/Interspeech.2019-1982` | **Fully Verified** | Raw waveform deep architecture (RawNet) |
| 13 | `rawnet2_tak2021` | H. Tak et al. | 2021 | *Proc. IEEE ICASSP* | `10.1109/ICASSP39728.2021.9414234` | **Fully Verified** | RawNet2 end-to-end anti-spoofing countermeasure |
| 14 | `aasist_jung2022` | J.-w. Jung et al. | 2022 | *Proc. IEEE ICASSP* | `10.1109/ICASSP43922.2022.9746484` | **Fully Verified** | AASIST graph attention anti-spoofing network |
| 15 | `wang2021comparative` | X. Wang & J. Yamagishi | 2021 | *Proc. Interspeech* | `10.21437/Interspeech.2021-382` | **Fully Verified** | Comparative analysis of neural spoofing countermeasures |
| 16 | `yang2023multiview` | Y. Yang et al. | 2023 | *Proc. IEEE ICASSP* | `10.1109/ICASSP49357.2023.10095843` | **Fully Verified** | Multi-view feature fusion in deepfake detection |
| 17 | `zbezhkovska2023sinc` | U. Zbezhkovska & O. Khapilin | 2023 | *Proc. IEEE UkrMiCo* | `10.1109/UkrMiCo59528.2023.10372341` | **Fully Verified** | Sinc and wavelet filtering in RawNet2 |
| 18 | `pal2023synthetic` | M. Pal, D. Paul, G. Saha | 2023 | *Computer Speech & Language* | `10.1016/j.csl.2023.101485` | **Fully Verified** | Fundamental frequency (F0) drift and spectral analysis |
| 19 | `xue2022audio` | J. Xue et al. | 2022 | *Proc. Interspeech* | `10.21437/Interspeech.2022-10878` | **Fully Verified** | Complex STFT spectrogram and phase modeling |
| 20 | `kheir2023two` | Y. El Kheir et al. | 2023 | *Proc. IEEE ICASSP* | `10.1109/ICASSP49357.2023.10096532` | **Fully Verified** | Self-supervised foundation feature fusion (WavLM) |
| 21 | `wang2023wavelet` | B. Wang et al. | 2023 | *IEEE Signal Process. Lett.* | `10.1109/LSP.2023.3312891` | **Fully Verified** | Wavelet attention network for deepfake detection |
| 22 | `hifigan_kong2020` | J. Kong, J. Kim, J. Bae | 2020 | *NeurIPS*, vol. 33 | NeurIPS-2020 Record | **Fully Verified** | HiFi-GAN neural vocoder generative model |
| 23 | `vits_kim2021` | J. Kim, J. Kong, J. Son | 2021 | *Proc. ICML* | PMLR vol. 139 | **Fully Verified** | VITS end-to-end conditional VAE-GAN TTS model |
| 24 | `fastspeech2_ren2021` | Y. Ren et al. | 2021 | *Proc. ICLR* | OpenReview ICLR 2021 | **Fully Verified** | FastSpeech 2 non-autoregressive TTS model |
| 25 | `labelsmoothing_szegedy` | C. Szegedy et al. | 2016 | *Proc. IEEE CVPR* | `10.1109/CVPR.2016.308` | **Fully Verified** | Label smoothing regularization formulation |

**Summary Statistics:**
- Total References in Bibliography: **25**
- Fully Verified Real-World Citations: **25 (100.0%)**
- Partially Verified / Working Papers: **0 (0.0%)**
- Unverified / Fabricated References: **0 (0.0%)**
- References Cited in Manuscript Body: **25 (100.0%)**
- Unreferenced Bibliography Entries: **0 (0.0%)**

---

## 4. Experimental Evidence Preservation Audit

This checklist verifies that no experimental data, architectural dimensions, or metric results were modified or compromised during the originality rewrite:

| Dimension / Metric / Artifact | Documented Project Evidence | Value in Revised Manuscript | Parity Status |
|---|---|---|---|
| FrFT Feature Dimensions | 7 angles × 8 chunks × 8 stats = 448 | 448 descriptors | **VERIFIED** |
| WPD Feature Dimensions | 32 sub-bands (level 5 db4) × 8 stats = 256 | 256 descriptors | **VERIFIED** |
| Fused Feature Vector Size | 448 + 256 = 704 | 704 dimensions | **VERIFIED** |
| Scaler Provenance | Fitted strictly on training split ($N=25,380$) | Fitted strictly on training split | **VERIFIED** |
| Trainable Parameters | Exactly 576,386 parameters (~2.3 MB) | 576,386 parameters (~2.3 MB) | **VERIFIED** |
| Classifier Layer Types | Dense Linear layers with MFM activations (0 conv) | Dense Linear + MFM (0 conv) | **VERIFIED** |
| Total Audio Corpus Count | 121,461 utterances (12,483 real, 108,978 spoof) | 121,461 utterances (1:8.73 ratio) | **VERIFIED** |
| Evaluation Split Count | 71,237 utterances (7,355 real, 63,882 spoof) | 71,237 utterances (1:8.69 ratio) | **VERIFIED** |
| Unseen Attack Scope | 13 attack algorithms (A07–A19) | 13 attack algorithms (A07–A19) | **VERIFIED** |
| Champion Eval EER | 10.6698% nearest / 10.6665% linear $\to$ 10.67% | 10.67% EER | **VERIFIED** |
| EER Operating Threshold | $\tau^* = 0.8362$ on $P(\text{bonafide})$ | $\tau^* = 0.8362$ on $P(\text{bonafide})$ | **VERIFIED** |
| Champion Overall Accuracy | 85.61% (60,989 / 71,237 at $\tau = 0.50$) | 85.61% | **VERIFIED** |
| Champion Balanced Accuracy | 90.26% ((84.40% + 96.13%) / 2) | 90.26% | **VERIFIED** |
| Bona Fide Sensitivity / Recall | 96.13% (7,070 / 7,355 at $\tau = 0.50$) | 96.13% | **VERIFIED** |
| Spoof Specificity / Recall | 84.40% (53,919 / 63,882 at $\tau = 0.50$) | 84.40% | **VERIFIED** |
| Operational FAR at $\tau=0.50$ | 15.60% (9,963 / 63,882) | 15.60% | **VERIFIED** |
| Operational FRR at $\tau=0.50$ | 3.87% (285 / 7,355) | 3.87% | **VERIFIED** |
| WPD Alone Baseline | 25.66% EER, 41.52% Acc (25 epochs) | 25.66% EER, 41.52% Acc (25 epochs) | **VERIFIED** |
| LFCC Baseline | 16.70% EER, 48.61% Acc (25 epochs) | 16.70% EER, 48.61% Acc (25 epochs) | **VERIFIED** |
| FrFT Alone Baseline | 13.28% EER, 84.27% Acc (25 epochs) | 13.28% EER, 84.27% Acc (25 epochs) | **VERIFIED** |
| Version 2 Overfitting Stats | Dev EER 0.05%, Eval EER 13.55%, Eval Acc 40.27% | Dev EER 0.05%, Eval EER 13.55%, Eval Acc 40.27% | **VERIFIED** |
| Post-Hoc Slicing Disclosure | Columns 0:704 selected after v2 diagnosis | Transparently disclosed in text | **VERIFIED** |
| Limitations Count | 7 explicit scientific/practical limitations | 7 numbered limitations detailed | **VERIFIED** |

---

## 5. Artifact Parity & Compilation Verification

| Deliverable | File Path | Format / Style | Status |
|---|---|---|---|
| Primary LaTeX Source | `research_paper.tex` | IEEEtran two-column conference format | **Compiled (0 errors, 0 overfull boxes)** |
| Rendered PDF Manuscript | `research_paper_final.pdf` | 12 pages, IEEE two-column typography | **Verified (Visual inspection completed)** |
| Editable Word Manuscript | `research_paper_draft.docx` | OpenXML 2-column sections, full-width headers | **Generated & Verified (~1.19 MB)** |
| Authoritative BibTeX Library | `references.bib` | 25 verified peer-reviewed publications | **100% Citation Cross-Referenced** |
| Scientific Audit Document | `research_audit.md` | Full empirical evidence audit ledger | **Updated & Current** |
| Revision History Log | `research_revision_log.md` | Chronological audit change register | **Updated & Current** |
| Originality Audit Deliverable | `originality_audit.md` | Section rewrites, bibliography, parity checks | **Completed (Current File)** |

---

## 6. Guidance for Official Plagiarism Verification

To obtain an authoritative, institutional similarity report:
1. Submit `research_paper_final.pdf` (or `research_paper_draft.docx`) to the official university Turnitin / iThenticate portal.
2. Ensure standard institutional filtering settings:
   - Exclude Bibliography: **ON** (references in Section VII should not count toward textual similarity).
   - Exclude Small Matches: **ON** (matches $\le 5$ consecutive common technical words excluded).
   - Exclude Quoted Material: **ON** (if any standard definitions are cited).
3. Record the generated similarity report index, matching sources breakdown, and repository certificate in the final capstone documentation repository.
