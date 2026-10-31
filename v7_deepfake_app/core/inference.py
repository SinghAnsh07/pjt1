"""
Core Inference Engine for the V7 Audio Deepfake Detection System.

Orchestrates the complete audio deepfake detection pipeline:
Waveform Validation -> Polyphase Resampling -> Wrap-Padding / Windowing ->
2D FrFT-Wavelet Extraction -> LCNN-2D Forward Pass -> OC-Softmax Scoring ->
Conservative Decision Evaluation -> Structured Result Packaging.
"""

import time
import torch
import numpy as np
from typing import Optional, Union, Tuple

from core.model import EndToEndLCNN2D
from core.checkpoint_loader import load_model
from core.preprocessor import preprocess_waveform
from core.decision_engine import DecisionEngine
from core.schemas import (
    InferenceResult,
    DecisionResult,
    ModelMetadata,
    TimingBreakdown,
)
from core.exceptions import (
    DeepfakeAppError,
    InferenceError,
)


class V7InferenceEngine:
    """
    State-managed Inference Engine for V7 Audio Deepfake Detection.

    Maintains a preloaded, frozen EndToEndLCNN2D model instance in evaluation mode
    and provides deterministic, thread-safe inference without global mutable state.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[EndToEndLCNN2D] = None,
        metadata: Optional[ModelMetadata] = None,
        device: str = "auto",
        decision_engine: Optional[DecisionEngine] = None,
    ):
        """
        Initializes the inference engine.
        Accepts either an existing preloaded model instance or loads from checkpoint_path.
        """
        if model is not None:
            self.model = model
            self.metadata = metadata
        elif checkpoint_path is not None:
            self.model, self.metadata = load_model(checkpoint_path, device=device)
        else:
            raise ValueError("Either 'model' or 'checkpoint_path' must be provided to V7InferenceEngine.")

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.decision_engine = decision_engine or DecisionEngine()

    def predict_waveform(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        filename: Optional[str] = None,
        channel_layout: Optional[str] = None,
    ) -> InferenceResult:
        """
        Executes complete deepfake inference on an in-memory audio waveform.

        Args:
            audio: Raw waveform as numpy array or torch tensor
            sample_rate: Sampling frequency in Hz
            filename: Optional display name for logging and reports
            channel_layout: Optional layout override ('frames_first', 'channels_first')

        Returns:
            InferenceResult: Fully typed, structured diagnostic result
        """
        t_total_start = time.perf_counter()

        # Stage 1: Preprocessing & Resampling
        t0 = time.perf_counter()
        audio_tensor, prep_result = preprocess_waveform(
            audio, sample_rate, channel_layout=channel_layout
        )
        t_prep_ms = (time.perf_counter() - t0) * 1000.0

        audio_tensor = audio_tensor.to(self.device)

        # Stage 2: Feature Extraction & Neural Forward Pass
        try:
            with torch.inference_mode():
                # Extract 3-channel 2D spatiotemporal tensor
                t1 = time.perf_counter()
                spec_2d = self.model.extractor(audio_tensor)
                t_feat_ms = (time.perf_counter() - t1) * 1000.0

                # LCNN-2D Backbone and OC-Softmax Scoring
                t2 = time.perf_counter()
                emb = self.model.backbone(spec_2d)
                scores = self.model.oc_loss.get_score(emb)
                t_fwd_ms = (time.perf_counter() - t2) * 1000.0

        except Exception as e:
            raise InferenceError(f"Neural forward pass failed during inference: {str(e)}") from e

        # Stage 3: Score Extraction and Finite Validation
        if not torch.isfinite(scores).all():
            raise InferenceError(f"Model generated non-finite cosine score: {scores}")

        raw_score = float(scores[0].item())

        # Stage 4: Conservative Decision Evaluation
        t3 = time.perf_counter()
        decision_result = self.decision_engine.decide(raw_score)
        t_dec_ms = (time.perf_counter() - t3) * 1000.0

        t_total_ms = (time.perf_counter() - t_total_start) * 1000.0

        timing = TimingBreakdown(
            preprocessing_ms=round(t_prep_ms, 2),
            feature_extraction_ms=round(t_feat_ms, 2),
            neural_forward_ms=round(t_fwd_ms, 2),
            decision_ms=round(t_dec_ms, 2),
            total_ms=round(t_total_ms, 2),
        )

        return InferenceResult(
            status="SUCCESS",
            filename=filename,
            model_score=round(raw_score, 4),
            decision=decision_result,
            preprocessing=prep_result,
            model_metadata=self.metadata,
            timing=timing,
            feature_tensor_shape=tuple(spec_2d.shape),
            embedding_shape=tuple(emb.shape),
            system_warnings=prep_result.warnings + decision_result.warnings,
        )


def predict_waveform(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    model: Optional[EndToEndLCNN2D] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    decision_engine: Optional[DecisionEngine] = None,
    filename: Optional[str] = None,
    channel_layout: Optional[str] = None,
) -> InferenceResult:
    """
    Stateless public helper for single-call waveform inference.
    Reuses provided model or instantiates a transient engine from checkpoint.
    """
    engine = V7InferenceEngine(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        decision_engine=decision_engine,
    )
    return engine.predict_waveform(
        audio, sample_rate, filename=filename, channel_layout=channel_layout
    )
