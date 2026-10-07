# ---------------------------------------------------------------------------
# MODULE: Inference service
#
# The single place where a request is turned into a tensor, run through a
# model, and turned back into labelled predictions. Routers stay thin so the
# exact same code path serves REST, the WebSocket stream and the test-suite.
# ---------------------------------------------------------------------------
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import preprocessing
from .registry import ModelRegistry
from .runtimes.base import ModelHandle, ModelLoadError, ModelSpec, RuntimeUnavailable
from .schemas import (
    FeaturesRequest,
    ImageRequest,
    LandmarksRequest,
    Modality,
    Prediction,
    PredictionResponse,
    VideoRequest,
)


# Upper bound on how many unknown-modality models may be loaded while looking
# for one that matches the requested input type (see _resolve_default).
MAX_PROBE_LOADS = 3


class InferenceError(Exception):
    """A user-facing error (bad payload, wrong endpoint for this model)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InferenceService:
    def __init__(self, registry: ModelRegistry, default_top_k: int = 5) -> None:
        self.registry = registry
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------
    def resolve(self, model_id: Optional[str], modality: Modality) -> ModelHandle:
        if model_id:
            handle = self._load(model_id)
            if handle.spec.modality not in ("unknown", modality):
                raise InferenceError(
                    f"Model '{model_id}' expects '{handle.spec.modality}' input — "
                    f"call /api/predict/{handle.spec.modality} instead of /api/predict/{modality}.",
                    status_code=409,
                )
            return handle
        return self._resolve_default(modality)

    def _resolve_default(self, modality: Modality) -> ModelHandle:
        """
        Pick a model for this modality when the caller did not name one.

        A model's modality is only certain once it is loaded (a Keras graph
        declares its input shape, a PyTorch checkpoint only hints at one), so
        candidates whose modality is still "unknown" are probed in order.
        """
        candidate = self.registry.default_model_id(modality)
        if candidate:
            handle = self._load(candidate)
            if handle.spec.modality in ("unknown", modality):
                return handle

        specs = self.registry.specs
        if not specs:
            raise InferenceError(
                "No model is available. Put a .pth/.keras/.h5 file in models/ (pull Git LFS first).",
                status_code=503,
            )

        probed = 0
        for model_id, spec in specs.items():
            if spec.lfs_pointer or spec.modality not in ("unknown", modality) or probed >= MAX_PROBE_LOADS:
                continue
            probed += 1
            try:
                handle = self._load(model_id)
            except InferenceError:
                continue
            if handle.spec.modality in ("unknown", modality):
                return handle

        known = ", ".join(
            f"{model_id}={spec.modality}" for model_id, spec in specs.items() if spec.modality != "unknown"
        ) or "none declared"
        raise InferenceError(
            f"No loadable model accepts '{modality}' input ({known}). "
            f"Pass an explicit model_id, or declare the modality in models/model_manifest.json.",
            status_code=409,
        )

    def _load(self, model_id: str) -> ModelHandle:
        try:
            return self.registry.handle_for(model_id)
        except ModelLoadError as exc:
            raise InferenceError(str(exc), status_code=503) from exc
        except RuntimeUnavailable as exc:
            raise InferenceError(str(exc), status_code=503) from exc

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    def predict_landmarks(self, request: LandmarksRequest) -> PredictionResponse:
        handle = self.resolve(request.model_id, "landmarks")
        spec = handle.spec
        tensor, filled, length = preprocessing.landmark_tensor(
            request.frames,
            input_shape=handle.input_shape,
            sequence_length=spec.sequence_length,
            hand=request.hand,
            normalization=spec.normalization,
            include_handedness=spec.include_handedness,
        )
        warnings: list[str] = []
        if filled == 0:
            warnings.append("no hand was present in any frame; the model saw a zero-filled input")
        elif length == 1:
            # A single-frame model only ever sees the newest hand, so a window
            # with hands in it is not "missing frames" — saying so would be wrong.
            warnings.append("model takes a single frame, so only the newest detected hand was used")
        elif filled < len(request.frames):
            warnings.append(f"{len(request.frames) - filled} frame(s) contained no hand and were zero-filled")
        return self._run(handle, tensor, request.top_k, frames_used=len(request.frames), extra_warnings=warnings)

    def predict_features(self, request: FeaturesRequest) -> PredictionResponse:
        handle = self.resolve(request.model_id, "features")
        tensor = preprocessing.features_to_vector(request.features, normalization=handle.spec.normalization)
        return self._run(handle, tensor, request.top_k, frames_used=1)

    def predict_image(self, request: ImageRequest) -> PredictionResponse:
        handle = self.resolve(request.model_id, "image")
        array = preprocessing.decode_image(request.image_base64)
        tensor = self._image_tensor(handle.spec, array)
        return self._run(handle, tensor, request.top_k, frames_used=1)

    def predict_video(self, request: VideoRequest) -> PredictionResponse:
        handle = self.resolve(request.model_id, "video")
        spec = handle.spec
        size = spec.image_size or 224
        target = spec.video_frames or len(request.frames_base64)
        arrays = [preprocessing.decode_image(payload) for payload in request.frames_base64]
        if not arrays:
            raise InferenceError("frames_base64 is empty")
        sampled = _sample_frames(arrays, target)
        prepared = [
            preprocessing.prepare_image(
                array,
                size=size,
                channel_order=spec.channel_order,
                scale=spec.scale,
                mean=spec.mean,
                std=spec.std,
            )[0]
            for array in sampled
        ]
        tensor = np.stack(prepared, axis=0)[None, :, :, :, :].astype(np.float32)
        return self._run(handle, tensor, request.top_k, frames_used=len(sampled))

    # ------------------------------------------------------------------
    def _image_tensor(self, spec: ModelSpec, array: np.ndarray) -> np.ndarray:
        size = spec.image_size or 224
        return preprocessing.prepare_image(
            array,
            size=size,
            channel_order=spec.channel_order,
            scale=spec.scale,
            mean=spec.mean,
            std=spec.std,
            data_format=spec.data_format,
        )

    def _run(
        self,
        handle: ModelHandle,
        tensor: np.ndarray,
        top_k: Optional[int],
        frames_used: int = 0,
        extra_warnings: Optional[list[str]] = None,
    ) -> PredictionResponse:
        warnings = list(handle.warnings) + list(extra_warnings or [])
        started = time.perf_counter()
        try:
            raw = handle.predict(tensor)
        except Exception as exc:  # noqa: BLE001 - report shape mismatches to the caller
            raise InferenceError(
                f"model '{handle.spec.id}' rejected the input of shape {list(tensor.shape)}: "
                f"{type(exc).__name__}: {exc}",
                status_code=422,
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        self.registry.record_inference(handle.spec.id, latency_ms)

        scores = _to_probabilities(raw, handle)
        if scores.size == 0:
            raise InferenceError(f"model '{handle.spec.id}' produced an empty output", status_code=500)
        if scores.size == 1:
            warnings.append("model has a single output unit — its score is reported as-is")

        limit = top_k or self.default_top_k
        predictions: list[Prediction] = []
        for index, score in preprocessing.top_k_indices(scores, limit):
            label, gloss = handle.spec.label_at(index)
            predictions.append(Prediction(index=index, label=label, gloss=gloss, score=round(score, 6)))

        return PredictionResponse(
            model_id=handle.spec.id,
            modality=handle.spec.modality,
            predictions=predictions,
            top=predictions[0] if predictions else None,
            frames_used=frames_used,
            latency_ms=round(latency_ms, 3),
            warnings=warnings,
        )


def _to_probabilities(raw: np.ndarray, handle: ModelHandle) -> np.ndarray:
    """Squeeze the batch dimension and normalize logits to probabilities."""
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    row = array[0]
    if handle.already_probabilistic:
        return row
    return preprocessing.softmax(row)


def _sample_frames(arrays: list[np.ndarray], target: int) -> list[np.ndarray]:
    """Uniformly sample `target` frames out of the clip (repeat-pad if short)."""
    if target <= 0:
        target = len(arrays)
    if len(arrays) == target:
        return arrays
    if len(arrays) > target:
        indexes = np.linspace(0, len(arrays) - 1, num=target).round().astype(int)
        return [arrays[i] for i in indexes]
    sampled = list(arrays)
    while len(sampled) < target:
        sampled.append(arrays[-1])
    return sampled
