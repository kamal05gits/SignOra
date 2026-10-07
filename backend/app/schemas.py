# ---------------------------------------------------------------------------
# MODULE: API request/response schemas (pydantic)
#
# These types are the contract between the React frontend (src/lib/backend.ts
# mirrors them in TypeScript) and this server. They are deliberately strict so
# that a malformed frame from the browser fails fast with a 422 instead of
# producing a silently wrong prediction.
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Framework = Literal["torch", "keras"]
Modality = Literal["image", "video", "landmarks", "features", "unknown"]
ModelStatus = Literal[
    "ready",  # loaded in memory, serving predictions
    "idle",  # discovered, weights present, not loaded yet
    "unloaded",  # explicitly unloaded
    "missing_weights",  # Git LFS pointer, real weights not pulled
    "needs_manifest",  # architecture could not be auto-detected
    "load_error",  # loading failed (see `detail`)
    "runtime_missing",  # torch / tensorflow not installed
]

MIN_LANDMARKS = 21  # MediaPipe Hands emits 21 landmarks per hand


class Landmark(BaseModel):
    """A single normalized MediaPipe hand landmark (0..1 range)."""

    x: float
    y: float
    z: float = 0.0


class Hand(BaseModel):
    landmarks: list[Landmark] = Field(..., min_length=MIN_LANDMARKS)
    handedness: Literal["Left", "Right"] = "Right"
    score: float = 1.0

    @field_validator("landmarks")
    @classmethod
    def _exactly_21(cls, v: list[Landmark]) -> list[Landmark]:
        if len(v) != MIN_LANDMARKS:
            raise ValueError(f"expected exactly {MIN_LANDMARKS} landmarks, got {len(v)}")
        return v


class Frame(BaseModel):
    """One video frame: every hand MediaPipe detected in it."""

    hands: list[Hand] = Field(default_factory=list)


class BasePredictRequest(BaseModel):
    model_id: Optional[str] = Field(
        default=None,
        description="Model to run. Omit to use the server default (first ready model of the right modality).",
    )
    top_k: int = Field(default=5, ge=1, le=100)


class LandmarksRequest(BasePredictRequest):
    """A landmark sequence: T frames, each with 0..N hands of 21 landmarks."""

    frames: list[Frame] = Field(..., min_length=1, max_length=300)
    # Which hand to feed the model when several are in frame.
    hand: Literal["first", "left", "right", "dominant"] = "first"


class ImageRequest(BasePredictRequest):
    """A single still frame, base64 encoded (JPEG/PNG/WebP)."""

    image_base64: str = Field(..., description="base64 image bytes, data: URL prefix optional")


class VideoRequest(BasePredictRequest):
    """A short clip as an ordered list of base64 frames."""

    frames_base64: list[str] = Field(..., min_length=1, max_length=120)


class FeaturesRequest(BasePredictRequest):
    """
    Engineered per-hand features (the HandFeatureSet objects the frontend
    already computes), for models trained on those rather than raw landmarks.
    """

    features: list[dict[str, Any]] = Field(..., min_length=1, max_length=10)


class DatasetEvaluateRequest(BaseModel):
    """Run a trained model over recorded dataset samples and report real metrics."""

    model_id: Optional[str] = None
    samples: list[DatasetSamplePayload] = Field(..., min_length=1, max_length=20000)
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class ClassMetrics(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationReport(BaseModel):
    model_id: str
    total_samples: int
    correct: int
    accuracy: float
    rejected: int
    classes: list[str]
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetrics]
    latency_ms: float


class Prediction(BaseModel):
    index: int
    label: str
    gloss: Optional[str] = None
    score: float


class PredictionResponse(BaseModel):
    model_id: str
    modality: Modality
    predictions: list[Prediction]
    top: Optional[Prediction] = None
    frames_used: int = 0
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    filename: str
    framework: Framework
    format: str
    size_bytes: int
    modality: Modality
    status: ModelStatus
    detail: Optional[str] = None
    architecture: Optional[str] = None
    input_shape: Optional[list[Any]] = None
    output_shape: Optional[list[Any]] = None
    num_classes: Optional[int] = None
    labels: Optional[list[str]] = None
    labels_source: Optional[str] = None
    loaded: bool = False
    lfs_pointer: bool = False
    load_time_ms: Optional[float] = None
    last_inference_ms: Optional[float] = None
    inference_count: int = 0
    hint: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    runtimes: dict[str, Optional[str]]
    models_total: int
    models_ready: int
    problems: list[str] = Field(default_factory=list)


class DatasetSamplePayload(BaseModel):
    id: str
    label: str
    createdAt: Optional[int] = None
    features: list[dict[str, Any]] = Field(default_factory=list)
    frames: list[Frame] = Field(default_factory=list)


class DatasetSyncRequest(BaseModel):
    samples: list[DatasetSamplePayload] = Field(default_factory=list)
    replace: bool = False


class DatasetSummary(BaseModel):
    total: int
    classes: dict[str, int]
    path: str


class StreamConfig(BaseModel):
    model_id: Optional[str] = None
    sequence_length: Optional[int] = Field(default=None, ge=2, le=300)
    stride: int = Field(default=1, ge=1, le=30)
    hand: Literal["first", "left", "right", "dominant"] = "first"
