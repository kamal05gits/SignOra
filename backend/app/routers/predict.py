# ---------------------------------------------------------------------------
# MODULE: Prediction endpoints
#
# One route per input modality, because the models in models/ are trained on
# different inputs and a mismatch is the single most common integration bug:
#
#   /api/predict/landmarks  <- (1, T, 63)   MediaPipe landmark sequences
#   /api/predict/image      <- (1, S, S, 3) a single still frame
#   /api/predict/video      <- (1, F, S, S, 3) a short clip
#   /api/predict/features   <- (1, D)       engineered HandFeatureSet vectors
#
# Every route runs the identical code path in InferenceService, and every
# response reports which model actually answered plus its latency.
# ---------------------------------------------------------------------------
from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..schemas import (
    FeaturesRequest,
    ImageRequest,
    LandmarksRequest,
    PredictionResponse,
    VideoRequest,
)

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/landmarks", response_model=PredictionResponse)
async def predict_landmarks(request: Request, payload: LandmarksRequest) -> PredictionResponse:
    """Classify a sequence of hand-landmark frames (the live-camera path)."""
    service = request.app.state.inference
    return await run_in_threadpool(service.predict_landmarks, payload)


@router.post("/image", response_model=PredictionResponse)
async def predict_image(request: Request, payload: ImageRequest) -> PredictionResponse:
    """Classify a single still frame sent as base64."""
    service = request.app.state.inference
    return await run_in_threadpool(service.predict_image, payload)


@router.post("/image/upload", response_model=PredictionResponse)
async def predict_image_upload(
    request: Request,
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    top_k: int = Form(default=5),
) -> PredictionResponse:
    """Multipart variant of /image, convenient for curl and the docs UI."""
    import base64

    blob = await file.read()
    payload = ImageRequest(
        model_id=model_id,
        top_k=top_k,
        image_base64=base64.b64encode(blob).decode("ascii"),
    )
    service = request.app.state.inference
    return await run_in_threadpool(service.predict_image, payload)


@router.post("/video", response_model=PredictionResponse)
async def predict_video(request: Request, payload: VideoRequest) -> PredictionResponse:
    """Classify a short clip sent as an ordered list of base64 frames."""
    service = request.app.state.inference
    return await run_in_threadpool(service.predict_video, payload)


@router.post("/features", response_model=PredictionResponse)
async def predict_features(request: Request, payload: FeaturesRequest) -> PredictionResponse:
    """Classify engineered per-hand features (the frontend's HandFeatureSet)."""
    service = request.app.state.inference
    return await run_in_threadpool(service.predict_features, payload)
