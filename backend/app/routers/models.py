# ---------------------------------------------------------------------------
# MODULE: Model management endpoints
# ---------------------------------------------------------------------------
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from ..registry import ModelNotFound
from ..runtimes.base import ModelLoadError, RuntimeUnavailable
from ..schemas import ModelInfo

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models(request: Request) -> list[ModelInfo]:
    """Describe every model file found in models/, loaded or not."""
    return request.app.state.registry.list_info()


@router.get("/{model_id}", response_model=ModelInfo)
def get_model(request: Request, model_id: str) -> ModelInfo:
    return request.app.state.registry.info(model_id)


@router.post("/refresh", response_model=list[ModelInfo])
def refresh(request: Request) -> list[ModelInfo]:
    """Re-scan models/ (use after dropping in new weights or a manifest)."""
    registry = request.app.state.registry
    registry.refresh()
    return registry.list_info()


@router.post("/{model_id}/load", response_model=ModelInfo)
async def load_model(request: Request, model_id: str) -> ModelInfo:
    """Load a model into memory now instead of on the first prediction."""
    registry = request.app.state.registry
    try:
        await run_in_threadpool(registry.load, model_id)
    except (ModelLoadError, RuntimeUnavailable):
        # The registry has already recorded the reason; report it back.
        return registry.info(model_id)
    return registry.info(model_id)


@router.post("/{model_id}/unload", response_model=ModelInfo)
def unload_model(request: Request, model_id: str) -> ModelInfo:
    registry = request.app.state.registry
    if not registry.unload(model_id):
        raise ModelNotFound(f"model '{model_id}' is not loaded")
    return registry.info(model_id)
