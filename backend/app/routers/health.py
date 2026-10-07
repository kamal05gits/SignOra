# ---------------------------------------------------------------------------
# MODULE: Health / diagnostics endpoints
# ---------------------------------------------------------------------------
from __future__ import annotations

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse, include_in_schema=False)
def health(request: Request) -> HealthResponse:
    """Liveness probe: runtime availability plus model readiness."""
    registry = request.app.state.registry
    runtimes = registry.runtime_versions()
    models = registry.list_info()
    ready = [m for m in models if m.status in ("ready", "idle")]
    problems = list(registry.discovery_problems)
    for model in models:
        if model.status in ("load_error", "runtime_missing", "missing_weights", "needs_manifest"):
            problems.append(f"{model.id}: {model.status} — {model.detail or 'not loadable'}")

    return HealthResponse(
        status="ok" if models and not problems else "degraded",
        version=request.app.state.settings.version,
        runtimes=runtimes,
        models_total=len(models),
        models_ready=len([m for m in models if m.loaded]) or len(ready),
        problems=problems,
    )


@router.get("/api/config")
def config(request: Request) -> dict:
    """Effective server configuration (safe subset — no secrets are stored here)."""
    settings = request.app.state.settings
    return {
        "version": settings.version,
        "model_dir": str(settings.model_dir),
        "data_dir": str(settings.data_dir),
        "preload_models": settings.preload_models,
        "max_loaded_models": settings.max_loaded_models,
        "default_top_k": settings.default_top_k,
        "default_sequence_length": settings.default_sequence_length,
        "default_image_size": settings.default_image_size,
        "default_video_frames": settings.default_video_frames,
        "manifest_path": str(settings.manifest_path),
        "manifest_exists": settings.manifest_path.is_file(),
    }
