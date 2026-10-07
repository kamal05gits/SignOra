# ---------------------------------------------------------------------------
# MODULE: FastAPI application factory
#
# Run it with:
#   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
# or:
#   python -m backend.app.main
#
# The app owns four long-lived services on `app.state`:
#   settings   resolved configuration
#   registry   every model in models/, loaded lazily
#   inference  the request -> tensor -> prediction pipeline
#   dataset    JSONL store for samples synced from the browser
# ---------------------------------------------------------------------------
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .dataset import DatasetStore
from .inference import InferenceError, InferenceService
from .preprocessing import PreprocessingError
from .registry import ModelNotFound, ModelRegistry
from .routers import dataset as dataset_router
from .routers import health as health_router
from .routers import models as models_router
from .routers import predict as predict_router
from .routers import stream as stream_router
from .runtimes.base import ModelLoadError, RuntimeUnavailable

logger = logging.getLogger("signora.backend")


def _json_error(status_code: int, exc: Exception, **extra: object) -> JSONResponse:
    payload: dict[str, object] = {"detail": str(exc), "error": type(exc).__name__}
    payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = ModelRegistry(settings)
        app.state.settings = settings
        app.state.registry = registry
        app.state.inference = InferenceService(registry, settings.default_top_k)
        app.state.dataset = DatasetStore(settings.dataset_path)

        for model in registry.list_info():
            logger.info("discovered model %s (%s, %s)", model.id, model.framework, model.status)
        if settings.preload_models:
            for spec in registry.specs.values():
                try:
                    registry.load(spec.id)
                except (ModelLoadError, RuntimeUnavailable) as exc:
                    logger.warning("preload of %s failed: %s", spec.id, exc)

        yield

        for model_id in list(registry.specs):
            registry.unload(model_id)

    app = FastAPI(
        title=settings.api_title,
        version=settings.version,
        lifespan=lifespan,
        description=(
            "Serves the trained ISL models in `models/` to the SignOra web app. "
            "Interactive docs: /docs. Live camera stream: /ws/stream."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Errors: every failure mode the model pipeline can produce gets an HTTP
    # status that the frontend can act on (404 unknown model, 503 not loadable,
    # 409 wrong modality, 422 bad payload).
    # ------------------------------------------------------------------
    @app.exception_handler(ModelNotFound)
    async def on_model_not_found(request: Request, exc: ModelNotFound) -> JSONResponse:
        available = sorted(request.app.state.registry.specs)
        return _json_error(404, exc, available_models=available)

    @app.exception_handler(ModelLoadError)
    async def on_model_load_error(request: Request, exc: ModelLoadError) -> JSONResponse:
        # A 503 (not 500) tells the frontend "this model can't serve right now,
        # fall back to the on-device classifier" rather than "the server broke".
        return _json_error(503, exc)

    @app.exception_handler(RuntimeUnavailable)
    async def on_runtime_unavailable(request: Request, exc: RuntimeUnavailable) -> JSONResponse:
        return _json_error(503, exc)

    @app.exception_handler(PreprocessingError)
    async def on_preprocessing_error(request: Request, exc: PreprocessingError) -> JSONResponse:
        return _json_error(422, exc)

    @app.exception_handler(InferenceError)
    async def on_inference_error(request: Request, exc: InferenceError) -> JSONResponse:
        return _json_error(exc.status_code, exc)

    # ------------------------------------------------------------------
    app.include_router(health_router.router)
    app.include_router(models_router.router)
    app.include_router(predict_router.router)
    app.include_router(dataset_router.router)
    app.include_router(stream_router.router)

    @app.get("/", include_in_schema=False)
    def index(request: Request) -> dict:
        base = str(request.base_url).rstrip("/")
        return {
            "name": settings.api_title,
            "version": settings.version,
            "docs": f"{base}/docs",
            "health": f"{base}/health",
            "models": f"{base}/api/models",
            "predict": {
                "landmarks": f"{base}/api/predict/landmarks",
                "image": f"{base}/api/predict/image",
                "video": f"{base}/api/predict/video",
                "features": f"{base}/api/predict/features",
            },
            "websocket": "/ws/stream",
        }

    return app


app = create_app()


def main() -> None:
    """Entry point for `python -m backend.app.main`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
