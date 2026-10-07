# ---------------------------------------------------------------------------
# MODULE: Dataset endpoints
#
# The browser records labelled samples in Dataset Collection; these endpoints
# let it push them to the backend so they can be exported as a training set or
# scored against a trained model (POST /api/dataset/evaluate).
# ---------------------------------------------------------------------------
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ..evaluation import evaluate
from ..registry import ModelNotFound
from ..runtimes.base import ModelLoadError, RuntimeUnavailable
from ..schemas import DatasetEvaluateRequest, DatasetSummary, DatasetSyncRequest, EvaluationReport

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.get("/summary", response_model=DatasetSummary)
def summary(request: Request) -> DatasetSummary:
    store = request.app.state.dataset
    result = store.summary()
    return DatasetSummary(total=result["total"], classes=result["classes"], path=str(store.path))


@router.post("/samples")
def add_samples(request: Request, payload: DatasetSyncRequest) -> dict:
    """Append recorded samples (`replace=true` rewrites the store instead)."""
    store = request.app.state.dataset
    written = store.add(payload.samples, replace=payload.replace)
    return {"written": written, "total": store.summary()["total"], "path": str(store.path)}


@router.delete("/samples")
def clear_samples(request: Request) -> dict:
    store = request.app.state.dataset
    store.clear()
    return {"written": 0, "total": 0, "path": str(store.path)}


@router.get("/export")
def export(request: Request, format: str = "npz", sequence_length: int = 30) -> Response:
    """Download the stored samples as npz (default), jsonl or csv."""
    store = request.app.state.dataset
    fmt = format.lower()
    if fmt == "jsonl":
        return Response(
            content=store.export_jsonl(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="signora_dataset.jsonl"'},
        )
    if fmt == "csv":
        return Response(
            content=store.export_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="signora_dataset.csv"'},
        )
    if fmt == "npz":
        return Response(
            content=store.export_npz(sequence_length=sequence_length),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="signora_dataset.npz"'},
        )
    return Response(status_code=400, content=f"unsupported format '{format}' (use npz, jsonl or csv)")


@router.post("/evaluate", response_model=EvaluationReport)
async def evaluate_dataset(request: Request, payload: DatasetEvaluateRequest) -> EvaluationReport:
    """Score recorded samples with a trained model and return real metrics."""
    registry = request.app.state.registry
    model_id = payload.model_id or registry.default_model_id()
    if model_id is None:
        raise ModelNotFound("no model available to evaluate against")
    try:
        handle = await run_in_threadpool(registry.handle_for, model_id)
    except (ModelLoadError, RuntimeUnavailable) as exc:
        raise ModelLoadError(str(exc)) from exc
    return await run_in_threadpool(evaluate, handle, payload)
