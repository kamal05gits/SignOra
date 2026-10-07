# ---------------------------------------------------------------------------
# MODULE: Model registry
#
# Single owner of every model in models/. Responsibilities:
#   * discover weight files and resolve their specs (see runtimes/spec.py)
#   * load models lazily on first use, keep an LRU of resident models
#   * remember *why* a model failed so /api/models can explain it
#   * record per-model load and inference timings for the UI
#
# Everything mutable is guarded by an RLock because FastAPI serves requests
# from a thread pool and `load` is expensive enough that two concurrent
# requests must not both pay for it.
# ---------------------------------------------------------------------------
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .runtimes.base import ModelHandle, ModelLoadError, ModelSpec, RuntimeUnavailable
from .runtimes.keras_runtime import KerasRuntime
from .runtimes.spec import build_specs
from .runtimes.torch_runtime import TorchRuntime
from .schemas import Framework, ModelInfo, Modality, ModelStatus

RUNTIMES = {"torch": TorchRuntime(), "keras": KerasRuntime()}


class ModelNotFound(KeyError):
    """Raised when a model id does not exist in models/."""


@dataclass
class ModelStats:
    load_time_ms: Optional[float] = None
    last_inference_ms: Optional[float] = None
    inference_count: int = 0
    last_used: float = field(default_factory=time.monotonic)
    status: ModelStatus = "idle"
    detail: Optional[str] = None
    hint: Optional[str] = None


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._specs: dict[str, ModelSpec] = {}
        self._handles: dict[str, ModelHandle] = {}
        self._stats: dict[str, ModelStats] = {}
        self.discovery_problems: list[str] = []
        self.refresh()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        with self._lock:
            specs, problems = build_specs(self.settings)
            self._specs = {spec.id: spec for spec in specs}
            self.discovery_problems = problems
            for spec_id in list(self._stats):
                if spec_id not in self._specs:
                    self._stats.pop(spec_id, None)
                    self._handles.pop(spec_id, None)
            for spec_id in self._specs:
                self._stats.setdefault(spec_id, ModelStats())

    @property
    def specs(self) -> dict[str, ModelSpec]:
        return dict(self._specs)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get_spec(self, model_id: str) -> ModelSpec:
        with self._lock:
            spec = self._specs.get(model_id)
        if spec is None:
            raise ModelNotFound(f"no model named '{model_id}'")
        return spec

    def runtime_for(self, framework: Framework):
        runtime = RUNTIMES.get(framework)
        if runtime is None:  # pragma: no cover - guarded by the schemas enum
            raise ModelLoadError(f"no runtime registered for framework '{framework}'")
        return runtime

    def runtime_versions(self) -> dict[str, Optional[str]]:
        versions: dict[str, Optional[str]] = {}
        for name, runtime in RUNTIMES.items():
            installed, version = runtime.availability()
            versions[name] = version if installed else None
        return versions

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._handles

    def handle_for(self, model_id: str) -> ModelHandle:
        """Return a loaded handle, loading on demand."""
        with self._lock:
            handle = self._handles.get(model_id)
            if handle is not None:
                self._stats[model_id].last_used = time.monotonic()
                return handle
        return self.load(model_id)

    def default_model_id(self, modality: Optional[Modality] = None) -> Optional[str]:
        """Pick the server default: a loaded model first, then any loadable one."""
        with self._lock:
            if modality:
                for model_id, handle in self._handles.items():
                    if handle.spec.modality == modality:
                        return model_id
                for model_id, spec in self._specs.items():
                    if spec.modality == modality and not spec.lfs_pointer:
                        return model_id
                return None
            if self._handles:
                return next(iter(self._handles))
            for model_id, spec in self._specs.items():
                if not spec.lfs_pointer:
                    return model_id
        return None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------
    def load(self, model_id: str) -> ModelHandle:
        spec = self.get_spec(model_id)
        with self._lock:
            existing = self._handles.get(model_id)
            if existing is not None:
                self._stats[model_id].last_used = time.monotonic()
                return existing
            stats = self._stats[model_id]

            installed, _ = self.runtime_for(spec.framework).availability()
            if not installed:
                stats.status = "runtime_missing"
                stats.detail = f"the '{spec.framework}' runtime is not installed"
                raise RuntimeUnavailable(stats.detail)
            if spec.lfs_pointer:
                stats.status = "missing_weights"
                stats.detail = (
                    f"{spec.filename} is a Git LFS pointer ({spec.size_bytes} bytes). "
                    "Run `git lfs install && git lfs pull`."
                )
                raise ModelLoadError(stats.detail)

            started = time.perf_counter()
            try:
                handle = self.runtime_for(spec.framework).load(spec)
            except ModelLoadError as exc:
                stats.status = "needs_manifest" if "model_manifest.json" in str(exc) else "load_error"
                stats.detail = str(exc)
                stats.hint = str(exc) if "model_manifest.json" in str(exc) else None
                raise
            except RuntimeUnavailable as exc:
                stats.status = "runtime_missing"
                stats.detail = str(exc)
                raise
            except Exception as exc:  # noqa: BLE001 - surface any loader failure as an API error
                stats.status = "load_error"
                stats.detail = f"{type(exc).__name__}: {exc}"
                raise ModelLoadError(stats.detail) from exc

            stats.load_time_ms = (time.perf_counter() - started) * 1000
            stats.status = "ready"
            stats.detail = None
            stats.hint = None
            stats.last_used = time.monotonic()
            self._handles[model_id] = handle
            self._evict_if_needed()
            return handle

    def unload(self, model_id: str) -> bool:
        with self._lock:
            handle = self._handles.pop(model_id, None)
            if handle is None:
                return False
            self._stats[model_id].status = "unloaded"
            del handle
            return True

    def _evict_if_needed(self) -> None:
        limit = max(1, self.settings.max_loaded_models)
        while len(self._handles) > limit:
            oldest = min(self._handles, key=lambda mid: self._stats[mid].last_used)
            self._handles.pop(oldest, None)
            self._stats[oldest].status = "unloaded"

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def record_inference(self, model_id: str, latency_ms: float) -> None:
        with self._lock:
            stats = self._stats.get(model_id)
            if stats is None:
                return
            stats.last_inference_ms = latency_ms
            stats.inference_count += 1
            stats.last_used = time.monotonic()

    def info(self, model_id: str) -> ModelInfo:
        spec = self.get_spec(model_id)
        return self._to_info(spec)

    def list_info(self) -> list[ModelInfo]:
        with self._lock:
            return [self._to_info(spec) for spec in self._specs.values()]

    def _to_info(self, spec: ModelSpec) -> ModelInfo:
        with self._lock:
            handle = self._handles.get(spec.id)
            stats = self._stats.get(spec.id) or ModelStats()
            if handle is not None:
                spec = handle.spec  # loading may have refined modality/architecture
            status, detail, hint = stats.status, stats.detail, stats.hint
            if spec.lfs_pointer:
                status = "missing_weights"
                detail = (
                    f"Git LFS pointer only ({spec.size_bytes} bytes of the real weights). "
                    "Run `git lfs install && git lfs pull`."
                )
                hint = None
            elif handle is None and status == "idle":
                installed, _ = self.runtime_for(spec.framework).availability()
                if not installed:
                    status = "runtime_missing"
                    detail = f"the '{spec.framework}' runtime is not installed in the backend environment"
            if spec.labels is None and not spec.lfs_pointer:
                hint = hint or (
                    "no label file found — predictions are returned as class_0, class_1, … "
                    "Add models/labels.json or a models/model_manifest.json `labels_file` entry."
                )
            return ModelInfo(
                id=spec.id,
                filename=spec.filename,
                framework=spec.framework,
                format=spec.format,
                size_bytes=spec.size_bytes,
                modality=spec.modality,
                status=status,
                detail=detail,
                architecture=spec.architecture,
                input_shape=handle.input_shape if handle else None,
                output_shape=handle.output_shape if handle else None,
                num_classes=handle.num_classes if handle else None,
                labels=spec.labels,
                labels_source=spec.labels_source,
                loaded=handle is not None,
                lfs_pointer=spec.lfs_pointer,
                load_time_ms=stats.load_time_ms,
                last_inference_ms=stats.last_inference_ms,
                inference_count=stats.inference_count,
                hint=hint,
            )
