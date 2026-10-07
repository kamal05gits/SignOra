# ---------------------------------------------------------------------------
# MODULE: Model discovery + manifest resolution
#
# Scans models/ for weight files and turns each one into a ModelSpec. Two
# things deserve a comment:
#
#  * Git LFS detection — the weights in this repository are tracked by Git LFS.
#    A checkout without `git lfs pull` leaves ~130 byte pointer files behind.
#    Loading those fails with a confusing "PytorchStreamReader failed" error,
#    so we detect the pointer up front and report `missing_weights` instead.
#
#  * models/model_manifest.json — an optional file that declares the
#    architecture / input shape / preprocessing of each checkpoint. A bare
#    .pth file does not store its architecture, so without the manifest the
#    backend falls back to structural auto-detection (see torch_runtime) and,
#    when that is inconclusive, reports `needs_manifest` with a ready-to-paste
#    snippet instead of guessing.
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ..config import MODEL_EXTENSIONS, Settings
from ..schemas import Framework, Modality
from .base import ModelSpec
from .labels import resolve_labels

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"
VALID_MODALITIES = {"image", "video", "landmarks", "features", "unknown"}


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def lfs_metadata(path: Path) -> dict[str, Any]:
    """Parse the oid/size out of an LFS pointer file (used in API diagnostics)."""
    meta: dict[str, Any] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, _, value = line.partition(" ")
            if key in ("oid", "size"):
                meta[key] = value.replace("sha256:", "").strip()
    except OSError:
        pass
    return meta


def model_id_for(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "model"


def load_manifest(settings: Settings) -> tuple[dict[str, Any], Optional[str]]:
    """Read models/model_manifest.json, tolerating a missing/invalid file."""
    path = settings.manifest_path
    if not path.is_file():
        return {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"model_manifest.json could not be parsed: {exc}"
    if not isinstance(payload, dict):
        return {}, "model_manifest.json must contain a JSON object"
    return payload, None


def _apply_overrides(spec: ModelSpec, overrides: dict[str, Any]) -> ModelSpec:
    """Copy manifest keys onto the spec, validating the ones that matter."""
    simple = (
        "architecture",
        "sequence_length",
        "image_size",
        "video_frames",
        "channel_order",
        "data_format",
        "scale",
        "normalization",
        "include_handedness",
    )
    for key in simple:
        if key in overrides:
            setattr(spec, key, overrides[key])
            spec.declared.add(key)
    for key in ("mean", "std"):
        if key in overrides:
            setattr(spec, key, list(overrides[key]) if overrides[key] else None)
    if "modality" in overrides:
        modality = str(overrides["modality"]).lower()
        spec.modality = modality if modality in VALID_MODALITIES else "unknown"
    if "preimport" in overrides:
        spec.preimport = [str(m) for m in (overrides["preimport"] or [])]
    reserved = {"mean", "std", "modality", "preimport", "labels", "labels_file", "id"}
    # Keys starting with "_" are documentation-only and never reach the model builders.
    spec.extra = {
        k: v for k, v in overrides.items() if not k.startswith("_") and k not in simple and k not in reserved
    }
    return spec


def build_specs(settings: Settings) -> tuple[list[ModelSpec], list[str]]:
    """Discover every weight file in models/ and resolve its spec."""
    manifest, manifest_error = load_manifest(settings)
    defaults: dict[str, Any] = manifest.get("defaults", {}) or {}
    model_overrides: dict[str, Any] = manifest.get("models", {}) or {}

    problems: list[str] = []
    if manifest_error:
        problems.append(manifest_error)

    if not settings.model_dir.is_dir():
        return [], problems + [f"model directory not found: {settings.model_dir}"]

    specs: list[ModelSpec] = []
    for path in sorted(settings.model_dir.iterdir()):
        framework: Optional[Framework] = MODEL_EXTENSIONS.get(path.suffix.lower())  # type: ignore[assignment]
        if framework is None or not path.is_file():
            continue

        overrides = dict(defaults)
        overrides.update(model_overrides.get(path.name, {}) or {})
        overrides.update(model_overrides.get(model_id_for(path.name), {}) or {})

        lfs = is_lfs_pointer(path)
        spec = ModelSpec(
            id=str(overrides.get("id") or model_id_for(path.name)),
            path=path,
            filename=path.name,
            framework=framework,
            format=path.suffix.lstrip(".").lower(),
            size_bytes=(int(overrides.get("size_bytes")) if lfs and overrides.get("size_bytes") else path.stat().st_size),
            modality="unknown",
            lfs_pointer=lfs,
        )
        _apply_overrides(spec, overrides)

        try:
            labels, glosses, source = resolve_labels(
                overrides.get("labels"), overrides.get("labels_file"), path, settings.model_dir
            )
        except ValueError as exc:
            problems.append(str(exc))
            labels = glosses = None
            source = None
        spec.labels, spec.glosses, spec.labels_source = labels, glosses, source

        if spec.modality == "unknown":
            spec.modality = _guess_modality(spec)
        specs.append(spec)

    # Guarantee unique ids: two files sharing a stem (e.g. sign_language_model.h5
    # and sign_language_model.keras) are disambiguated with their extension.
    seen: dict[str, int] = {}
    for spec in specs:
        seen[spec.id] = seen.get(spec.id, 0) + 1
    for spec in specs:
        if seen[spec.id] > 1:
            spec.id = re.sub(r"[^a-z0-9]+", "_", spec.filename.lower()).strip("_")
    return specs, problems


def _guess_modality(spec: ModelSpec) -> Modality:
    """Cheap modality guess from the file format / declared architecture."""
    arch = (spec.architecture or "").lower()
    if spec.framework == "keras":
        # Keras specs are refined from the real input shape once loaded.
        if "movinet" in spec.filename.lower() or "movinet" in arch or "video" in arch:
            return "video"
        return "unknown"
    if "resnet" in arch or "mobilenet" in arch or "efficientnet" in arch or "vgg" in arch:
        return "image"
    if "lstm" in arch or "gru" in arch or "transformer" in arch:
        return "landmarks"
    return "unknown"
