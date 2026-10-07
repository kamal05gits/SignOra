# ---------------------------------------------------------------------------
# MODULE: Label / gloss resolution
#
# Class names cannot be read out of a bare PyTorch checkpoint, and a Keras
# model only carries them if they were saved into the graph. Resolution order
# (first hit wins) keeps this deterministic and documented:
#
#   1. `labels` inline in models/model_manifest.json
#   2. `labels_file` referenced from the manifest (relative to models/)
#   3. a sidecar next to the weights: <stem>.labels.json / <stem>.labels.txt
#   4. models/labels.json or models/labels.txt (shared by every model)
#   5. nothing  -> synthetic "class_0", "class_1", ... and a warning in the API
#
# Label files may be a plain list of strings, a list of {"label","gloss"}
# objects, or an object with "labels"/"glosses" keys.
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

LabelResult = tuple[Optional[list[str]], Optional[list[str]], Optional[str]]


def _coerce(payload: Any) -> LabelResult:
    """Accept the three supported label-file shapes."""
    if payload is None:
        return None, None, None

    if isinstance(payload, dict):
        if "classes" in payload and isinstance(payload["classes"], dict):
            # {"classes": {"0": "A", "1": "B"}} — the format Keras' save writes.
            ordered = sorted(payload["classes"].items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
            labels = [str(v) for _, v in ordered]
            return labels, None, "classes-map"
        labels = payload.get("labels")
        glosses = payload.get("glosses")
        return _coerce(labels)[0], _coerce(glosses)[0], "object"

    if not isinstance(payload, list):
        raise ValueError(f"unsupported label payload type: {type(payload).__name__}")

    labels: list[str] = []
    glosses: list[Optional[str]] = []
    has_gloss = False
    for item in payload:
        if isinstance(item, str):
            labels.append(item)
            glosses.append(None)
        elif isinstance(item, dict):
            labels.append(str(item.get("label", item.get("class", item.get("name", "")))))
            gloss = item.get("gloss") or item.get("word")
            glosses.append(str(gloss) if gloss else None)
            has_gloss = has_gloss or bool(gloss)
        else:
            raise ValueError(f"unsupported label entry type: {type(item).__name__}")
    return labels, (glosses if has_gloss else None), "list"


def _read_file(path: Path) -> LabelResult:
    if path.suffix.lower() == ".json":
        return _coerce(json.loads(path.read_text(encoding="utf-8")))
    # Plain text: one class per line, optional "label<TAB>gloss".
    labels: list[str] = []
    glosses: list[Optional[str]] = []
    has_gloss = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            label, gloss = line.split("\t", 1)
            labels.append(label.strip())
            glosses.append(gloss.strip() or None)
            has_gloss = True
        elif "," in line and not line.startswith("{"):
            label, _, gloss = line.partition(",")
            labels.append(label.strip())
            glosses.append(gloss.strip() or None)
            has_gloss = has_gloss or bool(gloss.strip())
        else:
            labels.append(line)
            glosses.append(None)
    return labels, (glosses if has_gloss else None), "text"


def resolve_labels(spec_labels: Any, spec_labels_file: Any, weight_path: Path, model_dir: Path) -> LabelResult:
    """Apply the resolution order above and return (labels, glosses, source)."""
    if spec_labels:
        labels, glosses, _ = _coerce(spec_labels)
        if labels:
            return labels, glosses, "manifest-inline"

    candidates: list[tuple[Path, str]] = []
    if spec_labels_file:
        candidates.append((model_dir / str(spec_labels_file), "manifest-file"))
    stem = weight_path.stem
    candidates += [
        (weight_path.with_name(f"{stem}.labels.json"), "sidecar-json"),
        (weight_path.with_name(f"{stem}.labels.txt"), "sidecar-text"),
        (model_dir / "labels.json", "shared-json"),
        (model_dir / "labels.txt", "shared-text"),
    ]
    for path, source in candidates:
        if path.is_file():
            try:
                labels, glosses, _ = _read_file(path)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"{path.name}: {exc}") from exc
            if labels:
                return labels, glosses, source
    return None, None, None
