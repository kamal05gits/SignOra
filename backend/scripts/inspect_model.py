#!/usr/bin/env python3
"""
Inspect a checkpoint and print exactly what the backend needs to load it.

    python backend/scripts/inspect_model.py models/best_model_new.pth
    python backend/scripts/inspect_model.py models/            # every file

A bare PyTorch .pth stores weights only — no architecture. This script dumps
the tensor names and shapes, prints the architecture the backend can infer on
its own, and emits a ready-to-paste models/model_manifest.json entry when it
cannot. That turns "the model won't load" into a two-minute fix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.config import MODEL_EXTENSIONS  # noqa: E402
from app.runtimes.base import ModelSpec  # noqa: E402
from app.runtimes.spec import is_lfs_pointer, lfs_metadata, model_id_for  # noqa: E402


def inspect_torch(path: Path) -> dict:
    import torch

    try:
        scripted = torch.jit.load(str(path), map_location="cpu")
        scripted.eval()
        return {
            "kind": "torchscript",
            "note": "TorchScript archive — loads without a manifest entry.",
            "parameters": sum(p.numel() for p in scripted.parameters()),
        }
    except Exception:
        pass

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)

    if hasattr(payload, "state_dict") and hasattr(payload, "eval"):
        return {
            "kind": "pickled_module",
            "class": f"{type(payload).__module__}.{type(payload).__name__}",
            "note": "The whole nn.Module was pickled — it loads as-is if its class is importable.",
            "state_dict": _summarize(payload.state_dict()),
        }

    state_dict = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights", "net"):
            if isinstance(payload.get(key), dict):
                state_dict = payload[key]
                break
    if not isinstance(state_dict, dict):
        return {"kind": "unknown", "payload_type": type(payload).__name__}

    from app.runtimes.torch_runtime import TorchRuntime

    architecture, hint = TorchRuntime._infer_architecture(state_dict)
    result: dict = {
        "kind": "state_dict",
        "tensors": _summarize(state_dict),
        "inferred_architecture": architecture,
        "total_parameters": int(sum(getattr(v, "numel", lambda: 0)() for v in state_dict.values())),
    }
    if architecture is None:
        result["manifest_suggestion"] = {
            model_id_for(path.name): {
                "architecture": "<one of: mlp, lstm, gru, cnn1d_lstm, resnet50, …>",
                "modality": "<image | video | landmarks | features>",
                "sequence_length": 30,
                "input_dim": 63,
                "num_classes": "<width of the final layer>",
            }
        }
        result["hint"] = hint
    return result


def inspect_keras(path: Path) -> dict:
    try:
        import tensorflow as tf

        load = tf.keras.models.load_model
    except ImportError:
        import keras

        load = keras.models.load_model

    try:
        model = load(str(path), compile=False)
    except Exception as exc:
        return {"kind": "keras", "error": f"{type(exc).__name__}: {exc}"}

    def shape(value):
        try:
            return list(value.shape)
        except Exception:
            return None

    inputs = getattr(model, "inputs", None)
    outputs = getattr(model, "outputs", None)
    return {
        "kind": "keras",
        "input_shape": shape(inputs[0]) if inputs else None,
        "output_shape": shape(outputs[0]) if outputs else None,
        "layers": [f"{type(layer).__name__}({layer.name})" for layer in getattr(model, "layers", [])],
        "note": "Architecture is stored in the file — no manifest entry needed.",
    }


def _summarize(state_dict, limit: int = 30) -> dict:
    entries = {}
    for key, value in list(state_dict.items())[:limit]:
        entries[key] = list(getattr(value, "shape", ()))
    if len(state_dict) > limit:
        entries["…"] = f"{len(state_dict) - limit} more tensors"
    return entries


def inspect(path: Path) -> None:
    print(f"\n=== {path.name} ({path.stat().st_size:,} bytes) ===")
    if is_lfs_pointer(path):
        meta = lfs_metadata(path)
        print("This is a Git LFS POINTER, not the real weights.")
        print(f"  expected size: {int(meta.get('size', 0)):,} bytes")
        print("  fix: git lfs install && git lfs pull")
        return
    try:
        if MODEL_EXTENSIONS.get(path.suffix.lower()) == "torch":
            report = inspect_torch(path)
        else:
            report = inspect_keras(path)
    except Exception as exc:  # noqa: BLE001 - a broken file should not stop the sweep
        report = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, default=str))


def main() -> int:
    args = [Path(a) for a in sys.argv[1:]] or [REPO_ROOT / "models"]
    targets: list[Path] = []
    for arg in args:
        if arg.is_dir():
            targets += [p for p in sorted(arg.iterdir()) if p.suffix.lower() in MODEL_EXTENSIONS]
        else:
            targets.append(arg)
    if not targets:
        print("No .pth/.pt/.keras/.h5 files found.")
        return 1
    for path in targets:
        inspect(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
