# ---------------------------------------------------------------------------
# MODULE: PyTorch runtime
#
# Loads .pth / .pt checkpoints. Three shapes of checkpoint are handled, in
# this order:
#
#   1. TorchScript archive      -> torch.jit.load, no architecture needed
#   2. A whole pickled Module   -> used as-is
#   3. A state_dict             -> the architecture is taken from
#                                  models/model_manifest.json when present,
#                                  otherwise inferred structurally
#                                  (ResNet family, plain MLP, LSTM/GRU)
#
# When inference is inconclusive we raise ModelLoadError with a `hint`
# containing the actual layer shapes so the manifest entry can be written in
# one step instead of by guesswork.
# ---------------------------------------------------------------------------
from __future__ import annotations

import re
import time
from typing import Any, Optional

import numpy as np

from ..schemas import Modality
from .base import BaseRuntime, ModelHandle, ModelLoadError, ModelSpec, RuntimeUnavailable

NUMERIC_LAYER = re.compile(r"^(\d+)\.(weight|bias)$")
TORCHVISION_IMAGE_MODELS = {
    "resnet18": (2, "basic"),
    "resnet34": (3, "basic"),
    "resnet50": (3, "bottleneck"),
    "resnet101": (8, "bottleneck"),
    "resnet152": (12, "bottleneck"),
    "mobilenet_v2": (0, "mobilenet"),
    "mobilenet_v3_small": (0, "mobilenet"),
    "mobilenet_v3_large": (0, "mobilenet"),
    "efficientnet_b0": (0, "efficientnet"),
    "vgg16": (0, "vgg"),
    "densenet121": (0, "densenet"),
    "convnext_tiny": (0, "convnext"),
}


def _torch():
    try:
        import torch  # noqa: PLC0415 - imported lazily so the API runs without torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeUnavailable(
            "PyTorch is not installed. Run: pip install -r backend/requirements.txt"
        ) from exc
    return torch


class TorchRuntime(BaseRuntime):
    name = "torch"
    framework = "torch"

    @classmethod
    def availability(cls) -> tuple[bool, Optional[str]]:
        try:
            torch = _torch()
        except RuntimeUnavailable:
            return False, None
        return True, str(getattr(torch, "__version__", "unknown"))

    # ------------------------------------------------------------------
    def load(self, spec: ModelSpec) -> ModelHandle:
        torch = _torch()
        if spec.lfs_pointer:
            raise ModelLoadError(
                f"{spec.filename} is a Git LFS pointer, not the real weights. "
                "Run `git lfs install && git lfs pull` in the repository root."
            )
        if spec.path.stat().st_size < 1024:
            raise ModelLoadError(
                f"{spec.filename} is only {spec.path.stat().st_size} bytes — the weights look incomplete."
            )

        started = time.perf_counter()
        warnings: list[str] = []
        state_dict: Optional[dict[str, Any]] = None
        module: Any = None
        architecture = spec.architecture

        payload = self._load_payload(spec, torch)

        if hasattr(payload, "state_dict") and hasattr(payload, "eval"):
            # Case 2: a full pickled nn.Module.
            module = payload
            architecture = architecture or type(payload).__name__
        else:
            state_dict = _unwrap_state_dict(payload)
            if not state_dict:
                raise ModelLoadError(
                    f"{spec.filename}: expected a state_dict or nn.Module, got {type(payload).__name__}"
                )
            state_dict = _strip_prefixes(state_dict)

            if architecture in (None, "", "auto"):
                architecture, hint = self._infer_architecture(state_dict)
                if architecture is None:
                    raise ModelLoadError(hint or f"{spec.filename}: could not infer the architecture")
                spec.architecture = architecture
                warnings.append(f"architecture auto-detected as '{architecture}'")

            if architecture in TORCHVISION_IMAGE_MODELS:
                module = self._build_torchvision(architecture, _num_classes_from_state_dict(state_dict, torch), spec)
            else:
                module = self._build_custom(architecture, state_dict, spec, torch)

            module, missing, unexpected = _load_state_dict(module, state_dict, torch)
            if missing:
                warnings.append(f"{len(missing)} missing keys (first: {missing[0]})")
            if unexpected:
                warnings.append(f"{len(unexpected)} unexpected keys (first: {unexpected[0]})")

        module.eval()

        # torchvision models consume NCHW; Keras models consume NHWC. Unless the
        # manifest says otherwise, pick the convention of the framework the
        # checkpoint came from — a silent channel-order mismatch otherwise shows
        # up as nonsense predictions rather than an error.
        if (architecture or "") in TORCHVISION_IMAGE_MODELS and "data_format" not in spec.declared:
            spec.data_format = "channels_first"

        input_shape, output_shape, num_classes = self._introspect(module, state_dict, spec, torch)
        modality = _refine_modality(spec.modality, architecture, input_shape)
        spec.modality = modality

        return TorchModelHandle(
            spec=spec,
            model=module,
            input_shape=input_shape,
            output_shape=output_shape,
            num_classes=num_classes,
            already_probabilistic=_looks_probabilistic(module, torch),
            framework_version=str(getattr(torch, "__version__", "unknown")),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_payload(spec: ModelSpec, torch) -> Any:
        # TorchScript first: it is unambiguous and needs no architecture.
        if spec.architecture in (None, "torchscript", "auto"):
            try:
                scripted = torch.jit.load(str(spec.path), map_location="cpu")
                if spec.architecture == "torchscript":
                    return scripted
                # A real state_dict is not a TorchScript archive, so only accept
                # this when it genuinely behaved like one.
                if hasattr(scripted, "state_dict") and not spec.architecture:
                    scripted.eval()
                    return scripted
            except Exception:  # noqa: BLE001 - any failure means "not TorchScript"
                pass

        try:
            return torch.load(str(spec.path), map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - fall back to full unpickling
            try:
                return torch.load(str(spec.path), map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                raise ModelLoadError(f"{spec.filename}: torch.load failed ({type(exc).__name__}: {exc})") from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_architecture(state_dict: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        keys = set(state_dict)
        summary = _state_dict_summary(state_dict)

        # ResNet family: layerN.M.convK + a final fc layer.
        if "layer4.2.conv3.weight" in keys or any(k.startswith("layer4.") and k.endswith("conv3.weight") for k in keys):
            blocks = _max_block_index(keys, layer="layer4")
            arch = {2: "resnet50", 7: "resnet101", 11: "resnet152"}.get(blocks)
            if arch:
                return arch, None
        if any(k.startswith("layer4.") and k.endswith("conv2.weight") for k in keys):
            blocks = _max_block_index(keys, layer="layer4")
            arch = {1: "resnet18", 2: "resnet34"}.get(blocks)
            if arch:
                return arch, None

        # Pure MLP saved from nn.Sequential(Linear, ReLU, Linear, ...).
        if keys and all(NUMERIC_LAYER.match(k) for k in keys):
            weights = {k: v for k, v in state_dict.items() if k.endswith("weight")}
            if weights and all(_tensor_ndim(v) == 2 for v in weights.values()):
                return "mlp", None

        # Recurrent models.
        for kind, gate in (("lstm", 4), ("gru", 3)):
            first = f"{kind}.weight_ih_l0"
            if first in keys:
                return kind, None

        hint = (
            "Could not infer the architecture of this checkpoint. Add an entry to "
            "models/model_manifest.json, for example:\n\n"
            '  "models": {\n'
            '    "<filename>": {\n'
            '      "architecture": "cnn1d_lstm",\n'
            '      "modality": "landmarks",\n'
            '      "sequence_length": 30,\n'
            '      "input_dim": 63,\n'
            '      "hidden": [64, 128],\n'
            '      "lstm_hidden": 64,\n'
            '      "num_classes": 26\n'
            "    }\n  }\n\n"
            f"Checkpoint tensors:\n{summary}"
        )
        return None, hint

    # ------------------------------------------------------------------
    @staticmethod
    def _build_torchvision(architecture: str, num_classes: Optional[int], spec: ModelSpec) -> Any:
        try:
            import torchvision  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                f"'{architecture}' needs torchvision: pip install torchvision"
            ) from exc
        from torchvision import models  # noqa: PLC0415

        factory = getattr(models, architecture, None)
        if factory is None:
            raise ModelLoadError(f"torchvision has no model named '{architecture}'")
        kwargs: dict[str, Any] = {"weights": None}
        if num_classes:
            kwargs["num_classes"] = int(num_classes)
        kwargs.update({k: v for k, v in spec.extra.items() if k in ("dropout", "width_mult")})
        return factory(**kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_custom(architecture: str, state_dict: dict[str, Any], spec: ModelSpec, torch) -> Any:
        arch = (architecture or "").lower()
        extra = spec.extra
        num_classes = _num_classes_from_state_dict(state_dict, torch) or int(extra.get("num_classes") or 0)

        if arch == "mlp":
            return _build_sequential_mlp(state_dict, torch)

        if arch in ("lstm", "gru"):
            input_dim = int(extra.get("input_dim") or _recurrent_input_dim(state_dict, arch))
            hidden = int(extra.get("hidden") or _recurrent_hidden_dim(state_dict, arch))
            layers = int(extra.get("lstm_layers") or extra.get("layers") or _recurrent_layers(state_dict, arch))
            bidirectional = bool(extra.get("bidirectional", _recurrent_bidirectional(state_dict, arch)))
            return _RecurrentClassifier(
                torch, arch, input_dim, hidden, layers, bidirectional, num_classes,
                dropout=float(extra.get("dropout", 0.0)),
            )

        if arch in ("cnn1d_lstm", "cnn_lstm", "conv1d_lstm"):
            return _Conv1dLstm(
                torch,
                input_dim=int(extra.get("input_dim") or 63),
                channels=list(extra.get("channels") or [64, 128]),
                lstm_hidden=int(extra.get("lstm_hidden") or 64),
                lstm_layers=int(extra.get("lstm_layers") or 1),
                bidirectional=bool(extra.get("bidirectional", False)),
                num_classes=num_classes,
            )

        raise ModelLoadError(
            f"architecture '{architecture}' is not a built-in preset. Supported presets: "
            + ", ".join(sorted(list(TORCHVISION_IMAGE_MODELS) + ["mlp", "lstm", "gru", "cnn1d_lstm", "torchscript"]))
            + ". For anything else, export the model to TorchScript (torch.jit.script) and drop it in models/."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _introspect(module: Any, state_dict: Optional[dict[str, Any]], spec: ModelSpec, torch):
        input_shape: Optional[list[Any]] = None
        output_shape: Optional[list[Any]] = None
        num_classes: Optional[int] = None

        head = _find_head(module, torch)
        if head is not None and hasattr(head, "out_features"):
            num_classes = int(head.out_features)
            output_shape = [1, num_classes]

        if state_dict is not None and num_classes is None:
            num_classes = _num_classes_from_state_dict(state_dict, torch)
            if num_classes:
                output_shape = [1, num_classes]

        if (spec.architecture or "").lower() == "mlp" and state_dict is not None:
            input_dim = _mlp_input_dim(state_dict)
            if input_dim:
                input_shape = [1, input_dim]
        elif spec.modality == "image" or (spec.architecture or "") in TORCHVISION_IMAGE_MODELS:
            size = spec.image_size or 224
            input_shape = (
                [1, 3, size, size] if spec.data_format == "channels_first" else [1, size, size, 3]
            )
        elif spec.sequence_length:
            width = 63 + (1 if spec.include_handedness else 0)
            input_dim = spec.extra.get("input_dim")
            if isinstance(input_dim, int):
                width = input_dim
            input_shape = [1, spec.sequence_length, width]

        return input_shape, output_shape, num_classes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights", "net"):
            inner = payload.get(key)
            if isinstance(inner, dict) and inner:
                return inner
        # Some checkpoints are a bare {layer: tensor} mapping.
        if payload and all(hasattr(v, "shape") for v in payload.values()):
            return payload
    return {}


def _strip_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop DataParallel/DDP 'module.' wrappers so state dicts load cleanly."""
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _load_state_dict(module: Any, state_dict: dict[str, Any], torch) -> tuple[Any, list[str], list[str]]:
    try:
        result = module.load_state_dict(state_dict, strict=True)
        return module, list(result.missing_keys), list(result.unexpected_keys)
    except Exception as first_error:  # noqa: BLE001
        try:
            result = module.load_state_dict(state_dict, strict=False)
            return module, list(result.missing_keys), list(result.unexpected_keys)
        except Exception:  # noqa: BLE001
            raise ModelLoadError(
                f"state_dict does not match the '{type(module).__name__}' architecture: {first_error}"
            ) from first_error


def _tensor_ndim(value: Any) -> int:
    return int(len(getattr(value, "shape", ())))


def _max_block_index(keys: set[str], layer: str) -> int:
    indexes = []
    for key in keys:
        match = re.match(rf"^{re.escape(layer)}\.(\d+)\.", key)
        if match:
            indexes.append(int(match.group(1)))
    return max(indexes) if indexes else -1


def _state_dict_summary(state_dict: dict[str, Any], limit: int = 24) -> str:
    lines = [f"  {k}: {tuple(getattr(v, 'shape', ()))}" for k, v in list(state_dict.items())[:limit]]
    if len(state_dict) > limit:
        lines.append(f"  … {len(state_dict) - limit} more tensors")
    return "\n".join(lines)


def _num_classes_from_state_dict(state_dict: dict[str, Any], torch) -> Optional[int]:
    for key in ("fc.weight", "classifier.weight", "head.weight", "out.weight", "output.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and _tensor_ndim(tensor) == 2:
            return int(tensor.shape[0])
    for key, tensor in state_dict.items():
        if key.endswith("weight") and _tensor_ndim(tensor) == 2 and re.match(r"^\d+\.weight$", key):
            return int(tensor.shape[0])  # last Linear of a Sequential MLP wins below
    numeric = {int(m.group(1)): v for k, v in state_dict.items()
               if (m := NUMERIC_LAYER.match(k)) and k.endswith("weight") and _tensor_ndim(v) == 2}
    if numeric:
        return int(numeric[max(numeric)].shape[0])
    return None


def _mlp_input_dim(state_dict: dict[str, Any]) -> Optional[int]:
    """Input width of a sequential MLP: the in_features of its lowest-indexed Linear."""
    numeric = {
        int(m.group(1)): v
        for k, v in state_dict.items()
        if (m := NUMERIC_LAYER.match(k)) and k.endswith("weight") and _tensor_ndim(v) == 2
    }
    if not numeric:
        return None
    return int(numeric[min(numeric)].shape[1])


def _find_head(module: Any, torch) -> Any:
    for name in ("fc", "classifier", "head", "out", "output"):
        layer = getattr(module, name, None)
        if layer is not None and hasattr(layer, "out_features"):
            return layer
    last_linear = None
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            last_linear = child
    return last_linear


def _looks_probabilistic(module: Any, torch) -> bool:
    """True when the final layer already emits probabilities (softmax/sigmoid)."""
    layers = list(module.modules())
    for layer in reversed(layers):
        if isinstance(layer, (torch.nn.Softmax, torch.nn.LogSoftmax)):
            return True
        if isinstance(layer, torch.nn.Sigmoid):
            return True
    last = getattr(module, "activation_out", None)
    return isinstance(last, (torch.nn.Softmax, torch.nn.Sigmoid))


def _recurrent_input_dim(state_dict: dict[str, Any], kind: str) -> int:
    for key, value in state_dict.items():
        if key.startswith(f"{kind}.") and key.endswith("weight_ih_l0") and _tensor_ndim(value) == 2:
            return int(value.shape[1])
    raise ModelLoadError(f"could not read the input size of the {kind.upper()} from the checkpoint")


def _recurrent_hidden_dim(state_dict: dict[str, Any], kind: str) -> int:
    gate = 4 if kind == "lstm" else 3
    for key, value in state_dict.items():
        if key.startswith(f"{kind}.") and key.endswith("weight_ih_l0") and _tensor_ndim(value) == 2:
            return int(value.shape[0]) // gate
    raise ModelLoadError(f"could not read the hidden size of the {kind.upper()} from the checkpoint")


def _recurrent_layers(state_dict: dict[str, Any], kind: str) -> int:
    layers = {int(m.group(1)) for k in state_dict if (m := re.match(rf"^{kind}\.weight_ih_l(\d+)", k))}
    return max(layers) + 1 if layers else 1


def _recurrent_bidirectional(state_dict: dict[str, Any], kind: str) -> bool:
    return any(k.startswith(f"{kind}.") and "_reverse" in k for k in state_dict)


def _build_sequential_mlp(state_dict: dict[str, Any], torch) -> Any:
    """Rebuild nn.Sequential(Linear, ReLU, Linear, …) from its numeric keys."""
    indexes = sorted({int(NUMERIC_LAYER.match(k).group(1)) for k in state_dict if NUMERIC_LAYER.match(k)})
    layers: list[Any] = []
    previous_out: Optional[int] = None
    for index in range(0, max(indexes) + 1):
        weight = state_dict.get(f"{index}.weight")
        if weight is not None and _tensor_ndim(weight) == 2:
            out_features, in_features = (int(weight.shape[0]), int(weight.shape[1]))
            if previous_out is not None and in_features != previous_out:
                raise ModelLoadError(
                    f"MLP layer {index} expects {in_features} inputs but the previous layer outputs {previous_out}"
                )
            layers.append(torch.nn.Linear(in_features, out_features))
            previous_out = out_features
        else:
            layers.append(torch.nn.ReLU())
    if not any(isinstance(layer, torch.nn.Linear) for layer in layers):
        raise ModelLoadError("checkpoint does not look like a sequential MLP")
    return torch.nn.Sequential(*layers)


def _refine_modality(current: Modality, architecture: Optional[str], input_shape: Optional[list[Any]]) -> Modality:
    if current != "unknown":
        return current
    arch = (architecture or "").lower()
    if arch in TORCHVISION_IMAGE_MODELS:
        return "image"
    if input_shape and len(input_shape) == 3:
        return "landmarks"
    if input_shape and len(input_shape) == 2:
        # (batch, 63) is one hand pose; anything narrower is an engineered vector.
        width = input_shape[-1]
        return "landmarks" if width in (21, 42, 63, 64) else "features"
    return "unknown"


def _make_recurrent_class(torch):
    class RecurrentClassifier(torch.nn.Module):
        def __init__(self, kind, input_dim, hidden, layers, bidirectional, num_classes, dropout=0.0):
            super().__init__()
            rnn_cls = torch.nn.LSTM if kind == "lstm" else torch.nn.GRU
            self.rnn = rnn_cls(
                input_dim, hidden, num_layers=layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.fc = torch.nn.Linear(hidden * (2 if bidirectional else 1), num_classes)

        def forward(self, x):
            output, _ = self.rnn(x)
            return self.fc(output[:, -1, :])

    return RecurrentClassifier


def _make_conv1d_lstm_class(torch):
    class Conv1dLstm(torch.nn.Module):
        def __init__(self, input_dim, channels, lstm_hidden, lstm_layers, bidirectional, num_classes):
            super().__init__()
            blocks: list[Any] = []
            in_channels = input_dim
            for out_channels in channels:
                blocks += [
                    torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    torch.nn.BatchNorm1d(out_channels),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool1d(2),
                ]
                in_channels = out_channels
            self.conv = torch.nn.Sequential(*blocks)
            self.lstm = torch.nn.LSTM(
                in_channels, lstm_hidden, num_layers=lstm_layers,
                batch_first=True, bidirectional=bidirectional,
            )
            self.fc = torch.nn.Linear(lstm_hidden * (2 if bidirectional else 1), num_classes)

        def forward(self, x):
            features = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
            output, _ = self.lstm(features)
            return self.fc(output[:, -1, :])

    return Conv1dLstm


# Bind the real classes lazily (torch is imported on demand).
def _RecurrentClassifier(torch, *args, **kwargs):  # noqa: N802 - keeps call sites readable
    return _make_recurrent_class(torch)(*args, **kwargs)


def _Conv1dLstm(torch, *args, **kwargs):  # noqa: N802
    return _make_conv1d_lstm_class(torch)(*args, **kwargs)


class TorchModelHandle(ModelHandle):
    """A loaded torch module wrapped so `predict` accepts/returns numpy."""

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            torch_tensor = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32))
            output = self.model(torch_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.detach().cpu().numpy()
