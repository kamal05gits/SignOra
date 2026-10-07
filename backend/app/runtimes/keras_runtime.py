# ---------------------------------------------------------------------------
# MODULE: Keras / TensorFlow runtime
#
# Loads .keras (Keras 3 zip format) and .h5 (legacy HDF5) models. Unlike a
# PyTorch checkpoint these files carry their own graph, so the architecture
# never has to be declared — but two things are still resolved from the
# manifest when present:
#
#   * `preimport` — modules that must be imported before loading so that
#     custom layers (e.g. MoViNet's blocks from `tf-models-official`) are
#     registered with Keras. Without them `load_model` raises
#     "Unknown layer" and the error message does not say what to install.
#   * preprocessing — channel order / scale / mean / std, which are not
#     recoverable from the graph at all.
# ---------------------------------------------------------------------------
from __future__ import annotations

import importlib
from typing import Any, Optional

import numpy as np

from ..schemas import Modality
from .base import BaseRuntime, ModelHandle, ModelLoadError, ModelSpec, RuntimeUnavailable

# Landmark vector widths that mean "one hand per frame".
LANDMARK_WIDTHS = {21, 42, 63, 64, 66, 84, 126, 129}
PROBABILISTIC_ACTIVATIONS = {"softmax", "sigmoid"}


def _keras() -> tuple[Any, Any, Optional[str]]:
    """Prefer the Keras bundled with TensorFlow; fall back to standalone Keras."""
    try:
        import tensorflow as tf  # noqa: PLC0415

        return tf.keras, tf, str(getattr(tf, "__version__", "unknown"))
    except ImportError:
        pass
    try:
        import keras  # noqa: PLC0415

        return keras, keras, str(getattr(keras, "__version__", "unknown"))
    except ImportError as exc:
        raise RuntimeUnavailable(
            "Neither TensorFlow nor Keras is installed. Run: pip install -r backend/requirements.txt"
        ) from exc


class KerasModelHandle(ModelHandle):
    """A loaded Keras model wrapped so `predict` accepts/returns numpy."""

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        output = self.model(np.ascontiguousarray(tensor, dtype=np.float32), training=False)
        if isinstance(output, (list, tuple)):
            output = output[0]
        array = getattr(output, "numpy", None)
        return np.asarray(array() if callable(array) else output, dtype=np.float32)


class KerasRuntime(BaseRuntime):
    name = "keras"
    framework = "keras"

    @classmethod
    def availability(cls) -> tuple[bool, Optional[str]]:
        try:
            _, _, version = _keras()
        except RuntimeUnavailable:
            return False, None
        return True, version

    # ------------------------------------------------------------------
    def load(self, spec: ModelSpec) -> ModelHandle:
        keras, _, version = _keras()
        if spec.lfs_pointer:
            raise ModelLoadError(
                f"{spec.filename} is a Git LFS pointer, not the real weights. "
                "Run `git lfs install && git lfs pull` in the repository root."
            )

        warnings: list[str] = []
        for module_name in spec.preimport:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - a failed optional import is a warning
                warnings.append(
                    f"could not import '{module_name}' ({type(exc).__name__}: {exc}); "
                    "custom layers from that package will not deserialize"
                )

        try:
            model = keras.models.load_model(str(spec.path), compile=False)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "Unknown layer" in message or "not recognized" in message.lower():
                message += (
                    " — this model uses custom layers. Install the package that defines them and add it to "
                    "the manifest, e.g. \"preimport\": [\"official.projects.movinet.modeling.movinet\"]"
                )
            raise ModelLoadError(f"{spec.filename}: {type(exc).__name__}: {message}") from exc

        input_shape = _first_input_shape(model)
        output_shape = _first_output_shape(model)
        num_classes = _num_classes(output_shape)

        if spec.modality == "unknown":
            spec.modality = _modality_from_shape(input_shape)
        if spec.modality == "image" and not spec.image_size:
            spec.image_size = _square_edge(input_shape)
        if spec.modality == "video" and not spec.video_frames:
            spec.video_frames = _frame_count(input_shape)
        if spec.modality == "landmarks" and not spec.sequence_length:
            spec.sequence_length = _sequence_length(input_shape)

        if spec.labels and num_classes and len(spec.labels) != num_classes:
            warnings.append(
                f"labels file declares {len(spec.labels)} classes but the model outputs {num_classes}"
            )

        return KerasModelHandle(
            spec=spec,
            model=model,
            input_shape=BaseRuntime.shape_to_list(input_shape),
            output_shape=BaseRuntime.shape_to_list(output_shape),
            num_classes=num_classes,
            already_probabilistic=_looks_probabilistic(model),
            framework_version=version,
            # Only real problems belong here: these strings are echoed in every
            # prediction response, and the load time is already in /api/models.
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Shape introspection
# ---------------------------------------------------------------------------
def _first_input_shape(model: Any) -> Optional[tuple]:
    for accessor in ("inputs", "input"):
        value = getattr(model, accessor, None)
        try:
            if isinstance(value, (list, tuple)) and value:
                return tuple(value[0].shape)
            if value is not None and hasattr(value, "shape"):
                return tuple(value.shape)
        except Exception:  # noqa: BLE001 - some subclassed models expose neither
            continue
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        return tuple(shape[0])
    return tuple(shape) if shape else None


def _first_output_shape(model: Any) -> Optional[tuple]:
    for accessor in ("outputs", "output"):
        value = getattr(model, accessor, None)
        try:
            if isinstance(value, (list, tuple)) and value:
                return tuple(value[0].shape)
            if value is not None and hasattr(value, "shape"):
                return tuple(value.shape)
        except Exception:  # noqa: BLE001
            continue
    shape = getattr(model, "output_shape", None)
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        return tuple(shape[0])
    return tuple(shape) if shape else None


def _num_classes(output_shape: Optional[tuple]) -> Optional[int]:
    if not output_shape:
        return None
    last = output_shape[-1]
    if last is None:
        return None
    try:
        return int(last)
    except (TypeError, ValueError):
        return None


def _square_edge(input_shape: Optional[tuple]) -> Optional[int]:
    if not input_shape or len(input_shape) != 4:
        return None
    height, width = input_shape[1], input_shape[2]
    if isinstance(height, int) and isinstance(width, int):
        return int(max(height, width))
    return None


def _frame_count(input_shape: Optional[tuple]) -> Optional[int]:
    if not input_shape:
        return None
    if len(input_shape) == 5 and isinstance(input_shape[1], int):
        return int(input_shape[1])
    return None


def _sequence_length(input_shape: Optional[tuple]) -> Optional[int]:
    if input_shape and len(input_shape) == 3 and isinstance(input_shape[1], int):
        return int(input_shape[1])
    return None


def _modality_from_shape(input_shape: Optional[tuple]) -> Modality:
    """Map a Keras input shape onto the modality the API exposes."""
    if not input_shape:
        return "unknown"
    dims = [d for d in input_shape[1:] if d is not None]
    rank = len(input_shape)

    if rank == 5:  # (batch, frames, H, W, C)
        return "video"
    if rank == 4:
        first, second = dims[0], dims[1] if len(dims) > 1 else None
        third = dims[2] if len(dims) > 2 else None
        # Frames-first video: a small leading dim, then two equal spatial dims.
        if (
            isinstance(first, int)
            and isinstance(second, int)
            and isinstance(third, int)
            and first <= 32
            and second >= 64
            and second == third
        ):
            return "video"
        return "image"
    if rank == 3:
        width = dims[-1] if dims else None
        if isinstance(width, int) and width in LANDMARK_WIDTHS:
            return "landmarks"
        if isinstance(width, int) and width in (1, 3):
            return "video"
        return "landmarks"
    if rank == 2:
        width = dims[-1] if dims else None
        return "landmarks" if isinstance(width, int) and width in LANDMARK_WIDTHS else "features"
    return "unknown"


def _looks_probabilistic(model: Any) -> bool:
    """True when the output layer already normalizes to probabilities."""
    layers = list(getattr(model, "layers", []) or [])
    if not layers:
        return False
    last = layers[-1]
    name = type(last).__name__.lower()
    if name in ("softmax", "activation"):
        activation = getattr(last, "activation", None)
        return getattr(activation, "__name__", str(activation)).lower() in PROBABILISTIC_ACTIVATIONS
    config = getattr(last, "get_config", None)
    if callable(config):
        try:
            activation = str(last.get_config().get("activation", "")).lower()
        except Exception:  # noqa: BLE001
            activation = ""
        return activation in PROBABILISTIC_ACTIVATIONS
    return name in ("softmax", "sigmoid")
