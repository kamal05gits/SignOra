# ---------------------------------------------------------------------------
# MODULE: Runtime base classes + resolved model specification
#
# A "spec" is everything known about a model file *before* loading it; a
# "handle" is the loaded, callable model. Keeping the two apart means the
# /api/models endpoint can describe every model in models/ without paying the
# cost (or hitting the failure) of loading it.
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..schemas import Framework, Modality


@dataclass
class ModelSpec:
    """Static description of one model file plus its resolved preprocessing config."""

    id: str
    path: Path
    filename: str
    framework: Framework
    format: str
    size_bytes: int
    modality: Modality = "unknown"
    architecture: Optional[str] = None
    sequence_length: Optional[int] = None
    image_size: Optional[int] = None
    video_frames: Optional[int] = None
    channel_order: str = "RGB"
    data_format: str = "channels_last"
    scale: float = 1.0 / 255.0
    mean: Optional[list[float]] = None
    std: Optional[list[float]] = None
    normalization: str = "wrist"
    include_handedness: bool = False
    labels: Optional[list[str]] = None
    glosses: Optional[list[str]] = None
    labels_source: Optional[str] = None
    # Manifest keys that were explicitly set, so a runtime can tell "the user
    # chose channels_last" apart from "nobody said, use the framework default".
    declared: set[str] = field(default_factory=set)
    lfs_pointer: bool = False
    hint: Optional[str] = None
    preimport: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def label_at(self, index: int) -> tuple[str, Optional[str]]:
        if self.labels and 0 <= index < len(self.labels):
            gloss = self.glosses[index] if self.glosses and index < len(self.glosses) else None
            return self.labels[index], gloss
        return f"class_{index}", None


@dataclass
class ModelHandle:
    """A loaded model, ready for inference."""

    spec: ModelSpec
    model: Any
    input_shape: Optional[list[Any]] = None
    output_shape: Optional[list[Any]] = None
    num_classes: Optional[int] = None
    already_probabilistic: bool = False
    framework_version: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RuntimeUnavailable(RuntimeError):
    """Raised when the ML framework backing a runtime is not installed."""


class ModelLoadError(RuntimeError):
    """Raised when a model file cannot be turned into a callable model."""


class BaseRuntime:
    """Interface implemented by the torch and keras runtimes."""

    name: str = "base"
    framework: Framework = "torch"

    @classmethod
    def availability(cls) -> tuple[bool, Optional[str]]:
        """Return (installed, version_or_none)."""
        raise NotImplementedError

    def load(self, spec: ModelSpec) -> ModelHandle:
        raise NotImplementedError

    @staticmethod
    def shape_to_list(shape: Any) -> Optional[list[Any]]:
        """Normalize torch.Size / keras TensorShape / tuple into a JSON-safe list."""
        if shape is None:
            return None
        try:
            return [None if d is None or d == -1 else int(d) for d in shape]
        except TypeError:
            return None
