# ---------------------------------------------------------------------------
# MODULE: Preprocessing
#
# Converts what the browser sends (base64 images / MediaPipe landmark frames)
# into the numeric tensors the models expect. Every step here is declared in
# the model manifest so that training-time and inference-time preprocessing
# can never silently diverge — the same normalization used by the frontend's
# src/lib/geometry.ts (wrist origin, wrist→middle-MCP scale) is the default.
# ---------------------------------------------------------------------------
from __future__ import annotations

import base64
import binascii
import io
import re
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from .schemas import Frame, Hand, Landmark

WRIST = 0
MIDDLE_MCP = 9
DATA_URL_PREFIX = re.compile(r"^data:[a-z/+.-]+;base64,", re.IGNORECASE)


class PreprocessingError(ValueError):
    """Raised when an incoming payload cannot be turned into a valid tensor."""


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def decode_image(payload: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL prefixed) image into an HxWx3 uint8 array."""
    raw = DATA_URL_PREFIX.sub("", payload.strip())
    # Tolerate whitespace/newlines produced by some base64 encoders.
    raw = "".join(raw.split())
    try:
        blob = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PreprocessingError(f"image_base64 is not valid base64: {exc}") from exc
    if not blob:
        raise PreprocessingError("image_base64 decoded to zero bytes")
    try:
        image = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception as exc:  # PIL raises many types depending on the codec
        raise PreprocessingError(f"could not decode image ({type(exc).__name__}: {exc})") from exc
    return np.asarray(image, dtype=np.uint8)


def prepare_image(
    array: np.ndarray,
    size: int = 224,
    channel_order: str = "RGB",
    scale: float = 1.0 / 255.0,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    data_format: str = "channels_last",
) -> np.ndarray:
    """Resize + normalize a single image into a (1, ...) float32 batch."""
    image = Image.fromarray(array)
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)

    tensor = np.asarray(image, dtype=np.float32)
    if channel_order.upper() == "BGR":
        tensor = tensor[:, :, ::-1]
    tensor = tensor * scale
    if mean is not None:
        tensor = tensor - np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    if std is not None:
        std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        std_arr[std_arr == 0] = 1.0
        tensor = tensor / std_arr
    tensor = np.expand_dims(tensor, axis=0)
    if data_format == "channels_first":
        tensor = np.transpose(tensor, (0, 3, 1, 2))
    return np.ascontiguousarray(tensor, dtype=np.float32)


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------
def landmarks_matrix(hand: Hand) -> np.ndarray:
    """(21, 3) float32 matrix of a hand's raw landmarks."""
    return np.asarray([[p.x, p.y, p.z] for p in hand.landmarks], dtype=np.float32)


def pick_hand(frame: Frame, mode: str = "first") -> Optional[Hand]:
    """Select which hand of a frame feeds the model."""
    if not frame.hands:
        return None
    if mode == "left":
        for hand in frame.hands:
            if hand.handedness == "Left":
                return hand
    if mode == "right":
        for hand in frame.hands:
            if hand.handedness == "Right":
                return hand
    if mode == "dominant":
        return max(frame.hands, key=lambda h: h.score)
    return frame.hands[0]


def normalize_landmarks(points: np.ndarray, mode: str = "wrist") -> np.ndarray:
    """
    Make landmarks position/scale invariant.

    - "wrist": subtract the wrist, divide by the wrist→middle-MCP distance
      (identical to `handScale` in src/lib/geometry.ts).
    - "bbox": subtract the min corner, divide by the bounding-box diagonal.
    - "none": pass through unchanged (use for models trained on raw 0..1 coords).
    """
    if mode == "none":
        return points
    if mode == "bbox":
        mins = points.min(axis=0)
        span = float(np.linalg.norm(points.max(axis=0) - mins)) or 1.0
        return (points - mins) / span
    origin = points[WRIST].copy()
    shifted = points - origin
    scale = float(np.linalg.norm(points[MIDDLE_MCP] - origin))
    if scale < 1e-6:
        scale = 1.0
    return shifted / scale


def hand_vector(
    hand: Hand,
    width: int = 63,
    normalization: str = "wrist",
    include_handedness: bool = False,
) -> np.ndarray:
    """
    Flatten one hand into the vector width the model was trained with.

      63 -> x, y, z of all 21 landmarks        (the common default)
      64 -> the same plus a Right=1/Left=0 flag
      42 -> x, y only                          (models trained without depth)
      21 -> x only                             (rare, but cheap to support)
    """
    points = normalize_landmarks(landmarks_matrix(hand), normalization)
    if width == 42:
        vector = points[:, :2].reshape(-1)
    elif width == 21:
        vector = points[:, :1].reshape(-1)
    else:
        vector = points.reshape(-1)
    if include_handedness or width == 64:
        vector = np.concatenate(
            [vector[:63], np.asarray([1.0 if hand.handedness == "Right" else 0.0], dtype=np.float32)]
        )
    return vector.astype(np.float32)


def vector_width(width: Optional[int], include_handedness: bool) -> int:
    """Effective per-frame width for a declared (or default) model width."""
    if width in (21, 42, 63, 64):
        return int(width)
    return 63 + (1 if include_handedness else 0)


def frames_to_sequence(
    frames: Iterable[Frame],
    length: int,
    hand: str = "first",
    normalization: str = "wrist",
    include_handedness: bool = False,
    width: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """
    Build a (1, T, F) float32 tensor from a list of landmark frames.

    Returns the tensor and the number of frames that actually contained a hand
    (the rest are zero-filled). Short sequences are left-padded with zeros and
    long ones are truncated from the end, which is the convention used by most
    ISL landmark classifiers.
    """
    per_frame = vector_width(width, include_handedness)
    rows: list[np.ndarray] = []
    filled = 0
    for frame in frames:
        selected = pick_hand(frame, hand)
        if selected is None:
            rows.append(np.zeros(per_frame, dtype=np.float32))
            continue
        filled += 1
        rows.append(hand_vector(selected, per_frame, normalization, include_handedness))

    if not rows:
        raise PreprocessingError("no frames supplied")

    if len(rows) > length:
        rows = rows[-length:]
    while len(rows) < length:
        rows.insert(0, np.zeros(per_frame, dtype=np.float32))

    tensor = np.stack(rows, axis=0)[None, :, :].astype(np.float32)
    return np.ascontiguousarray(tensor), filled


def last_frame_vector(
    frames: Sequence[Frame],
    hand: str = "first",
    normalization: str = "wrist",
    include_handedness: bool = False,
    width: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """
    Build a (1, F) tensor for models that classify a single frame at a time
    (a plain CNN/MLP over one hand pose). The newest frame that actually
    contains a hand is used, so a dropped detection does not feed zeros.
    """
    per_frame = vector_width(width, include_handedness)
    for frame in reversed(list(frames)):
        selected = pick_hand(frame, hand)
        if selected is not None:
            return (
                hand_vector(selected, per_frame, normalization, include_handedness)[None, :].astype(np.float32),
                1,
            )
    return np.zeros((1, per_frame), dtype=np.float32), 0


def landmark_tensor(
    frames: Sequence[Frame],
    *,
    input_shape: Optional[Sequence[Any]] = None,
    sequence_length: Optional[int] = None,
    hand: str = "first",
    normalization: str = "wrist",
    include_handedness: bool = False,
) -> tuple[np.ndarray, int, int]:
    """
    Adapt a window of frames to whatever input shape the model declares.

    A rank-2 model (batch, F) classifies one hand pose, so the newest frame
    with a hand is used; a rank-3 model (batch, T, F) gets the whole window.
    The per-frame width follows the model too (63 / 64 / 42 / 21).

    Returns (tensor, frames_that_contained_a_hand, effective_length).
    """
    rank = len(input_shape) if input_shape else 3
    width: Optional[int] = None
    if input_shape:
        try:
            width = int(input_shape[-1])
        except (TypeError, ValueError):
            width = None

    if rank == 2:
        tensor, filled = last_frame_vector(
            frames, hand=hand, normalization=normalization,
            include_handedness=include_handedness, width=width,
        )
        return tensor, filled, 1

    length = sequence_length
    if length is None and input_shape and len(input_shape) >= 2 and isinstance(input_shape[1], int):
        length = int(input_shape[1])
    length = length or len(frames)

    tensor, filled = frames_to_sequence(
        frames, length=length, hand=hand, normalization=normalization,
        include_handedness=include_handedness, width=width,
    )
    return tensor, filled, length


ENGINEERED_FEATURE_DIM = 14


def features_to_vector(features: Sequence[dict], normalization: str = "wrist") -> np.ndarray:
    """
    Turn the frontend's serialized HandFeatureSet objects into a flat float32
    vector, for models trained on engineered features instead of raw landmarks.

    Layout (14 values per hand): 5 finger curls, 5 extension flags, spread,
    thumb-index distance, palm orientation, handedness. This is exactly
    HandFeatureSet minus `raw`, so training and inference agree by construction.
    """
    if not features:
        raise PreprocessingError("no features supplied")
    values: list[float] = []
    for feature in features:
        curl = feature.get("fingerCurl") or {}
        extended = feature.get("fingerExtended") or {}
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            values.append(float(curl.get(finger, 0.0)))
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            values.append(1.0 if extended.get(finger) else 0.0)
        values.append(float(feature.get("spread", 0.0)))
        values.append(float(feature.get("thumbIndexDistance", 0.0)))
        values.append(float(feature.get("palmOrientation", 0.0)))
        values.append(1.0 if feature.get("handedness") == "Right" else 0.0)
    return np.asarray([values], dtype=np.float32)


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def ensure_probabilities(logits: np.ndarray, already_probabilistic: bool) -> np.ndarray:
    """Apply a softmax unless the model's final layer already produced probabilities."""
    if already_probabilistic:
        return logits
    return softmax(logits)


def top_k_indices(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    k = max(1, min(k, flat.size))
    order = np.argsort(flat)[::-1][:k]
    return [(int(i), float(flat[i])) for i in order]
