# ---------------------------------------------------------------------------
# Shared test fixtures
#
# The real checkpoints in models/ are Git LFS objects, so the suite builds
# small-but-real checkpoints of the same shapes: a torchvision ResNet-18
# state_dict, a sequential-MLP state_dict, a Keras landmark model (.keras) and
# a Keras image model (.h5), plus a fake LFS pointer. Every test therefore
# exercises the same loader code paths the production models will hit.
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

NUM_CLASSES = 5
SEQUENCE_LENGTH = 12
LANDMARK_DIM = 63


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("models")

    _write_resnet(directory / "best_resnet18_isl.pth")
    _write_mlp(directory / "best_model_new.pth")
    _write_keras_landmarks(directory / "sign_language_model.keras")
    _write_keras_image(directory / "sign_language_model.h5")
    _write_lfs_pointer(directory / "movinet_isl_final_best.keras")

    (directory / "labels.json").write_text(
        json.dumps([f"class_{i}" for i in range(NUM_CLASSES)]),
        encoding="utf-8",
    )
    (directory / "model_manifest.json").write_text(
        json.dumps(
            {
                "defaults": {"normalization": "wrist", "image_size": 64, "sequence_length": SEQUENCE_LENGTH},
                "models": {
                    "best_resnet18_isl.pth": {
                        "architecture": "resnet18",
                        "modality": "image",
                        "image_size": 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture()
def env(tmp_path: Path, model_dir: Path):
    """Point the app at the fixture models and a scratch data directory."""
    data_dir = tmp_path / "data"
    os.environ["SIGNORA_MODEL_DIR"] = str(model_dir)
    os.environ["SIGNORA_DATA_DIR"] = str(data_dir)
    os.environ["SIGNORA_PRELOAD_MODELS"] = "false"
    from app.config import get_settings

    get_settings.cache_clear()
    yield {"model_dir": model_dir, "data_dir": data_dir}
    for key in ("SIGNORA_MODEL_DIR", "SIGNORA_DATA_DIR", "SIGNORA_PRELOAD_MODELS"):
        os.environ.pop(key, None)
    get_settings.cache_clear()


@pytest.fixture()
def client(env):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Checkpoint builders
# ---------------------------------------------------------------------------
def _write_resnet(path: Path) -> None:
    import torch
    from torchvision import models

    model = models.resnet18(weights=None, num_classes=NUM_CLASSES)
    torch.save({"state_dict": model.state_dict(), "epoch": 12}, path)


def _write_mlp(path: Path) -> None:
    """nn.Sequential(Linear, ReLU, Linear, ReLU, Linear) over landmark vectors."""
    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(LANDMARK_DIM, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, NUM_CLASSES),
    )
    torch.save(model.state_dict(), path)


def _write_keras_landmarks(path: Path) -> None:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(SEQUENCE_LENGTH, LANDMARK_DIM)),
            tf.keras.layers.LSTM(16, return_sequences=False),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.save(path)


def _write_keras_image(path: Path) -> None:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NUM_CLASSES),  # raw logits on purpose
        ]
    )
    model.save(path)


def _write_lfs_pointer(path: Path) -> None:
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:5e22a2923bc1c0d914076c9dcbefac1e6090d7e4f6a628e25abfe5065df7f5f4\n"
        "size 24313705\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------
def synthetic_hand(seed: int = 0, handedness: str = "Right") -> dict:
    """A plausible 21-landmark hand: a palm arc plus five fingers that wiggle."""
    rng = np.random.default_rng(seed)
    landmarks = []
    for index in range(21):
        angle = (index / 21) * math.pi
        landmarks.append(
            {
                "x": round(float(0.4 + 0.15 * math.cos(angle) + rng.normal(0, 0.01)), 5),
                "y": round(float(0.5 + 0.15 * math.sin(angle) + rng.normal(0, 0.01)), 5),
                "z": round(float(rng.normal(0, 0.01)), 5),
            }
        )
    return {"landmarks": landmarks, "handedness": handedness, "score": 0.97}


def synthetic_frames(count: int = 6) -> list[dict]:
    return [{"hands": [synthetic_hand(seed=i)]} for i in range(count)]


def png_base64(size: int = 48) -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).integers(0, 255, (size, size, 3))).astype("uint8")).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
