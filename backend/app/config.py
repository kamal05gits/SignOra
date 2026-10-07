# ---------------------------------------------------------------------------
# MODULE: Backend configuration
#
# Everything the server needs to know is read from environment variables with
# sensible defaults that work when the backend is started from the repository
# root (`uvicorn backend.app.main:app`). See backend/README.md for the full
# list of variables and the .env.example template.
# ---------------------------------------------------------------------------
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_EXTENSIONS = {
    ".pth": "torch",
    ".pt": "torch",
    ".keras": "keras",
    ".h5": "keras",
}


class Settings(BaseSettings):
    """Runtime configuration for the SignOra inference backend."""

    model_config = SettingsConfigDict(
        env_prefix="SIGNORA_",
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Where the trained model weights live (Git LFS in this repository).
    model_dir: Path = REPO_ROOT / "models"
    # Where captured dataset samples / exports are written.
    data_dir: Path = REPO_ROOT / "backend" / "data"

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]

    # Load models eagerly on startup instead of on first request.
    preload_models: bool = False
    # How many models may stay resident in memory at once (LRU eviction).
    max_loaded_models: int = 3

    # Default number of classes returned by /predict endpoints.
    default_top_k: int = 5
    # Number of landmark frames a temporal model expects (T). Overridable per
    # model through models/model_manifest.json.
    default_sequence_length: int = 30
    # Square edge length images are resized to when a model does not declare one.
    default_image_size: int = 224
    # Number of video frames sampled for video models when not declared.
    default_video_frames: int = 8

    api_title: str = "SignOra Inference API"
    version: str = "1.0.0"

    @property
    def manifest_path(self) -> Path:
        return self.model_dir / "model_manifest.json"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "dataset" / "samples.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
