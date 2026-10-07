# ---------------------------------------------------------------------------
# MODULE: Dataset store
#
# Samples recorded in the browser's Dataset Collection page can be pushed here
# so they survive a browser reset and can be turned into a training set
# (JSONL / CSV / .npz) for the offline training pipeline. One JSON object per
# line keeps the file append-only and cheap to stream.
# ---------------------------------------------------------------------------
from __future__ import annotations

import csv
import io
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import DatasetSamplePayload


class DatasetStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def read_all(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.is_file():
                return []
            samples: list[dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated final line
            return samples

    def add(self, samples: Iterable[DatasetSamplePayload], replace: bool = False) -> int:
        with self._lock:
            payload = [self._serialize(sample) for sample in samples]
            mode = "w" if replace else "a"
            with self.path.open(mode, encoding="utf-8") as handle:
                for record in payload:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return len(payload)

    def clear(self) -> None:
        with self._lock:
            if self.path.is_file():
                self.path.unlink()

    def replace_all(self, samples: Iterable[dict[str, Any]]) -> int:
        with self._lock:
            records = list(samples)
            with self.path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return len(records)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        samples = self.read_all()
        classes: dict[str, int] = {}
        for sample in samples:
            label = str(sample.get("label", "unlabelled"))
            classes[label] = classes.get(label, 0) + 1
        return {"total": len(samples), "classes": dict(sorted(classes.items()))}

    # ------------------------------------------------------------------
    def export_jsonl(self) -> bytes:
        with self._lock:
            return self.path.read_bytes() if self.path.is_file() else b""

    def export_csv(self) -> bytes:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["id", "label", "createdAt", "vector_dim", "vector"])
        for sample in self.read_all():
            vector = _sample_vector(sample)
            writer.writerow(
                [
                    sample.get("id", ""),
                    sample.get("label", ""),
                    sample.get("createdAt", ""),
                    len(vector),
                    " ".join(f"{v:.6f}" for v in vector),
                ]
            )
        return buffer.getvalue().encode("utf-8")

    def export_npz(self, sequence_length: int = 30) -> bytes:
        """
        Build (N, T, F) landmark sequences plus labels — the exact tensor shape
        the temporal models in models/ consume, so training needs no extra
        preprocessing step.
        """
        from .preprocessing import normalize_landmarks

        sequences: list[np.ndarray] = []
        labels: list[str] = []
        for sample in self.read_all():
            frames = sample.get("frames") or []
            if not frames:
                continue
            rows: list[np.ndarray] = []
            for frame in frames:
                hands = frame.get("hands") or []
                if not hands:
                    rows.append(np.zeros(63, dtype=np.float32))
                    continue
                points = np.asarray(
                    [[p["x"], p["y"], p.get("z", 0.0)] for p in hands[0]["landmarks"]],
                    dtype=np.float32,
                )
                rows.append(normalize_landmarks(points).reshape(-1).astype(np.float32))
            if len(rows) > sequence_length:
                rows = rows[-sequence_length:]
            while len(rows) < sequence_length:
                rows.insert(0, np.zeros(63, dtype=np.float32))
            sequences.append(np.stack(rows, axis=0))
            labels.append(str(sample.get("label", "unlabelled")))

        buffer = io.BytesIO()
        if sequences:
            array = np.stack(sequences, axis=0)
            unique = sorted(set(labels))
            mapping = {label: i for i, label in enumerate(unique)}
            np.savez_compressed(
                buffer,
                X=array,
                y=np.asarray([mapping[label] for label in labels], dtype=np.int64),
                labels=np.asarray(unique),
                sequence_length=np.asarray(sequence_length),
            )
        else:
            np.savez_compressed(
                buffer,
                X=np.zeros((0, sequence_length, 63), dtype=np.float32),
                y=np.zeros((0,), dtype=np.int64),
                labels=np.zeros((0,)),
                sequence_length=np.asarray(sequence_length),
            )
        return buffer.getvalue()

    # ------------------------------------------------------------------
    @staticmethod
    def _serialize(sample: DatasetSamplePayload) -> dict[str, Any]:
        return {
            "id": sample.id,
            "label": sample.label,
            "createdAt": sample.createdAt if sample.createdAt is not None else int(time.time() * 1000),
            "syncedAt": int(time.time() * 1000),
            "features": [dict(feature) for feature in sample.features],
            "frames": [frame.model_dump() for frame in sample.frames],
        }


def _sample_vector(sample: dict[str, Any]) -> list[float]:
    """Flatten a stored sample into the engineered-feature vector used by CSV export."""
    values: list[float] = []
    for feature in sample.get("features") or []:
        curl = feature.get("fingerCurl") or {}
        extended = feature.get("fingerExtended") or {}
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            values.append(float(curl.get(finger, 0.0)))
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            values.append(1.0 if extended.get(finger) else 0.0)
        values.append(float(feature.get("spread", 0.0)))
        values.append(float(feature.get("thumbIndexDistance", 0.0)))
        values.append(float(feature.get("palmOrientation", 0.0)))
    return values
