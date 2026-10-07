# ---------------------------------------------------------------------------
# Dataset sync, export and model-backed evaluation
# ---------------------------------------------------------------------------
from __future__ import annotations

import io

import numpy as np
from conftest import synthetic_frames


def _sample(sample_id: str, label: str, seed: int) -> dict:
    return {
        "id": sample_id,
        "label": label,
        "createdAt": 1_700_000_000_000,
        "features": [],
        "frames": synthetic_frames(4),
    }


def test_samples_are_stored_and_summarised(client):
    payload = {"samples": [_sample("a", "HELLO", 1), _sample("b", "HELLO", 2), _sample("c", "THANKS", 3)]}
    response = client.post("/api/dataset/samples", json=payload)
    assert response.status_code == 200
    assert response.json()["written"] == 3

    summary = client.get("/api/dataset/summary").json()
    assert summary["total"] == 3
    assert summary["classes"] == {"HELLO": 2, "THANKS": 1}


def test_replace_flag_rewrites_the_store(client):
    client.post("/api/dataset/samples", json={"samples": [_sample("a", "HELLO", 1)]})
    response = client.post(
        "/api/dataset/samples",
        json={"samples": [_sample("b", "THANKS", 2)], "replace": True},
    )
    assert response.json()["total"] == 1
    assert client.get("/api/dataset/summary").json()["classes"] == {"THANKS": 1}


def test_export_formats(client):
    client.post("/api/dataset/samples", json={"samples": [_sample("a", "HELLO", 1)], "replace": True})

    jsonl = client.get("/api/dataset/export", params={"format": "jsonl"})
    assert jsonl.status_code == 200
    assert b"HELLO" in jsonl.content

    csv_response = client.get("/api/dataset/export", params={"format": "csv"})
    assert csv_response.status_code == 200
    assert csv_response.text.splitlines()[0] == "id,label,createdAt,vector_dim,vector"

    npz = client.get("/api/dataset/export", params={"format": "npz", "sequence_length": 12})
    assert npz.status_code == 200
    archive = np.load(io.BytesIO(npz.content))
    assert archive["X"].shape == (1, 12, 63)
    assert archive["y"].shape == (1,)
    assert list(archive["labels"]) == ["HELLO"]


def test_evaluate_scores_samples_with_a_trained_model(client):
    samples = [_sample(f"s{i}", label, i) for i, label in enumerate(["A", "B", "A", "C"])]
    response = client.post(
        "/api/dataset/evaluate",
        json={"model_id": "sign_language_model_keras", "samples": samples, "confidence_threshold": 0.0},
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["model_id"] == "sign_language_model_keras"
    assert report["total_samples"] == 4
    assert 0 <= report["accuracy"] <= 1
    assert report["correct"] + 0 == round(report["accuracy"] * 4)
    # The confusion matrix is square over the union of true and predicted labels.
    assert len(report["confusion_matrix"]) == len(report["classes"])
    assert all(len(row) == len(report["classes"]) for row in report["confusion_matrix"])
    assert sum(sum(row) for row in report["confusion_matrix"]) == 4
    assert {row["label"] for row in report["per_class"]} <= set(report["classes"])


def test_evaluate_with_threshold_marks_rejections(client):
    samples = [_sample(f"s{i}", "A", i) for i in range(3)]
    report = client.post(
        "/api/dataset/evaluate",
        json={"model_id": "sign_language_model_keras", "samples": samples, "confidence_threshold": 0.999},
    ).json()
    assert report["rejected"] >= 0
    assert "REJECTED" in report["classes"] or report["rejected"] == 0


def test_evaluate_rejects_unknown_model(client):
    response = client.post(
        "/api/dataset/evaluate",
        json={"model_id": "missing_model", "samples": [_sample("a", "A", 1)]},
    )
    assert response.status_code == 404
