# ---------------------------------------------------------------------------
# Prediction endpoints — torch checkpoints
# ---------------------------------------------------------------------------
from __future__ import annotations

from conftest import png_base64, synthetic_frames, synthetic_hand


def test_landmarks_through_torch_mlp(client):
    """The .pth whose architecture must be auto-detected (sequential MLP over landmarks)."""
    response = client.post(
        "/api/predict/landmarks",
        json={"model_id": "best_model_new", "frames": synthetic_frames(8), "top_k": 3},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_id"] == "best_model_new"
    assert body["modality"] == "landmarks"
    assert body["frames_used"] == 8
    assert len(body["predictions"]) == 3
    assert body["top"]["label"] == body["predictions"][0]["label"]
    # Auto-detected MLP over 21*3 inputs, 5 output classes.
    info = client.get("/api/models/best_model_new").json()
    assert info["architecture"] == "mlp"
    assert info["input_shape"] == [1, 63]
    assert info["num_classes"] == 5
    assert info["inference_count"] == 1
    assert info["last_inference_ms"] > 0


def test_landmarks_scores_are_probabilities(client):
    response = client.post(
        "/api/predict/landmarks",
        json={"model_id": "best_model_new", "frames": synthetic_frames(8), "top_k": 5},
    )
    scores = [p["score"] for p in response.json()["predictions"]]
    assert scores == sorted(scores, reverse=True)
    assert 0.99 <= sum(scores) <= 1.01


def test_empty_frames_are_flagged_not_silently_wrong(client):
    frames = [{"hands": []} for _ in range(5)]
    response = client.post("/api/predict/landmarks", json={"model_id": "best_model_new", "frames": frames})
    assert response.status_code == 200
    assert any("zero-filled" in w for w in response.json()["warnings"])


def test_image_through_torch_resnet(client):
    response = client.post(
        "/api/predict/image",
        json={"model_id": "best_resnet18_isl", "image_base64": png_base64(), "top_k": 5},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_id"] == "best_resnet18_isl"
    assert body["modality"] == "image"
    assert len(body["predictions"]) == 5
    assert body["predictions"][0]["label"] == "class_0" or body["predictions"][0]["label"].startswith("class_")


def test_image_upload_multipart_route(client):
    import base64

    blob = base64.b64decode(png_base64())
    response = client.post(
        "/api/predict/image/upload",
        files={"file": ("frame.png", blob, "image/png")},
        data={"model_id": "best_resnet18_isl", "top_k": 2},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 2


def test_default_model_is_chosen_when_none_given(client):
    response = client.post("/api/predict/landmarks", json={"frames": synthetic_frames(4)})
    assert response.status_code == 200
    assert response.json()["model_id"]


def test_deterministic_output_for_identical_input(client):
    payload = {"model_id": "best_model_new", "frames": [{"hands": [synthetic_hand(seed=3)]} for _ in range(6)]}
    first = client.post("/api/predict/landmarks", json=payload).json()
    second = client.post("/api/predict/landmarks", json=payload).json()
    assert [p["score"] for p in first["predictions"]] == [p["score"] for p in second["predictions"]]
