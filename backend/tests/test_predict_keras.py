# ---------------------------------------------------------------------------
# Prediction endpoints — Keras checkpoints (.keras and .h5)
# ---------------------------------------------------------------------------
from __future__ import annotations

from conftest import png_base64, synthetic_frames


def test_keras_landmark_model_infers_modality_from_graph(client):
    response = client.post(
        "/api/predict/landmarks",
        json={"model_id": "sign_language_model_keras", "frames": synthetic_frames(12), "top_k": 5},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_id"] == "sign_language_model_keras"
    assert body["modality"] == "landmarks"

    info = client.get("/api/models/sign_language_model_keras").json()
    assert info["num_classes"] == 5
    # The graph ends in a softmax, so no second softmax may be applied.
    assert info["input_shape"] == [None, 12, 63]


def test_keras_landmark_predictions_sum_to_one(client):
    response = client.post(
        "/api/predict/landmarks",
        json={"model_id": "sign_language_model_keras", "frames": synthetic_frames(12)},
    )
    scores = [p["score"] for p in response.json()["predictions"]]
    assert 0.99 <= sum(scores) <= 1.01


def test_keras_h5_image_model_infers_image_modality(client):
    response = client.post(
        "/api/predict/image",
        json={"model_id": "sign_language_model_h5", "image_base64": png_base64(), "top_k": 3},
    )
    assert response.status_code == 200, response.text
    info = client.get("/api/models/sign_language_model_h5").json()
    assert info["modality"] == "image"
    assert info["num_classes"] == 5
    assert info["input_shape"] == [None, 64, 64, 3]


def test_short_sequence_is_padded_not_rejected(client):
    """3 frames for a model that wants 12: zero-padded, with a warning."""
    response = client.post(
        "/api/predict/landmarks",
        json={"model_id": "sign_language_model_keras", "frames": synthetic_frames(3)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["frames_used"] == 3


def test_long_sequence_is_truncated(client):
    response = client.post(
        "/api/predict/landmarks",
        json={"model_id": "sign_language_model_keras", "frames": synthetic_frames(40)},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 5
