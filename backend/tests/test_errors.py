# ---------------------------------------------------------------------------
# Error handling: every failure mode must map to a status the UI can act on
# ---------------------------------------------------------------------------
from __future__ import annotations

from conftest import png_base64, synthetic_frames


def test_unknown_model_returns_404(client):
    response = client.post("/api/predict/landmarks", json={"model_id": "nope", "frames": synthetic_frames(4)})
    assert response.status_code == 404
    assert "available_models" in response.json()


def test_missing_weights_returns_503_with_the_fix(client):
    response = client.post(
        "/api/predict/landmarks",
        json={"model_id": "movinet_isl_final_best", "frames": synthetic_frames(4)},
    )
    assert response.status_code == 503
    assert "git lfs pull" in response.json()["detail"]


def test_invalid_base64_returns_422(client):
    response = client.post(
        "/api/predict/image",
        json={"model_id": "best_resnet18_isl", "image_base64": "!!!not base64!!!"},
    )
    assert response.status_code == 422


def test_undecodable_image_returns_422(client):
    import base64

    response = client.post(
        "/api/predict/image",
        json={
            "model_id": "best_resnet18_isl",
            "image_base64": base64.b64encode(b"definitely not an image").decode("ascii"),
        },
    )
    assert response.status_code == 422
    assert "could not decode image" in response.json()["detail"]


def test_wrong_landmark_count_is_rejected(client):
    frames = [{"hands": [{"landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0}] * 10}]}]
    response = client.post("/api/predict/landmarks", json={"model_id": "best_model_new", "frames": frames})
    assert response.status_code == 422


def test_modality_mismatch_points_at_the_right_route(client):
    response = client.post(
        "/api/predict/image",
        json={"model_id": "sign_language_model_keras", "image_base64": png_base64()},
    )
    assert response.status_code == 409
    assert "/api/predict/landmarks" in response.json()["detail"]


def test_empty_frame_list_is_a_validation_error(client):
    response = client.post("/api/predict/landmarks", json={"model_id": "best_model_new", "frames": []})
    assert response.status_code == 422


def test_unsupported_dataset_export_format(client):
    assert client.get("/api/dataset/export", params={"format": "parquet"}).status_code == 400
