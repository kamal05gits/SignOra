# ---------------------------------------------------------------------------
# Health + model discovery
# ---------------------------------------------------------------------------
from __future__ import annotations


def test_health_reports_runtimes_and_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["models_total"] == 5
    assert "torch" in body["runtimes"] and "keras" in body["runtimes"]
    assert body["runtimes"]["torch"], "torch should be importable in the test environment"


def test_index_lists_endpoints(client):
    body = client.get("/").json()
    assert body["predict"]["landmarks"].endswith("/api/predict/landmarks")
    assert body["websocket"] == "/ws/stream"


def test_config_exposes_model_dir(client, env):
    body = client.get("/api/config").json()
    assert body["model_dir"] == str(env["model_dir"])
    assert body["manifest_exists"] is True


def test_models_are_discovered_with_metadata(client):
    models = {m["id"]: m for m in client.get("/api/models").json()}
    assert set(models) == {
        "best_resnet18_isl",
        "best_model_new",
        "sign_language_model_keras",
        "sign_language_model_h5",
        "movinet_isl_final_best",
    }
    resnet = models["best_resnet18_isl"]
    assert resnet["framework"] == "torch"
    assert resnet["modality"] == "image"
    assert resnet["architecture"] == "resnet18"
    assert resnet["status"] == "idle"
    assert resnet["loaded"] is False
    assert resnet["labels_source"] == "shared-json"


def test_lfs_pointer_is_reported_not_crashed(client):
    models = {m["id"]: m for m in client.get("/api/models").json()}
    pointer = models["movinet_isl_final_best"]
    assert pointer["lfs_pointer"] is True
    assert pointer["status"] == "missing_weights"
    assert "git lfs pull" in (pointer["detail"] or "")


def test_load_lfs_pointer_returns_status_instead_of_500(client):
    response = client.post("/api/models/movinet_isl_final_best/load")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "missing_weights"
    assert body["loaded"] is False


def test_unknown_model_is_404(client):
    response = client.get("/api/models/does_not_exist")
    assert response.status_code == 404
    assert "available_models" in response.json()


def test_refresh_rescans(client, env):
    extra = env["model_dir"] / "extra_model.pth"
    extra.write_bytes(b"not really a checkpoint")
    try:
        ids = {m["id"] for m in client.post("/api/models/refresh").json()}
        assert "extra_model" in ids
    finally:
        extra.unlink()


def test_unload_marks_model_unloaded(client):
    assert client.post("/api/models/best_model_new/load").json()["loaded"] is True
    assert client.get("/api/models/best_model_new").json()["status"] == "ready"
    assert client.post("/api/models/best_model_new/unload").json()["status"] == "unloaded"
    # Unloading twice is an error the caller can rely on.
    assert client.post("/api/models/best_model_new/unload").status_code == 404
