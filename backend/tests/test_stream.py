# ---------------------------------------------------------------------------
# WebSocket live stream
# ---------------------------------------------------------------------------
from __future__ import annotations

import json

from conftest import synthetic_hand


def _read_until(websocket, message_type: str, limit: int = 10) -> dict:
    for _ in range(limit):
        payload = json.loads(websocket.receive_text())
        if payload.get("type") == message_type:
            return payload
    raise AssertionError(f"never received a '{message_type}' message")


def _read_prediction_at(websocket, frame: int, limit: int = 20) -> dict:
    """Stride is 1 in these tests, so every frame yields a prediction."""
    for _ in range(limit):
        payload = json.loads(websocket.receive_text())
        if payload.get("type") == "prediction" and payload.get("frame") == frame:
            return payload
    raise AssertionError(f"never received a prediction for frame {frame}")


def test_stream_emits_predictions(client):
    with client.websocket_connect("/ws/stream?model_id=best_model_new&sequence_length=6&stride=1") as socket:
        ready = _read_until(socket, "ready")
        assert ready["model_id"] == "best_model_new"
        assert ready["sequence_length"] == 6

        for index in range(6):
            socket.send_text(json.dumps({"type": "frame", "hands": [synthetic_hand(seed=index)]}))

        prediction = _read_prediction_at(socket, 6)
        assert prediction["model_id"] == "best_model_new"
        assert prediction["top"]["label"]
        assert prediction["frames_in_window"] == 6
        assert prediction["latency_ms"] >= 0


def test_ping_and_reset(client):
    with client.websocket_connect("/ws/stream?model_id=best_model_new&sequence_length=4") as socket:
        _read_until(socket, "ready")
        socket.send_text(json.dumps({"type": "ping"}))
        assert _read_until(socket, "pong")["type"] == "pong"

        socket.send_text(json.dumps({"type": "frame", "hands": [synthetic_hand(seed=1)]}))
        socket.send_text(json.dumps({"type": "reset"}))
        assert _read_until(socket, "reset")["type"] == "reset"

        # After a reset the window is empty again: the next frame is number 1.
        for index in range(2):
            socket.send_text(json.dumps({"type": "frame", "hands": [synthetic_hand(seed=index)]}))
        assert _read_prediction_at(socket, 2)["frame"] == 2


def test_invalid_frame_is_reported_without_closing_the_socket(client):
    with client.websocket_connect("/ws/stream?model_id=best_model_new&sequence_length=4") as socket:
        _read_until(socket, "ready")
        socket.send_text("this is not json")
        error = _read_until(socket, "error")
        assert error["fatal"] is False
        assert "invalid JSON" in error["message"]

        socket.send_text(json.dumps({"type": "frame", "hands": [{"landmarks": [{"x": 0.1, "y": 0.2}]}]}))
        assert _read_until(socket, "error")["type"] == "error"


def test_unknown_model_closes_the_stream(client):
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect("/ws/stream?model_id=does_not_exist") as socket:
        payload = json.loads(socket.receive_text())
        assert payload["type"] == "error"
        assert payload["fatal"] is True
        try:
            socket.receive_text()
            raise AssertionError("socket should have been closed")
        except WebSocketDisconnect:
            pass


def test_configure_switches_model_mid_stream(client):
    with client.websocket_connect("/ws/stream?model_id=best_model_new&sequence_length=6") as socket:
        assert _read_until(socket, "ready")["model_id"] == "best_model_new"

        socket.send_text(
            json.dumps({"type": "configure", "model_id": "sign_language_model_keras", "sequence_length": 12})
        )
        ready = _read_until(socket, "ready")
        assert ready["model_id"] == "sign_language_model_keras"
        assert ready["sequence_length"] == 12

        for index in range(12):
            socket.send_text(json.dumps({"type": "frame", "hands": [synthetic_hand(seed=index)]}))
        prediction = _read_prediction_at(socket, 12)
        assert prediction["model_id"] == "sign_language_model_keras"
