# ---------------------------------------------------------------------------
# MODULE: WebSocket live-stream endpoint
#
# The live camera path. The browser keeps MediaPipe running locally (landmark
# detection is fast and private), sends only the 21x3 landmark vectors of each
# frame, and the backend keeps the temporal window and runs the trained model.
#
# Protocol (JSON text frames both ways):
#   client -> server
#     {"type":"frame","hands":[{"landmarks":[{x,y,z} x21],"handedness":"Right"}]}
#     {"type":"configure","model_id":"…","sequence_length":30,"stride":2,"hand":"first"}
#     {"type":"reset"}   drop the window (e.g. after pausing)
#     {"type":"ping"}
#   server -> client
#     {"type":"ready","model_id":"…","sequence_length":30,"stride":1}
#     {"type":"prediction","frame":123,"model_id":"…","top":{…},"predictions":[…],
#      "latency_ms":8.4,"dropped":0,"frames_in_window":30}
#     {"type":"error","message":"…","fatal":false}
#     {"type":"pong"} / {"type":"reset"}
#
# Backpressure: if the model is still busy with a previous window the frame is
# dropped and counted (`dropped`), instead of queueing up and drifting behind
# the camera.
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
from collections import deque
from typing import Any, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from ..inference import InferenceError
from ..registry import ModelNotFound
from ..runtimes.base import ModelLoadError, RuntimeUnavailable
from ..schemas import Frame, Hand, Landmark, LandmarksRequest, StreamConfig

router = APIRouter(tags=["stream"])

MAX_MESSAGE_BYTES = 512 * 1024


@router.websocket("/ws/stream")
async def stream(
    websocket: WebSocket,
    model_id: Optional[str] = Query(default=None),
    sequence_length: Optional[int] = Query(default=None, ge=2, le=300),
    stride: int = Query(default=1, ge=1, le=30),
    hand: str = Query(default="first"),
) -> None:
    await websocket.accept()
    registry = websocket.app.state.registry
    service = websocket.app.state.inference
    settings = websocket.app.state.settings

    config = StreamConfig(
        model_id=model_id,
        sequence_length=sequence_length or settings.default_sequence_length,
        stride=stride,
        hand=hand if hand in ("first", "left", "right", "dominant") else "first",
    )
    window: deque[Frame] = deque(maxlen=config.sequence_length)
    frame_index = 0
    dropped = 0
    busy = False

    async def send(payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload))

    async def apply_config() -> bool:
        """Resolve the model up front so failures surface immediately."""
        nonlocal window
        try:
            handle = await run_in_threadpool(
                registry.handle_for, config.model_id or registry.default_model_id() or ""
            )
        except (ModelNotFound, ModelLoadError, RuntimeUnavailable, InferenceError) as exc:
            await send({"type": "error", "message": str(exc), "fatal": True})
            return False
        # deque.maxlen is read-only, so a new length means a new deque.
        target_length = config.sequence_length or handle.spec.sequence_length or settings.default_sequence_length
        if window.maxlen != target_length:
            window = deque(window, maxlen=target_length)
        await send(
            {
                "type": "ready",
                "model_id": handle.spec.id,
                "modality": handle.spec.modality,
                "sequence_length": window.maxlen,
                "stride": config.stride,
                "hand": config.hand,
            }
        )
        return True

    if not await apply_config():
        # 1000, not 1011: an abnormal close makes WebSocket clients discard the
        # buffered error frame, so the browser would reconnect forever without
        # ever learning why the model could not be loaded.
        await websocket.close(code=1000)
        return

    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > MAX_MESSAGE_BYTES:
                await send({"type": "error", "message": "message too large", "fatal": False})
                continue

            try:
                message = json.loads(raw)
            except json.JSONDecodeError as exc:
                await send({"type": "error", "message": f"invalid JSON: {exc}", "fatal": False})
                continue

            kind = message.get("type", "frame")

            if kind == "ping":
                await send({"type": "pong"})
                continue

            if kind == "reset":
                window.clear()
                frame_index = 0
                dropped = 0
                await send({"type": "reset"})
                continue

            if kind == "configure":
                try:
                    config = StreamConfig(**{**config.model_dump(), **{k: v for k, v in message.items() if k != "type"}})
                except ValidationError as exc:
                    await send({"type": "error", "message": f"invalid configuration: {exc.errors()}", "fatal": False})
                    continue
                if not await apply_config():
                    await websocket.close(code=1000)
                    return
                continue

            # Default: a camera frame.
            try:
                hands = [
                    Hand(
                        landmarks=[Landmark(**point) for point in item.get("landmarks", [])],
                        handedness=item.get("handedness", "Right"),
                        score=float(item.get("score", 1.0)),
                    )
                    for item in message.get("hands", [])
                ]
            except (ValidationError, TypeError, ValueError) as exc:
                await send({"type": "error", "message": f"malformed frame: {exc}", "fatal": False})
                continue

            frame_index += 1
            window.append(Frame(hands=hands))

            if frame_index % max(1, config.stride) != 0:
                continue
            if busy:
                dropped += 1
                continue
            if not window:
                continue

            busy = True
            try:
                response = await run_in_threadpool(
                    service.predict_landmarks,
                    LandmarksRequest(
                        model_id=config.model_id,
                        frames=list(window),
                        top_k=message.get("top_k", 5),
                        hand=config.hand,
                    ),
                )
            except (InferenceError, ModelLoadError, RuntimeUnavailable) as exc:
                await send({"type": "error", "message": str(exc), "fatal": False})
            except ValidationError as exc:
                await send({"type": "error", "message": f"invalid request: {exc.errors()}", "fatal": False})
            except Exception as exc:  # noqa: BLE001 - never kill the socket on one bad window
                await send({"type": "error", "message": f"{type(exc).__name__}: {exc}", "fatal": False})
            else:
                await send(
                    {
                        "type": "prediction",
                        "frame": frame_index,
                        "model_id": response.model_id,
                        "top": response.top.model_dump() if response.top else None,
                        "predictions": [p.model_dump() for p in response.predictions],
                        "latency_ms": response.latency_ms,
                        "frames_in_window": len(window),
                        "dropped": dropped,
                        "warnings": response.warnings,
                    }
                )
                dropped = 0
            finally:
                busy = False

    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - close cleanly on any transport failure
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            return
