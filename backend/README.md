# SignOra Inference Backend

A FastAPI service that loads the trained models in [`models/`](../models) and serves them to the
web app — over REST for one-shot calls and over a WebSocket for the live camera.

```
models/                         <- your trained weights (Git LFS)
   │
   ▼
backend/app                     <- registry + torch/keras runtimes + API
   │  REST  /api/predict/{landmarks|image|video|features}
   │  WS    /ws/stream
   ▼
src/lib/backend.ts              <- typed client + SignStream
   ▼
src/hooks/useRecognitionEngine  <- landmarks -> model -> smoothing -> sentence
```

## Quick start

```bash
./scripts/dev.sh          # backend on :8000, vite on :5173, proxying /api and /ws
```

Or by hand:

```bash
python3 -m venv backend/.venv && source backend/.venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload

npm install && npm run dev
```

Interactive API docs: <http://127.0.0.1:8000/docs> · health: <http://127.0.0.1:8000/health>

### Weights are Git LFS objects

The four checkpoints are stored with Git LFS. A plain clone leaves ~130-byte pointer files, and
loading one fails with a cryptic `PytorchStreamReader` error. The backend detects pointers up front
and reports `missing_weights` with the fix:

```bash
git lfs install
git lfs pull
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | runtime versions, model counts, every problem found |
| GET | `/api/config` | effective configuration (model dir, defaults) |
| GET | `/api/models` | every model file with status, shapes, labels, timings |
| GET | `/api/models/{id}` | one model |
| POST | `/api/models/refresh` | re-scan `models/` after adding files |
| POST | `/api/models/{id}/load` | warm a model into memory |
| POST | `/api/models/{id}/unload` | free its memory |
| POST | `/api/predict/landmarks` | `(1, T, 63)` landmark sequence → top-k |
| POST | `/api/predict/image` | base64 still frame → top-k |
| POST | `/api/predict/image/upload` | same, as multipart (handy with curl) |
| POST | `/api/predict/video` | ordered base64 frames → top-k |
| POST | `/api/predict/features` | engineered `HandFeatureSet` vector → top-k |
| POST | `/api/dataset/samples` | store samples synced from the browser |
| GET | `/api/dataset/summary` | counts per class |
| GET | `/api/dataset/export?format=npz` | training set as `npz` / `csv` / `jsonl` |
| POST | `/api/dataset/evaluate` | score recorded samples with a trained model |
| WS | `/ws/stream` | live landmark stream (see below) |

### Error contract

The status code tells the frontend what to do, so nothing is guessed:

| Code | Meaning | Frontend behaviour |
| --- | --- | --- |
| 404 | unknown model id (`available_models` is returned) | show the list |
| 409 | wrong endpoint for that model's input type | point at the right route |
| 422 | malformed payload / undecodable image | show the message |
| 503 | model cannot be loaded (LFS pointer, missing runtime, bad architecture) | fall back on-device |

## Live stream protocol

The browser keeps MediaPipe locally and sends only landmark vectors — no pixels.

```jsonc
// client -> server
{"type":"frame","hands":[{"landmarks":[{"x":0.4,"y":0.5,"z":0.0}, …21],"handedness":"Right","score":0.97}]}
{"type":"configure","model_id":"best_model_new","sequence_length":30,"stride":2,"hand":"first"}
{"type":"reset"}      // drop the window (after pausing)
{"type":"ping"}

// server -> client
{"type":"ready","model_id":"best_model_new","modality":"landmarks","sequence_length":30,"stride":1}
{"type":"prediction","frame":123,"model_id":"…","top":{"index":3,"label":"HELLO","score":0.81},
 "predictions":[…],"latency_ms":7.4,"frames_in_window":30,"dropped":0}
{"type":"error","message":"…","fatal":false}
```

If the model is still working on the previous window the new frame is dropped and counted in
`dropped` — the stream never queues up behind the camera.

## Configuring a model: `models/model_manifest.json`

Keras files (`.keras`, `.h5`) store their own graph, so they need nothing. A PyTorch `.pth` stores
weights only, so the architecture is either inferred from the tensor shapes (ResNet family,
sequential MLP, LSTM, GRU) or declared in the manifest. When inference is inconclusive,
`/api/models` returns `needs_manifest` and its `hint` field contains the actual tensor shapes plus a
ready-to-paste block.

```bash
python backend/scripts/inspect_model.py models/best_model_new.pth
```

```jsonc
{
  "defaults": { "normalization": "wrist", "image_size": 224, "sequence_length": 30 },
  "models": {
    "best_model_new.pth": {
      "architecture": "cnn1d_lstm",   // mlp | lstm | gru | cnn1d_lstm | resnet50 | torchscript | …
      "modality": "landmarks",        // image | video | landmarks | features
      "sequence_length": 30,
      "input_dim": 63,                // 63 = x,y,z of 21 landmarks; 42 = x,y only
      "channels": [64, 128],
      "lstm_hidden": 64,
      "num_classes": 26,
      "normalization": "wrist",       // wrist | bbox | none
      "labels_file": "labels.json"
    },
    "movinet_isl_final_best.keras": {
      "modality": "video",
      "video_frames": 8,
      "image_size": 224,
      // MoViNet's layers live in a separate package; importing it registers them
      "preimport": ["official.projects.movinet.modeling.movinet"]
    }
  }
}
```

Preprocessing keys: `image_size`, `channel_order` (`RGB`/`BGR`), `data_format`
(`channels_last`/`channels_first` — torchvision is NCHW, Keras is NHWC), `scale`, `mean`, `std`,
`normalization`, `include_handedness`. After editing, hit `POST /api/models/refresh`.

### Labels

Class names cannot be read out of a PyTorch checkpoint. Resolution order:

1. `labels` inline in the manifest
2. `labels_file` from the manifest (relative to `models/`)
3. a sidecar next to the weights — `<stem>.labels.json` or `.txt`
4. `models/labels.json` or `models/labels.txt`, shared by every model
5. nothing → predictions are `class_0`, `class_1`, … and the API says so in `hint`

`models/labels.example.json` shows both accepted shapes (a flat list, or `{label, gloss}` objects
where `gloss` is the word the app speaks).

## Environment variables

All optional; also read from `.env` (see `.env.example`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIGNORA_MODEL_DIR` | `<repo>/models` | where the weights live |
| `SIGNORA_DATA_DIR` | `<repo>/backend/data` | dataset samples and exports |
| `SIGNORA_HOST` / `SIGNORA_PORT` | `0.0.0.0` / `8000` | bind address |
| `SIGNORA_CORS_ORIGINS` | `["*"]` | JSON list of allowed origins |
| `SIGNORA_PRELOAD_MODELS` | `false` | load every model at startup |
| `SIGNORA_MAX_LOADED_MODELS` | `3` | LRU cap on resident models |
| `SIGNORA_DEFAULT_SEQUENCE_LENGTH` | `30` | frames a temporal model expects |
| `SIGNORA_DEFAULT_IMAGE_SIZE` | `224` | resize edge when undeclared |
| `SIGNORA_DEFAULT_VIDEO_FRAMES` | `8` | frames sampled for video models |

## Tests

```bash
source backend/.venv/bin/activate
python -m pytest          # from backend/
```

The suite builds small-but-real checkpoints (a torchvision ResNet-18 `state_dict`, a sequential-MLP
`state_dict`, a Keras landmark model saved as `.keras`, a Keras image model saved as `.h5`, and a
fake LFS pointer), so the loaders, the modality inference, the WebSocket stream, the dataset export
and the error contract are all exercised against real torch and TensorFlow — no mocking of the ML
layer.

## Docker

```bash
docker compose up --build      # backend :8000 + web :5173
```

Mount your weights over `/app/models` if you would rather not bake 190 MB into the image.
