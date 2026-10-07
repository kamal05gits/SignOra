# SignOra — Real-Time Indian Sign Language Translator

A webcam-first ISL translator: MediaPipe tracks the hand in the browser, a classifier turns the
landmarks into signs, and the app builds a sentence that can be spoken and translated. Two
recognition engines are wired into the same pipeline:

- **Backend** — the trained models in [`models/`](models), served by the FastAPI service in
  [`backend/`](backend) (PyTorch `.pth`, Keras `.keras`/`.h5`).
- **On-device** — the transparent geometric classifier in `src/lib/signClassifier.ts`, which needs
  no server at all.

## Run it

```bash
git lfs install && git lfs pull      # the model weights are LFS objects (~190 MB)
./scripts/dev.sh                     # backend :8000 + web :5173, proxying /api and /ws
```

Then open <http://localhost:5173> and press **Start Camera**. API docs live at
<http://localhost:8000/docs>.

Docker instead:

```bash
docker compose up --build
```

If the backend is not running the app still works — set *Settings → Recognition engine* to
**On-device**, or leave it on **Auto** and it falls back by itself.

## Deploying to Vercel

SignOra runs on Vercel as **two projects** connected to this same repository: a static
frontend and the FastAPI inference backend.

1. **Frontend project** — Root Directory `.`, **Vite** preset, install `npm ci`, build
   `npm run build`, output directory `dist`. Optionally set the env var `VITE_BACKEND_URL`
   to the API project URL so every visitor defaults to backend recognition (per-browser
   overrides in Settings still win).
2. **API project** (e.g. `signora-api`) — Root Directory `.` (not `backend/` — the weights
   live in the root-level `models/` folder), **FastAPI** preset. The entrypoint is declared
   in [`pyproject.toml`](pyproject.toml) and [`vercel.json`](vercel.json) bundles
   `models/**` into the function.
   - **Settings → Git**: enable **Git LFS** (the weights are LFS objects), then redeploy.
   - **Settings → Functions**: enable **Fluid Compute** (required for the `/ws/stream`
     WebSocket).
   - Environment variables:
     ```text
     SIGNORA_DATA_DIR=/tmp/signora-data
     SIGNORA_PRELOAD_MODELS=false
     SIGNORA_MAX_LOADED_MODELS=1
     SIGNORA_CORS_ORIGINS=["https://<frontend>.vercel.app"]
     ```
     `/tmp` is the only writable path in a Vercel function, so backend-synced dataset
     samples are not durable across redeploys — use external storage if you need them.
3. Smoke-test the API URL: `GET /health` and `GET /api/models`.

**Size limits**: the ~190 MB of weights plus PyTorch and TensorFlow push the function past
Vercel's standard 500 MB limit — you may need large functions (`VERCEL_SUPPORT_LARGE_FUNCTIONS=1`),
and the Hobby plan caps function memory at 2 GB. The backend tolerates a missing runtime
(models report `runtime_missing` instead of crashing), so dropping `tensorflow-cpu` or
`torch` from `pyproject.toml` is a valid way to slim the function to the model family you
actually serve.

## How a frame becomes a word

```
webcam frame
   │  MediaPipe Hands (in the browser, no pixels leave the machine)
   ▼
21 landmarks × 3 floats
   │
   ├── on-device ──► geometric classifier  (src/lib/signClassifier.ts)
   │
   └── backend ────► WebSocket /ws/stream  (src/lib/backend.ts → backend/app)
                        │  server keeps the temporal window
                        ▼
                     trained model in models/
   │
   ▼
ClassificationResult { label, gloss, confidence, source }
   │  TemporalSmoother: majority vote + cooldown   (src/lib/sequenceBuffer.ts)
   ▼
token → sentence → speech + translation
```

Both engines emit the same `ClassificationResult`, so smoothing, token building, speech and
translation are shared and cannot drift between them. Which engine is active is always visible —
in the sidebar and on the camera overlay.

## Repository layout

```
models/                       trained weights (Git LFS) + model_manifest.json
backend/
  app/main.py                 FastAPI app factory, CORS, error handlers
  app/registry.py             discovery, lazy loading, LRU, timings
  app/inference.py            request → tensor → labelled predictions
  app/runtimes/               torch + keras loaders, preprocessing, labels
  app/routers/                health, models, predict, dataset, ws stream
  scripts/inspect_model.py    dump a checkpoint and print its manifest entry
  tests/                      51 tests against real torch/TensorFlow
src/
  hooks/useRecognitionEngine  the orchestration layer
  lib/backend.ts              typed API client + SignStream WebSocket
  pages/ModelPage.tsx         model registry, image test, backend evaluation
scripts/dev.sh                one-command dev workflow
```

## Verifying the backend can serve your models

```bash
npm run inspect:models      # tensor shapes + inferred architecture per file
npm run test:backend        # pytest: loaders, stream, dataset, error contract
```

`GET /api/models` reports the honest status of every file: `ready`, `idle`, `missing_weights` (LFS
pointer not pulled), `needs_manifest` (architecture not inferable — the `hint` field contains the
tensor shapes and a paste-ready manifest block), `load_error`, or `runtime_missing`.

See [backend/README.md](backend/README.md) for the endpoint reference, manifest keys, label
resolution order, environment variables and the WebSocket protocol.

## Notes

- Nothing but landmark coordinates is ever sent to the backend; no video is uploaded or recorded.
- Dataset samples captured in the browser include the raw landmark frame, so they can be exported
  (`GET /api/dataset/export?format=npz`) as a ready-to-train `(N, T, 63)` tensor set and scored
  against a trained model with `POST /api/dataset/evaluate`.
- Class names are not stored inside a PyTorch checkpoint. Drop a `models/labels.json` (template in
  `models/labels.example.json`) so predictions come back as words instead of `class_0`.
