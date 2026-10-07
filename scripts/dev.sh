#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start the whole workflow with one command:
#
#   ./scripts/dev.sh
#
# 1. creates backend/.venv and installs backend/requirements.txt on first run
# 2. starts the FastAPI inference backend on :8000
# 3. installs npm dependencies on first run
# 4. starts the vite dev server on :5173, proxying /api, /health and /ws to :8000
#
# Environment overrides:
#   BACKEND_PORT / FRONTEND_PORT / SIGNORA_BACKEND_URL / SKIP_INSTALL=1
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
VENV="$ROOT/backend/.venv"
STAMP="$VENV/.requirements-installed"

log() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }

# --- 1. Python environment -------------------------------------------------
if [ ! -d "$VENV" ]; then
  log "creating virtualenv in backend/.venv"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ "${SKIP_INSTALL:-0}" != "1" ] && [ ! -f "$STAMP" ]; then
  log "installing backend dependencies (first run only)"
  pip install --quiet --upgrade pip
  pip install --quiet -r backend/requirements.txt
  touch "$STAMP"
fi

# --- 2. Git LFS weights ----------------------------------------------------
if ! git lfs version >/dev/null 2>&1; then
  log "WARNING: git-lfs is not installed — model weights may be pointer files only"
elif find models -maxdepth 1 -type f -size -1k \( -name '*.pth' -o -name '*.keras' -o -name '*.h5' \) | grep -q .; then
  log "WARNING: some weights in models/ look like Git LFS pointers — run: git lfs pull"
fi

# --- 3. Backend ------------------------------------------------------------
log "starting inference backend on http://0.0.0.0:${BACKEND_PORT} (docs at /docs)"
SIGNORA_PORT="$BACKEND_PORT" uvicorn app.main:app \
  --app-dir backend \
  --host 0.0.0.0 \
  --port "$BACKEND_PORT" \
  --reload \
  --reload-dir backend/app &
BACKEND_PID=$!

cleanup() {
  log "stopping backend (pid $BACKEND_PID)"
  kill "$BACKEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 4. Frontend -----------------------------------------------------------
if [ "${SKIP_INSTALL:-0}" != "1" ] && [ ! -d node_modules ]; then
  log "installing npm dependencies (first run only)"
  npm install
fi

log "starting vite dev server on http://0.0.0.0:${FRONTEND_PORT}"
SIGNORA_BACKEND_URL="${SIGNORA_BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}" \
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
