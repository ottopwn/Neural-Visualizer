#!/usr/bin/env bash
# Start the Neural Forge backend (FastAPI + PyTorch, :8000) and frontend (Vite, :5173).
#   ./start.sh            start both (Ctrl+C stops both)
#   ./start.sh --setup    install Python and Node dependencies first (CPU-only PyTorch)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if [[ "${1:-}" == "--setup" ]]; then
  echo "Installing backend dependencies (CPU-only PyTorch)…"
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip install -r backend/requirements-dev.txt
  echo "Installing frontend dependencies…"
  (cd frontend && npm ci)
fi

if ! "$PY" -c "import torch, fastapi" 2>/dev/null; then
  echo "PyTorch/FastAPI not found for $PY. Run: ./start.sh --setup  (or see README)" >&2
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  echo "frontend/node_modules missing. Run: ./start.sh --setup" >&2
  exit 1
fi

echo "Starting FastAPI backend on http://127.0.0.1:8000 …"
(cd backend && "$PY" -m uvicorn main:app --host 127.0.0.1 --port 8000) &
BACKEND_PID=$!

echo "Starting Vite dev server on http://127.0.0.1:5173 …"
(cd frontend && VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://127.0.0.1:8000}" npx vite --host 127.0.0.1 --port 5173) &
FRONTEND_PID=$!

trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT TERM
echo ""
echo "  Open http://127.0.0.1:5173   (Ctrl+C stops both servers)"
wait
