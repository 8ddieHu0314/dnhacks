#!/usr/bin/env bash
# One-command setup for a fresh clone. Creates .venv with Python 3.11, installs
# pinned dependencies, downloads the SH17 weights, and runs the smoke test.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null || { echo "need python3.11 on PATH (or set PYTHON=/path/to/python3.11)"; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements-lock.txt

mkdir -p weights
[ -f weights/yolo8s.pt ] || curl -L -o weights/yolo8s.pt https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8s.pt
[ -f weights/yolo8m.pt ] || curl -L -o weights/yolo8m.pt https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8m.pt

[ -f .env ] || { echo "ROBOFLOW_API_KEY=" > .env; chmod 600 .env; echo "wrote empty .env, add your ROBOFLOW_API_KEY"; }

.venv/bin/python smoke.py test_images/bare_hands_wrench_1.jpg
echo "setup ok"
