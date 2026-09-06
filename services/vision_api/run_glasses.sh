#!/bin/sh
set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$REPO_DIR"

: "${VLM_API_KEY:?Export VLM_API_KEY in this terminal before starting the glasses server}"
export VISION_BACKEND="${VISION_BACKEND:-anthropic_catalog_identification}"
export VLM_MODEL="${VLM_MODEL:-claude-haiku-4-5-20251001}"
export SPEECH_MODE="${SPEECH_MODE:-glasses}"
export GLASSES_FRAME_INTERVAL_SECONDS="${GLASSES_FRAME_INTERVAL_SECONDS:-1.0}"

exec .venv/bin/uvicorn vision_api.main:app --app-dir services \
  --host "${VISION_API_HOST:-0.0.0.0}" --port "${VISION_API_PORT:-8000}"
