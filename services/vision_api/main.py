from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import settings
from .models import (
    FrameMetadata,
    IngestAcknowledgement,
    SessionCreated,
    SessionMetrics,
)
from .pipeline import VisionPipeline
from .segmentation import build_segmentation_engine


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


pipeline = VisionPipeline(
    build_segmentation_engine(settings.segmentation_backend),
    queue_capacity=settings.vision_frame_queue_capacity,
    result_history=settings.vision_result_history,
)
sessions: set[str] = set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await pipeline.start()
    try:
        yield
    finally:
        await pipeline.stop()


app = FastAPI(
    title="Ray-Ban Meta Vision Pipeline",
    version="0.1.0",
    description="Low-latency decoded-frame ingestion and segmentation scaffold.",
    lifespan=lifespan,
)


def validate_session(session_id: str) -> None:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session")


def validate_frame_bytes(image_bytes: bytes) -> None:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Frame body is empty")
    if len(image_bytes) > settings.vision_max_frame_bytes:
        raise HTTPException(status_code=413, detail="Frame exceeds VISION_MAX_FRAME_BYTES")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "segmentation_backend": pipeline.backend_name}


@app.post("/v1/sessions", response_model=SessionCreated, status_code=201)
async def create_session() -> SessionCreated:
    session_id = str(uuid4())
    sessions.add(session_id)
    return SessionCreated(session_id=session_id, created_at=utc_now())
