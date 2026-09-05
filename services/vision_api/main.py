from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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


@app.post("/v1/sessions/{session_id}/frames", response_model=IngestAcknowledgement)
async def ingest_frame(session_id: str, request: Request) -> IngestAcknowledgement:
    """HTTP fallback for a native device bridge posting one JPEG/PNG/WebP frame."""

    validate_session(session_id)
    try:
        metadata = FrameMetadata.model_validate_json(request.headers["x-frame-metadata"])
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail="x-frame-metadata header containing FrameMetadata JSON is required",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid frame metadata: {exc}") from exc
    image_bytes = await request.body()
    validate_frame_bytes(image_bytes)
    dropped = await pipeline.submit(session_id, metadata, image_bytes)
    return IngestAcknowledgement(
        session_id=session_id, frame_id=metadata.frame_id, dropped_stale_frames=dropped
    )


async def receive_metadata(websocket: WebSocket) -> FrameMetadata | None:
    """Receive one valid metadata text message, reporting protocol errors inline."""

    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect()
    text = message.get("text")
    if text is None:
        await websocket.send_json({"error": "Send FrameMetadata before binary frame"})
        return None
    try:
        return FrameMetadata.model_validate_json(text)
    except ValueError as exc:
        await websocket.send_json({"error": f"Invalid frame metadata: {exc}"})
        return None
