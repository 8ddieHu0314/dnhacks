from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import settings
from .models import (
    FrameMetadata,
    IngestAcknowledgement,
    SessionCreated,
    SessionMetrics,
    SessionRequest,
)
from .pipeline import VisionPipeline
from .segmentation import build_vision_engine
from .workflows import WorkflowRegistry


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


pipeline = VisionPipeline(
    build_vision_engine(
        backend=settings.vision_backend,
        base_url=settings.vlm_base_url,
        api_key=settings.vlm_api_key,
        model=settings.vlm_model,
        timeout_seconds=settings.vlm_timeout_seconds,
    ),
    queue_capacity=settings.vision_frame_queue_capacity,
    result_history=settings.vision_result_history,
)
workflow_directory = Path(settings.workflow_definitions_dir or Path(__file__).with_name("workflow_definitions"))
workflow_registry = WorkflowRegistry.from_directory(workflow_directory)
sessions: dict[str, str] = {}


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


@app.websocket("/v1/sessions/{session_id}/frames")
async def ingest_frames_websocket(websocket: WebSocket, session_id: str) -> None:
    """Accept alternating metadata JSON and decoded-image binary messages.

    WebRTC/H.264 decoding belongs in a transport adapter before this endpoint; the
    vision service only consumes timestamped JPEG, PNG, or WebP frames.
    """

    if session_id not in sessions:
        await websocket.close(code=4404, reason="Unknown session")
        return
    await websocket.accept()
    try:
        while True:
            metadata = await receive_metadata(websocket)
            if metadata is None:
                continue
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            image_bytes = message.get("bytes")
            if image_bytes is None:
                await websocket.send_json({"error": "Send binary frame after metadata"})
                continue
            try:
                validate_frame_bytes(image_bytes)
                dropped = await pipeline.submit(session_id, metadata, image_bytes)
                acknowledgement = IngestAcknowledgement(
                    session_id=session_id,
                    frame_id=metadata.frame_id,
                    dropped_stale_frames=dropped,
                )
                await websocket.send_text(acknowledgement.model_dump_json())
            except HTTPException as exc:
                await websocket.send_json({"error": exc.detail})
    except WebSocketDisconnect:
        return


@app.get("/v1/sessions/{session_id}/results")
async def get_results(session_id: str) -> JSONResponse:
    validate_session(session_id)
    return JSONResponse([result.model_dump(mode="json") for result in pipeline.results_for(session_id)])


@app.get("/v1/sessions/{session_id}/metrics", response_model=SessionMetrics)
async def get_metrics(session_id: str) -> SessionMetrics:
    validate_session(session_id)
    return pipeline.metrics_for(session_id)
