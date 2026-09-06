from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from .component_knowledge import ComponentKnowledgeBase
from .config import settings
from .models import (
    AdvisoryFieldSignal,
    AdvisoryFieldSignalAcknowledgement,
    FrameMetadata,
    IngestAcknowledgement,
    RetrievedComponent,
    SessionCreated,
    SessionMetrics,
    SessionRequest,
    WorkflowDefinition,
)
from .pipeline import VisionPipeline
from .segmentation import build_vision_engine
from .signals import AdvisorySignalStore
from .workflows import WorkflowRegistry


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


component_knowledge_path = Path(settings.component_knowledge_path) if settings.component_knowledge_path else (
    Path(__file__).resolve().parents[2] / "docs/components/components.json"
)
component_knowledge = ComponentKnowledgeBase.from_path(component_knowledge_path)
pipeline = VisionPipeline(
    build_vision_engine(
        backend=settings.vision_backend,
        base_url=settings.vlm_base_url,
        api_key=settings.vlm_api_key,
        model=settings.vlm_model,
        timeout_seconds=settings.vlm_timeout_seconds,
        component_knowledge=component_knowledge,
        component_knowledge_top_k=settings.component_knowledge_top_k,
    ),
    queue_capacity=settings.vision_frame_queue_capacity,
    result_history=settings.vision_result_history,
)
workflow_directory = Path(settings.workflow_definitions_dir or Path(__file__).with_name("workflow_definitions"))
workflow_registry = WorkflowRegistry.from_directory(workflow_directory)
sessions: dict[str, str] = {}
signals = AdvisorySignalStore()


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


def workflow_for_session(session_id: str) -> WorkflowDefinition:
    try:
        return workflow_registry.get(sessions[session_id])
    except (KeyError, LookupError) as exc:
        raise HTTPException(status_code=404, detail="Unknown session")


def validate_session(session_id: str) -> None:
    workflow_for_session(session_id)


def validate_frame_bytes(image_bytes: bytes, encoding: str) -> None:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Frame body is empty")
    if len(image_bytes) > settings.vision_max_frame_bytes:
        raise HTTPException(status_code=413, detail="Frame exceeds VISION_MAX_FRAME_BYTES")
    detected_encoding = (
        "jpeg" if image_bytes.startswith(b"\xff\xd8") else
        "png" if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") else
        "webp" if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP" else None
    )
    if detected_encoding is None:
        raise HTTPException(status_code=415, detail="Frame is not a JPEG, PNG, or WebP image")
    if detected_encoding != encoding:
        raise HTTPException(status_code=422, detail="Frame encoding does not match FrameMetadata")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "segmentation_backend": pipeline.backend_name}


@app.get("/webcam", include_in_schema=False)
async def webcam_test_page() -> FileResponse:
    return FileResponse(Path(__file__).with_name("static") / "webcam.html")


@app.get("/v1/workflows", response_model=list[WorkflowDefinition])
async def list_workflows() -> list[WorkflowDefinition]:
    return workflow_registry.list()


@app.get("/v1/components/search", response_model=list[RetrievedComponent])
async def search_components(q: str, limit: int = 3) -> list[RetrievedComponent]:
    """Return deterministic knowledge-base candidates for a spoken or typed phrase."""

    return component_knowledge.summaries(component_knowledge.search(q, limit=min(max(limit, 1), 5)))


@app.get("/v1/components/{component_id}")
async def get_component(component_id: str) -> JSONResponse:
    record = component_knowledge.record_for(component_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown component")
    return JSONResponse(record)


@app.post("/v1/sessions", response_model=SessionCreated, status_code=201)
async def create_session(request: SessionRequest | None = None) -> SessionCreated:
    try:
        workflow = workflow_registry.get((request or SessionRequest()).workflow_id)
    except LookupError as exc:
        raise HTTPException(status_code=422, detail="Unknown workflow_id") from exc
    session_id = str(uuid4())
    sessions[session_id] = workflow.id
    return SessionCreated(
        session_id=session_id,
        created_at=utc_now(),
        workflow_id=workflow.id,
        workflow_version=workflow.version,
    )


@app.post(
    "/v1/sessions/{session_id}/advisory-field-signals",
    response_model=AdvisoryFieldSignalAcknowledgement,
)
async def ingest_advisory_field_signal(
    session_id: str, signal: AdvisoryFieldSignal
) -> AdvisoryFieldSignalAcknowledgement:
    """Record a non-contact signal that may warn but cannot clear electrical work."""

    validate_session(session_id)
    accepted = signals.record_field_signal(session_id, signal)
    return AdvisoryFieldSignalAcknowledgement(session_id=session_id, state=accepted.state)


@app.post("/v1/sessions/{session_id}/frames", response_model=IngestAcknowledgement)
async def ingest_frame(session_id: str, request: Request) -> IngestAcknowledgement:
    """HTTP fallback for a native device bridge posting one JPEG/PNG/WebP frame."""

    workflow = workflow_for_session(session_id)
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
    validate_frame_bytes(image_bytes, metadata.encoding)
    dropped = await pipeline.submit(session_id, metadata, image_bytes, workflow)
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
    workflow = workflow_for_session(session_id)
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
                validate_frame_bytes(image_bytes, metadata.encoding)
                dropped = await pipeline.submit(session_id, metadata, image_bytes, workflow)
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


@app.get("/v1/sessions/{session_id}/results/latest")
async def get_latest_result(session_id: str) -> Response:
    validate_session(session_id)
    results = pipeline.results_for(session_id)
    return JSONResponse(results[-1].model_dump(mode="json")) if results else Response(status_code=204)


@app.get("/v1/sessions/{session_id}/metrics", response_model=SessionMetrics)
async def get_metrics(session_id: str) -> SessionMetrics:
    validate_session(session_id)
    return pipeline.metrics_for(session_id)
