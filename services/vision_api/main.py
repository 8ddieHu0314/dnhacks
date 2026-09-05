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
