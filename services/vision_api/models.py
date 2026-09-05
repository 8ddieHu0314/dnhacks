from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


ImageEncoding = Literal["jpeg", "png", "webp"]


class WorkflowCheckpoint(BaseModel):
    """One observable stage in a workflow; it never grants operational authority."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    title: str = Field(min_length=1, max_length=160)
    evidence_prompt: str = Field(min_length=1, max_length=1_000)


class WorkflowDefinition(BaseModel):
    """Versioned, worker-agnostic instructions selected for a stream session."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    version: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=160)
    instructions: str = Field(min_length=1, max_length=4_000)
    checkpoints: list[WorkflowCheckpoint] = Field(default_factory=list, max_length=20)


class FrameMetadata(BaseModel):
    """Metadata sent before a binary frame on the WebSocket transport."""

    frame_id: str = Field(min_length=1, max_length=128)
    captured_at: datetime = Field(default_factory=utc_now)
    width: int = Field(gt=0, le=8_192)
    height: int = Field(gt=0, le=8_192)
    encoding: ImageEncoding = "jpeg"
    rotation_degrees: Literal[0, 90, 180, 270] = 0


@dataclass(frozen=True, slots=True)
class Frame:
    session_id: str
    metadata: FrameMetadata
    image_bytes: bytes
    received_at: datetime
    workflow: WorkflowDefinition | None = None


class BoundingBox(BaseModel):
    x: Annotated[float, Field(ge=0.0, le=1.0)]
    y: Annotated[float, Field(ge=0.0, le=1.0)]
    width: Annotated[float, Field(gt=0.0, le=1.0)]
    height: Annotated[float, Field(gt=0.0, le=1.0)]


class SegmentationRegion(BaseModel):
    label: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    bounding_box: BoundingBox
    # Normalized polygon points: [x0, y0, x1, y1, ...]. Keeping this compact
    # makes it friendly to a mobile client and is replaceable by an RLE mask later.
    polygon: list[float] = Field(min_length=6)


class ActionProposal(BaseModel):
    """A VLA recommendation; clients must confirm before any physical action."""

    action: str = Field(min_length=1, max_length=240)
    target: str | None = Field(default=None, max_length=240)
    rationale: str = Field(min_length=1, max_length=1_000)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    requires_confirmation: bool = True


class VisionAnalysis(BaseModel):
    """Grounded VLM observations and VLA-style action proposals for one frame."""

    summary: str = Field(min_length=1, max_length=2_000)
    observations: list[str] = Field(default_factory=list, max_length=20)
    safety_alerts: list[str] = Field(default_factory=list, max_length=10)
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=10)


class VisionOutput(BaseModel):
    """Unified output from a segmentation-only or multimodal vision engine."""

    regions: list[SegmentationRegion] = Field(default_factory=list)
    analysis: VisionAnalysis | None = None


class SegmentationResult(BaseModel):
    session_id: str
    frame_id: str
    backend: str
    completed_at: datetime = Field(default_factory=utc_now)
    latency_ms: float = Field(ge=0.0)
    regions: list[SegmentationRegion]
    analysis: VisionAnalysis | None = None


class SessionCreated(BaseModel):
    session_id: str
    created_at: datetime


class IngestAcknowledgement(BaseModel):
    session_id: str
    frame_id: str
    accepted: bool = True
    dropped_stale_frames: int = Field(ge=0)


class SessionMetrics(BaseModel):
    session_id: str
    received_frames: int = Field(ge=0)
    processed_frames: int = Field(ge=0)
    failed_frames: int = Field(ge=0)
    dropped_stale_frames: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
