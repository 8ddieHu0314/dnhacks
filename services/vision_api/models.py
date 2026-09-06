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
    user_request: str | None = Field(
        default=None,
        min_length=1,
        max_length=1_000,
        description="Optional spoken or typed question that belongs to this camera frame.",
    )


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


class RetrievedComponent(BaseModel):
    """A record selected by the server before the model is asked to reason about it."""

    component_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    canonical_name: str = Field(min_length=1, max_length=240)
    data_confidence: Literal["high", "medium", "low"]
    retrieval_score: float = Field(ge=0.0)


class ComponentIdentification(BaseModel):
    """A visual identification that must name one server-retrieved record."""

    component_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    observed_evidence: list[str] = Field(default_factory=list, max_length=8)
    uncertainty: str | None = Field(default=None, max_length=800)


class ComponentGuidance(BaseModel):
    """Knowledge-base grounding attached to an image analysis.

    ``retrieved_components`` is supplied by the server, not trusted from the
    model response. Identification and wiring feedback remain observations or
    proposed checks; the client must keep a human in the loop.
    """

    retrieved_components: list[RetrievedComponent] = Field(default_factory=list, max_length=5)
    identified_components: list[ComponentIdentification] = Field(default_factory=list, max_length=3)
    wiring_feedback: list[str] = Field(default_factory=list, max_length=8)
    clarifying_questions: list[str] = Field(default_factory=list, max_length=5)
    data_caveats: list[str] = Field(default_factory=list, max_length=5)


class VisionAnalysis(BaseModel):
    """Grounded VLM observations and VLA-style action proposals for one frame."""

    summary: str = Field(min_length=1, max_length=2_000)
    observations: list[str] = Field(default_factory=list, max_length=20)
    safety_alerts: list[str] = Field(default_factory=list, max_length=10)
    proposed_actions: list[ActionProposal] = Field(default_factory=list, max_length=10)
    component_guidance: ComponentGuidance | None = None


class VisionOutput(BaseModel):
    """Unified output from a segmentation-only or multimodal vision engine."""

    regions: list[SegmentationRegion] = Field(default_factory=list)
    analysis: VisionAnalysis | None = None


class SegmentationResult(BaseModel):
    session_id: str
    frame_id: str
    backend: str
    workflow_id: str | None = None
    completed_at: datetime = Field(default_factory=utc_now)
    latency_ms: float = Field(ge=0.0)
    regions: list[SegmentationRegion]
    analysis: VisionAnalysis | None = None


class SessionRequest(BaseModel):
    workflow_id: str = Field(default="generic-field-support", min_length=1, max_length=63)


class SessionCreated(BaseModel):
    session_id: str
    created_at: datetime
    workflow_id: str
    workflow_version: str


class IngestAcknowledgement(BaseModel):
    session_id: str
    frame_id: str
    accepted: bool = True
    dropped_stale_frames: int = Field(ge=0)


class AdvisoryFieldSignal(BaseModel):
    """Non-contact field observation; never evidence that equipment is deenergized."""

    level: float = Field(ge=0.0)
    state: Literal["ambient", "field_detected", "unknown"] = "unknown"
    observed_at: datetime = Field(default_factory=utc_now)


class AdvisoryFieldSignalAcknowledgement(BaseModel):
    session_id: str
    state: Literal["ambient", "field_detected", "unknown"]
    accepted: bool = True


class SessionMetrics(BaseModel):
    session_id: str
    received_frames: int = Field(ge=0)
    processed_frames: int = Field(ge=0)
    failed_frames: int = Field(ge=0)
    dropped_stale_frames: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
