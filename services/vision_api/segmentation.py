from __future__ import annotations

import hashlib
from typing import Protocol

from .models import BoundingBox, Frame, SegmentationRegion, VisionOutput
from .vlm import OpenAICompatibleVLM


class SegmentationEngine(Protocol):
    """Contract implemented by a segmentation model or multimodal adapter."""

    name: str

    async def analyze(self, frame: Frame) -> VisionOutput: ...


class MockSegmentationEngine:
    """Deterministic placeholder that exercises the full real-time data path.

    It is deliberately not presented as computer vision. Replace this class with a
    SAM 2, YOLO-seg, or custom model adapter once the runtime and model weights are
    chosen. Keeping the response shape real lets the mobile/UI side be built now.
    """

    name = "mock"

    async def segment(self, frame: Frame) -> list[SegmentationRegion]:
        digest = hashlib.blake2s(frame.image_bytes, digest_size=4).digest()
        # Produce a stable, bounded region so repeated frames render identically.
        x = 0.12 + (digest[0] / 255) * 0.18
        y = 0.12 + (digest[1] / 255) * 0.18
        width = 0.32 + (digest[2] / 255) * 0.14
        height = 0.32 + (digest[3] / 255) * 0.14
        return [
            SegmentationRegion(
                label="mock-region",
                confidence=0.50,
                bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
                polygon=[x, y, x + width, y, x + width, y + height, x, y + height],
            )
        ]

    async def analyze(self, frame: Frame) -> VisionOutput:
        return VisionOutput(regions=await self.segment(frame))


def build_segmentation_engine(backend: str) -> SegmentationEngine:
    if backend == "mock":
        return MockSegmentationEngine()
    raise ValueError(
        f"Unsupported SEGMENTATION_BACKEND={backend!r}. "
        "Implement the SegmentationEngine protocol and register it here."
    )


def build_vision_engine(
    *, base_url: str | None, api_key: str | None, backend: str, model: str, timeout_seconds: float
) -> SegmentationEngine:
    if backend == "mock":
        return MockSegmentationEngine()
    if backend == "openai_compatible":
        return OpenAICompatibleVLM(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported VISION_BACKEND={backend!r}")
