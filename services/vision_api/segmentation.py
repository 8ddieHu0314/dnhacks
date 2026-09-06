from __future__ import annotations

import hashlib
from typing import Protocol

from .component_knowledge import AnthropicCatalogIdentificationVLM, AnthropicComponentKnowledgeVLM, ComponentKnowledgeBase, ComponentKnowledgeVLM
from .models import BoundingBox, ComponentGuidance, Frame, SegmentationRegion, VisionAnalysis, VisionOutput
from .vlm import OpenAICompatibleVLM


class VisionEngine(Protocol):
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


class MockComponentKnowledgeEngine:
    """No-key integration harness; it retrieves typed cues but does not inspect pixels."""

    name = "component-knowledge-mock"

    def __init__(self, knowledge_base: ComponentKnowledgeBase, top_k: int) -> None:
        self._knowledge_base, self._top_k = knowledge_base, top_k

    async def analyze(self, frame: Frame) -> VisionOutput:
        request = frame.metadata.user_request or ""
        matches = self._knowledge_base.search(request, limit=self._top_k)
        candidates = self._knowledge_base.summaries(matches)
        if not candidates:
            return VisionOutput(analysis=VisionAnalysis(
                summary="Local component mode cannot identify image pixels without a VLM.",
                component_guidance=ComponentGuidance(clarifying_questions=[
                    "Type a visible marking or component name to exercise retrieval."
                ]),
            ))
        caveats = [f"{item.canonical_name} has {item.data_confidence}-confidence kit data."
                   for item in candidates if item.data_confidence != "high"]
        return VisionOutput(analysis=VisionAnalysis(
            summary=f"Local retrieval found {candidates[0].canonical_name}; visual identification is not simulated.",
            observations=["This no-key mode retrieves from the typed request only."],
            component_guidance=ComponentGuidance(retrieved_components=candidates, data_caveats=caveats),
        ))


def build_vision_engine(
    *, base_url: str | None, api_key: str | None, backend: str, model: str, timeout_seconds: float,
    component_knowledge: ComponentKnowledgeBase | None = None, component_knowledge_top_k: int = 3,
) -> VisionEngine:
    if backend == "mock":
        return MockSegmentationEngine()
    if backend == "openai_compatible":
        return OpenAICompatibleVLM(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    if backend == "component_knowledge_mock":
        if component_knowledge is None:
            raise ValueError("A component knowledge base is required for component_knowledge_mock")
        return MockComponentKnowledgeEngine(component_knowledge, component_knowledge_top_k)
    if backend == "component_knowledge":
        if component_knowledge is None:
            raise ValueError("A component knowledge base is required for component_knowledge")
        return ComponentKnowledgeVLM(
            base_url=base_url, api_key=api_key, model=model, timeout_seconds=timeout_seconds,
            knowledge_base=component_knowledge, top_k=component_knowledge_top_k,
        )
    if backend == "anthropic_component_knowledge":
        if component_knowledge is None:
            raise ValueError("A component knowledge base is required for anthropic_component_knowledge")
        return AnthropicComponentKnowledgeVLM(
            base_url=base_url, api_key=api_key, model=model, timeout_seconds=timeout_seconds,
            knowledge_base=component_knowledge, top_k=component_knowledge_top_k,
        )
    if backend == "anthropic_catalog_identification":
        if component_knowledge is None:
            raise ValueError("A component knowledge base is required for anthropic_catalog_identification")
        return AnthropicCatalogIdentificationVLM(
            base_url=base_url, api_key=api_key, model=model, timeout_seconds=timeout_seconds,
            knowledge_base=component_knowledge, top_k=component_knowledge_top_k,
        )
    raise ValueError(f"Unsupported VISION_BACKEND={backend!r}")
