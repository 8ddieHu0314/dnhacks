from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .models import ComponentGuidance, Frame, VisionAnalysis, VisionOutput, RetrievedComponent
from .vlm import VisionModelError, encoded_image_url, instruction_for

_TOKEN = re.compile(r"[a-z0-9]+")

def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(value.lower())

@dataclass(frozen=True, slots=True)
class ComponentMatch:
    record: dict[str, Any]
    score: float

class ComponentKnowledgeBase:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("Component knowledge base must contain at least one record")
        self._records = records
        self._by_id = {str(record["id"]): record for record in records}
        if len(self._by_id) != len(records):
            raise ValueError("Component ids must be unique")
        self._search_text = [json.dumps(record, ensure_ascii=True).lower() for record in records]

    @classmethod
    def from_path(cls, path: Path) -> ComponentKnowledgeBase:
        try:
            records = json.loads(path.read_text())["components"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load component knowledge base at {path}") from exc
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("components.json must contain a components array of objects")
        return cls(records)

    def record_for(self, component_id: str) -> dict[str, Any] | None:
        return self._by_id.get(component_id)

    def search(self, query: str, *, limit: int = 3) -> list[ComponentMatch]:
        terms = _tokens(query)
        if limit < 1 or not terms:
            return []
        scores = [sum(text.count(term) for term in terms) for text in self._search_text]
        exact = [index for index, record in enumerate(self._records) if record["id"] in query.lower()]
        order = exact + sorted((i for i, score in enumerate(scores) if score and i not in exact), key=scores.__getitem__, reverse=True)
        return [ComponentMatch(self._records[index], max(float(scores[index]), 100.0 if index in exact else 0.0)) for index in order[:limit]]

    @staticmethod
    def summaries(matches: list[ComponentMatch]) -> list[RetrievedComponent]:
        summaries = []
        for match in matches:
            record = match.record
            confidence = str(record.get("confidence", "low")).lower()
            summaries.append(RetrievedComponent(
                component_id=str(record["id"]), canonical_name=str(record["canonical_name"]),
                data_confidence=confidence if confidence in {"high", "medium", "low"} else "low",
                retrieval_score=max(0.0, match.score),
            ))
        return summaries

    @staticmethod
    def prompt_records(matches: list[ComponentMatch]) -> list[dict[str, Any]]:
        fields = ("identity", "function", "visual_identification", "pins", "electrical",
                  "wiring_to_uno", "safety", "troubleshooting")
        records = []
        for match in matches:
            record, details = match.record, match.record.get("details", {})
            records.append({
                "id": record["id"], "canonical_name": record["canonical_name"],
                "data_confidence": record.get("confidence", "low"),
                "source_count": record.get("source_count", 0),
                **{field: details.get(field, {}) for field in fields},
            })
        return records


class SceneDescription(BaseModel):
    scene_description: str = Field(min_length=1, max_length=1_500)
    visible_text: list[str] = Field(default_factory=list, max_length=20)
    likely_component_terms: list[str] = Field(default_factory=list, max_length=12)

    def query(self, user_request: str | None) -> str:
        return " ".join([self.scene_description, *self.visible_text, *self.likely_component_terms, user_request or ""])


class ComponentKnowledgeVLM:
    """Use a VLM only after a frame description narrows the component context."""

    name = "component-knowledge-vlm"

    def __init__(self, *, base_url: str | None, api_key: str | None, model: str,
                 timeout_seconds: float, knowledge_base: ComponentKnowledgeBase, top_k: int = 3,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not base_url or not model:
            raise ValueError("VLM_BASE_URL and VLM_MODEL are required for component_knowledge")
        if not 1 <= top_k <= 5:
            raise ValueError("COMPONENT_KNOWLEDGE_TOP_K must be between 1 and 5")
        self._base_url, self._api_key, self._model = base_url.rstrip("/"), api_key, model
        self._timeout, self._knowledge_base, self._top_k = timeout_seconds, knowledge_base, top_k
        self._transport = transport

    def _body(self, *, system: str, text: str, frame: Frame) -> dict[str, Any]:
        return {"model": self._model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": encoded_image_url(frame)}},
                ]}]}

    async def _complete(self, *, system: str, text: str, frame: Frame) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(f"{self._base_url}/chat/completions", headers=headers,
                                             json=self._body(system=system, text=text, frame=frame))
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(item.get("text", "") for item in content if item.get("type") == "text")
            if not isinstance(content, str):
                raise TypeError("completion content was not text")
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise TypeError("completion JSON was not an object")
            return payload
        except httpx.HTTPError as exc:
            raise VisionModelError(f"VLM request failed: {exc}") from exc
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VisionModelError("VLM response did not contain a valid JSON object") from exc

    @staticmethod
    def _scene_prompt(user_request: str | None) -> str:
        return """Describe only what is visible in this electronics-kit image. Return JSON with
scene_description, visible_text, and likely_component_terms. Transcribe markings verbatim;
do not identify a part with certainty or give wiring advice. User request: """ + (user_request or "none")

    @staticmethod
    def _guidance_prompt(frame: Frame, records: list[dict[str, Any]]) -> str:
        ids = [record["id"] for record in records]
        return f"""You are an educational electronics-kit copilot. Return only JSON matching
VisionOutput. Use only the candidate records below; allowed component ids are {ids}. If the
image cannot distinguish them, identify none and ask a short clarifying question. Never invent
a specification, component id, or connection. Image evidence cannot prove a wire is electrically
connected, correct, or safe: state uncertainty and propose a human-confirmed check. Do not direct
mains/high-voltage work. Every proposed action requires_confirmation=true. Put component evidence,
wiring feedback, questions, and caveats in analysis.component_guidance.\n\nWorkflow:\n{instruction_for(frame)}
\n\nUser request: {frame.metadata.user_request or 'identify the part and give safe context.'}
\n\nCandidate records:\n{json.dumps(records, ensure_ascii=True)}"""
