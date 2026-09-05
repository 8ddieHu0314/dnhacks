"""Provider-neutral adapter for OpenAI-compatible vision-language models."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from .models import Frame, VisionOutput


class VisionModelError(RuntimeError):
    """A model endpoint failed or returned an invalid structured response."""


DEFAULT_INSTRUCTION = (
    "Analyze this field-view image. Return only JSON with regions and optional analysis. "
    "Any proposed action must be advisory and require human confirmation."
)


def instruction_for(frame: Frame) -> str:
    """Merge the generic safety boundary with the session's workflow package."""

    if frame.workflow is None:
        return DEFAULT_INSTRUCTION
    checkpoints = "\n".join(
        f"- {checkpoint.title}: {checkpoint.evidence_prompt}"
        for checkpoint in frame.workflow.checkpoints
    )
    return (
        f"{DEFAULT_INSTRUCTION}\n\n"
        f"Selected workflow ({frame.workflow.id} v{frame.workflow.version}): "
        f"{frame.workflow.title}\n{frame.workflow.instructions}\n\n"
        f"Workflow checkpoints:\n{checkpoints}"
    )


def encoded_image_url(frame: Frame) -> str:
    """Return a data URL accepted by OpenAI-compatible image message formats."""

    image_base64 = base64.b64encode(frame.image_bytes).decode("ascii")
    return f"data:image/{frame.metadata.encoding};base64,{image_base64}"


def completion_to_output(payload: dict[str, Any]) -> VisionOutput:
    """Validate the first chat-completion JSON message against our public schema."""

    try:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        if not isinstance(content, str):
            raise TypeError("completion content was not text")
        return VisionOutput.model_validate(json.loads(content))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise VisionModelError("VLM response did not contain a valid VisionOutput JSON object") from exc


class OpenAICompatibleVLM:
    """Vision-language/action adapter for a `/v1/chat/completions`-style service."""

    name = "openai-compatible-vlm"

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url or not model:
            raise ValueError("VLM_BASE_URL and VLM_MODEL are required for openai_compatible")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def _request_body(self, frame: Frame) -> dict[str, Any]:
        return {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instruction_for(frame)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Return the requested JSON for this frame."},
                        {"type": "image_url", "image_url": {"url": encoded_image_url(frame)}},
                    ],
                },
            ],
        }

    async def analyze(self, frame: Frame) -> VisionOutput:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=self._request_body(frame),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionModelError(f"VLM request failed: {exc}") from exc
        return completion_to_output(response.json())
