"""Provider-neutral adapter for OpenAI-compatible vision-language models."""

from __future__ import annotations

import base64
import json
from typing import Any

from .models import Frame, VisionOutput


class VisionModelError(RuntimeError):
    """A model endpoint failed or returned an invalid structured response."""


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
