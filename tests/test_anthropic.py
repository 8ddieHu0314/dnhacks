import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vision_api.component_knowledge import AnthropicComponentKnowledgeVLM, ComponentKnowledgeBase
from vision_api.models import Frame, FrameMetadata


class AnthropicComponentKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_base64_frames_to_messages_api(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            content = {"scene_description": "HC-SR04", "visible_text": ["HC-SR04"]}
            if len(calls) == 2:
                content = {"regions": [], "analysis": {"summary": "Sensor visible.",
                    "component_guidance": {"identified_components": [{"component_id": "hc-sr04",
                    "confidence": 0.9, "observed_evidence": ["label"]}]}}}
            return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(content)}]})

        knowledge = ComponentKnowledgeBase.from_path(Path(__file__).parents[1] / "docs/components/components.json")
        engine = AnthropicComponentKnowledgeVLM(base_url="https://model.example", api_key="test-key",
            model="claude-test", timeout_seconds=1, knowledge_base=knowledge, transport=httpx.MockTransport(handler))
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2), b"image", datetime.now(timezone.utc))
        output = await engine.analyze(frame)

        payload = json.loads(calls[0].content)
        self.assertEqual(calls[0].url.path, "/v1/messages")
        self.assertEqual(calls[0].headers["x-api-key"], "test-key")
        self.assertEqual(calls[0].headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["messages"][0]["content"][1]["source"]["data"], "aW1hZ2U=")
        self.assertEqual(output.analysis.component_guidance.identified_components[0].component_id, "hc-sr04")
