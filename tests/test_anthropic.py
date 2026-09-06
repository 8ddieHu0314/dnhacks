import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vision_api.component_knowledge import AnthropicComponentKnowledgeVLM, ComponentKnowledgeBase
from vision_api.models import Frame, FrameMetadata


class AnthropicComponentKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    def test_accepts_json_surrounded_by_model_text(self) -> None:
        payload = AnthropicComponentKnowledgeVLM._json_object("Here is the result: {\"ok\": true}")
        self.assertEqual(payload, {"ok": True})

    async def test_sends_base64_frames_to_messages_api(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            content = {"scene_description": "HC-SR04", "visible_text": ["HC-SR04"]}
            if len(calls) == 2:
                content = {"regions": [], "analysis": {"summary": "Sensor visible.",
                    "component_guidance": {"identified_components": [{"component_id": "hc-sr04",
                    "confidence": 0.9, "observed_evidence": ["label"]}]}}}
            return httpx.Response(200, json={"content": [{"type": "tool_use", "name": "submit_result", "input": content}]})

        knowledge = ComponentKnowledgeBase.from_path(Path(__file__).parents[1] / "docs/components/components.json")
        engine = AnthropicComponentKnowledgeVLM(base_url="https://model.example", api_key="test-key",
            model="claude-test", timeout_seconds=1, knowledge_base=knowledge, transport=httpx.MockTransport(handler))
        self.assertEqual(engine._timeout, 45.0)
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2), b"image", datetime.now(timezone.utc))
        output = await engine.analyze(frame)

        payload = json.loads(calls[0].content)
        self.assertEqual(calls[0].url.path, "/v1/messages")
        self.assertEqual(calls[0].headers["x-api-key"], "test-key")
        self.assertEqual(calls[0].headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["messages"][0]["content"][1]["source"]["data"], "aW1hZ2U=")
        self.assertEqual(payload["tool_choice"]["type"], "tool")
        self.assertEqual(json.loads(calls[1].content)["tools"][0]["input_schema"]["required"], ["analysis"])
        self.assertEqual(output.analysis.component_guidance.identified_components[0].component_id, "hc-sr04")

    async def test_uses_typed_cue_without_a_scene_round_trip(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            content = {"analysis": {"summary": "Breadboard visible.", "component_guidance": {
                "identified_components": [{"component_id": "breadboard-830", "confidence": 0.9}]}}}
            return httpx.Response(200, json={"content": [{"type": "tool_use", "name": "submit_result", "input": content}]})

        knowledge = ComponentKnowledgeBase.from_path(Path(__file__).parents[1] / "docs/components/components.json")
        engine = AnthropicComponentKnowledgeVLM(base_url="https://model.example", api_key="test-key",
            model="claude-test", timeout_seconds=1, knowledge_base=knowledge, transport=httpx.MockTransport(handler))
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2, user_request="breadboard"),
                      b"image", datetime.now(timezone.utc))
        await engine.analyze(frame)

        self.assertEqual(len(calls), 1)
