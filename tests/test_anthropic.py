import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vision_api.component_knowledge import AnthropicCatalogIdentificationVLM, AnthropicComponentKnowledgeVLM, ComponentKnowledgeBase
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
            return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(content)}]})

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
        guidance_body = json.loads(calls[1].content)
        schema = guidance_body["tools"][0]["input_schema"]
        self.assertEqual(schema["required"], ["analysis"])
        self.assertEqual(schema["properties"]["analysis"]["properties"]["debug_guidance"]["properties"]["steps"]["maxItems"], 8)
        self.assertEqual(guidance_body["max_tokens"], 1400)
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

    async def test_uses_the_cached_catalog_identification_shape(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            content = {"id": "hc-sr04", "confidence": 0.9, "name": "Ultrasonic sensor", "evidence": "two round transducers"}
            return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(content)}]})

        knowledge = ComponentKnowledgeBase.from_path(Path(__file__).parents[1] / "docs/components/components.json")
        engine = AnthropicCatalogIdentificationVLM(base_url="https://model.example", api_key="test-key",
            model="claude-test", timeout_seconds=1, knowledge_base=knowledge, transport=httpx.MockTransport(handler))
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2), b"image", datetime.now(timezone.utc))
        output = await engine.analyze(frame)

        body = json.loads(calls[0].content)
        self.assertEqual(body["max_tokens"], 160)
        self.assertEqual(body["system"][0]["cache_control"]["type"], "ephemeral")
        self.assertIn("breadboard-830", body["system"][0]["text"])
        self.assertIn("text:", body["system"][0]["text"])
        self.assertIn("looks:", body["system"][0]["text"])
        self.assertNotIn("temperature", body)
        self.assertNotIn("tools", body)
        self.assertEqual(output.analysis.mode, "identification")
        self.assertEqual(output.analysis.component_guidance.identified_components[0].component_id, "hc-sr04")

    async def test_marks_a_breadboard_identification_as_debug_mode(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            content = {"id": "breadboard-830", "confidence": 0.9, "name": "Breadboard", "evidence": "power rails"}
            return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(content)}]})

        knowledge = ComponentKnowledgeBase.from_path(Path(__file__).parents[1] / "docs/components/components.json")
        engine = AnthropicCatalogIdentificationVLM(base_url="https://model.example", api_key="test-key",
            model="claude-test", timeout_seconds=1, knowledge_base=knowledge, transport=httpx.MockTransport(handler))
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2, user_request="What is wrong?"),
                      b"image", datetime.now(timezone.utc))
        output = await engine.analyze(frame)

        self.assertEqual(len(calls), 1)
        self.assertEqual(output.analysis.mode, "debug")
        self.assertEqual(output.analysis.component_guidance.identified_components[0].component_id, "breadboard-830")

    async def test_treats_non_json_catalog_responses_as_unclear(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"content": [{"type": "text", "text": "No image is clear."}]})

        knowledge = ComponentKnowledgeBase.from_path(Path(__file__).parents[1] / "docs/components/components.json")
        engine = AnthropicCatalogIdentificationVLM(base_url="https://model.example", api_key="test-key",
            model="claude-test", timeout_seconds=1, knowledge_base=knowledge, transport=httpx.MockTransport(handler))
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2), b"image", datetime.now(timezone.utc))
        output = await engine.analyze(frame)

        self.assertEqual(output.analysis.component_guidance.identified_components, [])
        self.assertIn("Hold one component closer", output.analysis.component_guidance.clarifying_questions[0])
