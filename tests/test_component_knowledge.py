import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from vision_api.component_knowledge import ComponentKnowledgeBase, ComponentKnowledgeVLM
from vision_api.main import app
from vision_api.models import Frame, FrameMetadata
from vision_api.segmentation import build_vision_engine


class ComponentKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.knowledge = ComponentKnowledgeBase.from_path(
            Path(__file__).parents[1] / "docs/components/components.json"
        )

    async def test_grounds_identifications_to_retrieved_records(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            content = {"scene_description": "HC-SR04 sensor", "visible_text": ["HC-SR04"]} if len(calls) == 1 else {
                "regions": [], "analysis": {"summary": "An ultrasonic sensor is visible.",
                "component_guidance": {"identified_components": [
                    {"component_id": "hc-sr04", "confidence": 0.9, "observed_evidence": ["visible label"]},
                    {"component_id": "made-up", "confidence": 0.9, "observed_evidence": []},
                ]}}}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})

        engine = ComponentKnowledgeVLM(base_url="https://model.example/v1", api_key=None,
            model="test-vlm", timeout_seconds=1, knowledge_base=self.knowledge,
            transport=httpx.MockTransport(handler))
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2,
            user_request="What is this sensor?"), b"x", datetime.now(timezone.utc))
        output = await engine.analyze(frame)

        self.assertEqual(len(calls), 2)
        self.assertIn("What is this sensor?", calls[0]["messages"][0]["content"])
        self.assertIn("hc-sr04", calls[1]["messages"][0]["content"])
        self.assertIn("sources", calls[1]["messages"][0]["content"])
        self.assertEqual(output.analysis.component_guidance.retrieved_components[0].component_id, "hc-sr04")
        self.assertEqual([item.component_id for item in output.analysis.component_guidance.identified_components], ["hc-sr04"])

    async def test_no_key_mode_exercises_typed_component_retrieval(self) -> None:
        engine = build_vision_engine(base_url=None, api_key=None, backend="component_knowledge_mock",
            model="", timeout_seconds=1, component_knowledge=self.knowledge)
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2,
            user_request="HC-SR04 ultrasonic sensor"), b"x", datetime.now(timezone.utc))
        output = await engine.analyze(frame)
        self.assertEqual(output.analysis.component_guidance.retrieved_components[0].component_id, "hc-sr04")
        self.assertIn("not simulated", output.analysis.summary)

    def test_component_endpoints_search_and_return_a_record(self) -> None:
        with TestClient(app) as client:
            search = client.get("/v1/components/search", params={"q": "hc-sr04"})
            record = client.get("/v1/components/hc-sr04")
        self.assertEqual(search.status_code, 200)
        self.assertEqual(search.json()[0]["component_id"], "hc-sr04")
        self.assertEqual(record.json()["canonical_name"], "HC-SR04 ultrasonic distance sensor")
