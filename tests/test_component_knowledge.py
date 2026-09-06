import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from vision_api.component_knowledge import ComponentKnowledgeBase, ComponentKnowledgeVLM, SceneDescription
from vision_api.main import app
from vision_api.models import Frame, FrameMetadata, VisionAnalysis
from vision_api.segmentation import build_vision_engine


class ComponentKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    def test_debug_steps_require_confirmation_and_can_request_a_view(self) -> None:
        analysis = VisionAnalysis.model_validate({"summary": "Inspect power.", "mode": "debug",
            "debug_guidance": {"status": "needs_context", "problem": "Power is unclear.",
                "steps": [{"instruction": "Disconnect power.", "reason": "Avoid shorts.",
                    "expected_evidence": "Power LED turns off.", "requires_confirmation": False}],
                "visual_clarification": {"target": "power rails", "requested_view": "top-down close-up",
                    "reason": "Rail breaks are hidden."}}})

        self.assertTrue(analysis.debug_guidance.steps[0].requires_confirmation)
        self.assertEqual(analysis.debug_guidance.visual_clarification.requested_view, "top-down close-up")

    def test_accepts_string_or_mapping_retrieval_cues(self) -> None:
        scene = SceneDescription.model_validate({"scene_description": "breadboard",
            "visible_text": {"label": "HC-SR04"}, "likely_component_terms": "sensor"})
        self.assertEqual(scene.visible_text, ["HC-SR04"])
        self.assertEqual(scene.likely_component_terms, ["sensor"])

    def test_prefers_components_named_in_a_typed_cue(self) -> None:
        ids = [match.record["id"] for match in self.knowledge.search("breadboard and jumper wires")]
        self.assertIn("breadboard-830", ids)
        self.assertIn("jumper-wire", ids)

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
        frame = Frame("session", FrameMetadata(frame_id="frame", width=2, height=2),
                      b"x", datetime.now(timezone.utc))
        output = await engine.analyze(frame)

        self.assertEqual(len(calls), 2)
        self.assertIn("Describe this frame for retrieval.", calls[0]["messages"][1]["content"][0]["text"])
        self.assertIn("hc-sr04", calls[1]["messages"][0]["content"])
        records = json.loads(calls[1]["messages"][0]["content"].split("Candidate records:\n", 1)[1])
        self.assertNotIn("sources", records[0])
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
