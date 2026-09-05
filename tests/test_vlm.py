import json
import unittest
from datetime import datetime, timezone

import httpx

from vision_api.models import Frame, FrameMetadata, WorkflowCheckpoint, WorkflowDefinition
from vision_api.vlm import OpenAICompatibleVLM


class OpenAICompatibleVLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_a_frame_and_validates_vla_proposals(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            image = payload["messages"][1]["content"][1]["image_url"]["url"]
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertTrue(image.startswith("data:image/jpeg;base64,"))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": json.dumps({
                        "regions": [],
                        "analysis": {
                            "summary": "A hard hat is visible.",
                            "proposed_actions": [{
                                "action": "Inspect PPE fit",
                                "rationale": "Helmet position is uncertain.",
                                "confidence": 0.7,
                            }],
                        },
                    })}}],
                },
            )

        engine = OpenAICompatibleVLM(
            base_url="https://model.example/v1",
            api_key="test-key",
            model="test-vlm",
            timeout_seconds=1,
            transport=httpx.MockTransport(handler),
        )
        frame = Frame(
            "session",
            FrameMetadata(frame_id="frame", width=2, height=2),
            b"x",
            datetime.now(timezone.utc),
        )
        output = await engine.analyze(frame)
        self.assertTrue(output.analysis.proposed_actions[0].requires_confirmation)

    def test_includes_selected_workflow_in_the_model_instruction(self) -> None:
        workflow = WorkflowDefinition(
            id="asset-inspection",
            version="1",
            title="Asset inspection",
            instructions="Identify the asset and report uncertainty.",
            checkpoints=[
                WorkflowCheckpoint(
                    id="identify-asset",
                    title="Identify asset",
                    evidence_prompt="Read visible asset labels.",
                )
            ],
        )
        frame = Frame(
            "session",
            FrameMetadata(frame_id="frame", width=2, height=2),
            b"x",
            datetime.now(timezone.utc),
            workflow,
        )
        engine = OpenAICompatibleVLM(
            base_url="https://model.example/v1",
            api_key=None,
            model="test-vlm",
            timeout_seconds=1,
        )

        instruction = engine._request_body(frame)["messages"][0]["content"]
        self.assertIn("asset-inspection v1", instruction)
        self.assertIn("Read visible asset labels.", instruction)
