import json
import unittest
from datetime import datetime, timezone

import httpx

from vision_api.models import Frame, FrameMetadata
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
