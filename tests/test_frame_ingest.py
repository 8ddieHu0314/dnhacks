import json
import time
import unittest

from fastapi.testclient import TestClient

from vision_api.main import app


class FrameIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.client.__enter__()
        self.session_id = self.client.post("/v1/sessions").json()["session_id"]

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def post_frame(self, body: bytes, encoding: str = "jpeg"):
        metadata = {"frame_id": "frame", "width": 2, "height": 2, "encoding": encoding}
        return self.client.post(f"/v1/sessions/{self.session_id}/frames", content=body,
                                headers={"x-frame-metadata": json.dumps(metadata)})

    def test_accepts_a_matching_jpeg_signature(self) -> None:
        self.assertEqual(self.post_frame(b"\xff\xd8\xff\x00").status_code, 200)

    def test_rejects_non_image_bytes(self) -> None:
        self.assertEqual(self.post_frame(b"not-an-image").status_code, 415)

    def test_rejects_an_encoding_mismatch(self) -> None:
        self.assertEqual(self.post_frame(b"\xff\xd8\xff\x00", encoding="png").status_code, 422)

    def test_serves_the_webcam_test_page(self) -> None:
        page = self.client.get("/webcam")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Component knowledge webcam test", page.text)
        self.assertIn("latestResult", page.text)
        self.assertIn("Exact JPEG submitted", page.text)

    def test_returns_only_the_latest_result_for_polling_clients(self) -> None:
        self.assertEqual(self.client.get(f"/v1/sessions/{self.session_id}/results/latest").status_code, 204)
        self.post_frame(b"\xff\xd8\xff\x00")
        for _ in range(20):
            latest = self.client.get(f"/v1/sessions/{self.session_id}/results/latest")
            if latest.status_code == 200:
                break
            time.sleep(0.01)
        self.assertEqual(latest.json()["frame_id"], "frame")
