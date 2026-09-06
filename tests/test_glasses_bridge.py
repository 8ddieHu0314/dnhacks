import struct
import time
import unittest

from fastapi.testclient import TestClient

from vision_api.glasses_bridge import decode_relay_frame, jpeg_dimensions
from vision_api.main import app


JPEG = b"\xff\xd8\xff\xc0\x00\x11\x08\x01\xe0\x02\x80"


class GlassesBridgeTests(unittest.TestCase):
    def test_reads_dimensions_from_a_jpeg_start_of_frame(self) -> None:
        self.assertEqual(jpeg_dimensions(JPEG), (640, 480))

    def test_decodes_the_ios_timestamp_prefix(self) -> None:
        timestamp_ms = 1_700_000_000_125
        frame = decode_relay_frame(struct.pack(">Q", timestamp_ms) + JPEG)
        self.assertEqual(frame.image_bytes, JPEG)
        self.assertEqual((frame.width, frame.height), (640, 480))
        self.assertEqual(int(frame.captured_at.timestamp() * 1000), timestamp_ms)

    def test_accepts_the_legacy_bare_jpeg(self) -> None:
        self.assertEqual(decode_relay_frame(JPEG).image_bytes, JPEG)

    def test_rejects_unknown_binary_messages(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected"):
            decode_relay_frame(b"not a frame")

    def test_existing_ios_websocket_reaches_the_vision_pipeline(self) -> None:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/ingest") as socket:
                hello = socket.receive_json()
                socket.send_bytes(struct.pack(">Q", 1_700_000_000_125) + JPEG)
                for _ in range(50):
                    latest = client.get(f"/v1/sessions/{hello['session_id']}/results/latest")
                    if latest.status_code == 200:
                        break
                    time.sleep(0.01)
        self.assertEqual(hello["type"], "relay_session")
        self.assertEqual(latest.json()["captured_at"], "2023-11-14T22:13:20.125000Z")


if __name__ == "__main__":
    unittest.main()
