import struct
import unittest

from vision_api.glasses_bridge import decode_relay_frame, jpeg_dimensions


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


if __name__ == "__main__":
    unittest.main()
