from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass(frozen=True, slots=True)
class RelayFrame:
    captured_at: datetime
    image_bytes: bytes
    width: int
    height: int

_START_OF_FRAME = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                   0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}

def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Relay frame is not JPEG")
    offset = 2
    while offset + 9 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker in _START_OF_FRAME:
            height = int.from_bytes(data[offset + 5:offset + 7], "big")
            width = int.from_bytes(data[offset + 7:offset + 9], "big")
            if width and height:
                return width, height
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        size = int.from_bytes(data[offset + 2:offset + 4], "big")
        if size < 2:
            break
        offset += size + 2
    raise ValueError("JPEG dimensions are missing")

def decode_relay_frame(payload: bytes) -> RelayFrame:
    if payload.startswith(b"\xff\xd8"):
        image, captured_at = payload, datetime.now(timezone.utc)
    elif len(payload) > 10 and payload[8:10] == b"\xff\xd8":
        timestamp_ms = struct.unpack(">Q", payload[:8])[0]
        image = payload[8:]
        captured_at = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
    else:
        raise ValueError("Expected timestamp-prefixed or bare JPEG")
    width, height = jpeg_dimensions(image)
    return RelayFrame(captured_at, image, width, height)
