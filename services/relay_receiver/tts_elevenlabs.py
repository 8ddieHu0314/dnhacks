"""ElevenLabs text-to-speech for the relay.

Streams raw 16-bit 24 kHz mono PCM for one sentence. The relay forwards the chunks to the phone
as base64 inside JSON, and the phone schedules them onto an AVAudioEngine player node, so the
wearer hears the first words a few hundred milliseconds after the sentence lands.

Config (services/relay_receiver/.env, sourced by run.sh):
  ELEVENLABS_API_KEY    required to enable; without it the phone falls back to Apple's voice
  ELEVENLABS_VOICE_ID   voice to use
  ELEVENLABS_MODEL      default eleven_flash_v2_5 (lowest latency)
  ELEVENLABS_GAIN       linear gain applied to the PCM before it leaves the Mac, default 2.0.
                        Glasses over Bluetooth run quiet; samples are hard clipped at full scale.
  ELEVENLABS_SPEED      0.7 to 1.2, default 1.0
  ELEVENLABS_STABILITY  0.0 to 1.0, lower is more expressive, default 0.5
  ELEVENLABS_STYLE      0.0 to 1.0, style exaggeration, default 0.0 (costs latency above 0)
"""
import os
from collections.abc import AsyncIterator

import httpx
import numpy as np

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "mqlDiDxS84MhnMijtd3t")
MODEL_ID = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
SAMPLE_RATE = 24000
OUTPUT_FORMAT = f"pcm_{SAMPLE_RATE}"
CHUNK_BYTES = SAMPLE_RATE * 2 // 5          # 200 ms of s16 mono per message
GAIN = float(os.environ.get("ELEVENLABS_GAIN", "2.0"))
VOICE_SETTINGS = {
    "speed": float(os.environ.get("ELEVENLABS_SPEED", "1.0")),
    "stability": float(os.environ.get("ELEVENLABS_STABILITY", "0.5")),
    "similarity_boost": 0.75,
    "style": float(os.environ.get("ELEVENLABS_STYLE", "0.0")),
}

_client: httpx.AsyncClient | None = None


def enabled() -> bool:
    return bool(API_KEY)


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url="https://api.elevenlabs.io", timeout=httpx.Timeout(20.0, connect=5.0))
    return _client


async def stream_pcm(text: str) -> AsyncIterator[bytes]:
    """Yield PCM chunks of about 200 ms each. Raises httpx errors on failure."""
    url = f"/v1/text-to-speech/{VOICE_ID}/stream"
    body = {"text": text, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS}
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    buf = b""
    async with _http().stream("POST", url, params={"output_format": OUTPUT_FORMAT}, json=body, headers=headers) as r:
        if r.status_code != 200:
            detail = (await r.aread())[:200].decode(errors="replace")
            raise RuntimeError(f"elevenlabs {r.status_code}: {detail}")
        async for piece in r.aiter_bytes():
            buf += piece
            while len(buf) >= CHUNK_BYTES:
                yield _gain(buf[:CHUNK_BYTES])
                buf = buf[CHUNK_BYTES:]
    if len(buf) >= 2:
        yield _gain(buf[: len(buf) - len(buf) % 2])


def _gain(chunk: bytes) -> bytes:
    if GAIN == 1.0:
        return chunk
    a = np.frombuffer(chunk, dtype="<i2").astype(np.float32) * GAIN
    return np.clip(a, -32768, 32767).astype("<i2").tobytes()
