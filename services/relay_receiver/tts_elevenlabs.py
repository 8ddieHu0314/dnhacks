"""ElevenLabs text-to-speech for the relay.

Streams raw 16-bit 24 kHz mono PCM for one sentence. The relay forwards the chunks to the phone
as base64 inside JSON, and the phone schedules them onto an AVAudioEngine player node, so the
wearer hears the first words a few hundred milliseconds after the sentence lands.

Config (services/relay_receiver/.env, sourced by run.sh):
  ELEVENLABS_API_KEY    required to enable; without it the phone falls back to Apple's voice
  ELEVENLABS_VOICE_ID   voice to use
  ELEVENLABS_MODEL      default eleven_flash_v2_5 (lowest latency)
"""
import os
from collections.abc import AsyncIterator

import httpx

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "mqlDiDxS84MhnMijtd3t")
MODEL_ID = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
SAMPLE_RATE = 24000
OUTPUT_FORMAT = f"pcm_{SAMPLE_RATE}"
CHUNK_BYTES = SAMPLE_RATE * 2 // 5          # 200 ms of s16 mono per message

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
    body = {"text": text, "model_id": MODEL_ID}
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    buf = b""
    async with _http().stream("POST", url, params={"output_format": OUTPUT_FORMAT}, json=body, headers=headers) as r:
        if r.status_code != 200:
            detail = (await r.aread())[:200].decode(errors="replace")
            raise RuntimeError(f"elevenlabs {r.status_code}: {detail}")
        async for piece in r.aiter_bytes():
            buf += piece
            while len(buf) >= CHUNK_BYTES:
                yield buf[:CHUNK_BYTES]
                buf = buf[CHUNK_BYTES:]
    if len(buf) >= 2:
        yield buf[: len(buf) - len(buf) % 2]
