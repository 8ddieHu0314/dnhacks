# Relay receiver (Mac)

Minimal receiver used during the hackathon to prove the glasses -> phone -> laptop pipeline.
Accepts frames from the iOS app over `WS /ws/ingest` (also `WS /`), fans them out to browsers on
`WS /ws/view`, serves a live dashboard on `/` (canvas, rotate/fit, latency HUD), and exposes
`POST /inspect` (Claude vision on the latest frame; needs `ANTHROPIC_API_KEY`).

    ./run.sh          # venv + deps, listens on 0.0.0.0:8787, advertises _glassesrelay._tcp via Bonjour

The Bonjour TXT record carries the receiver's current URLs (refreshed every 10 s) so the phone
never needs a typed IP. `--ws-ping-timeout 90` matters: the phone's socket stalls otherwise.

## Claude -> glasses audio
`POST /inspect` and `POST /narrate {"enabled":true,"interval":8}` run Claude on the latest frame and stream each
finished sentence to the connected phone(s) as `{"type":"speak","text":...}` then `{"type":"speak_end"}`; the iOS
app speaks them into the glasses. Put `ANTHROPIC_API_KEY=...` in `services/relay_receiver/.env` (gitignored);
`INSPECT_FAKE=1 ./run.sh` streams canned sentences to test the audio path without a key.

## Voice: ElevenLabs on the Mac, Apple voice as fallback

With `ELEVENLABS_API_KEY` in `services/relay_receiver/.env`, the relay renders each finished
sentence with ElevenLabs (`eleven_flash_v2_5`, raw 24 kHz PCM) and streams it to the phone as
`{"type":"audio","id":n,"rate":24000,"pcm":"<base64>"}` chunks of about 200 ms, ending with
`{"type":"audio_end","id":n}`. The matching `speak` message carries `"audio": true` so the phone
shows the caption but does not synthesize. If ElevenLabs fails before any audio is sent, the
relay sends `{"type":"speak_fallback","text":...}` and the phone reads it with Apple's voice.
Without the key, behaviour is unchanged: plain `speak` messages, Apple voice on the phone.

    ELEVENLABS_API_KEY=sk_...
    ELEVENLABS_VOICE_ID=mqlDiDxS84MhnMijtd3t    # Christopher, Friendly American
    ELEVENLABS_MODEL=eleven_flash_v2_5          # optional

`GET /health` reports `tts.provider`, characters sent, and the last error. The starter plan
caps at 30,000 characters a month, roughly 300 narration sentences.
