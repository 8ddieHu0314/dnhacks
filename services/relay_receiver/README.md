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
    ELEVENLABS_GAIN=2.0                         # louder for the glasses; 1.0 = as rendered, clips above ~2.4
    ELEVENLABS_SPEED=1.0                        # 0.7 to 1.2
    ELEVENLABS_STABILITY=0.5                    # lower = more expressive
    ELEVENLABS_STYLE=0.0                        # style exaggeration, adds latency above 0

`GET /health` reports `tts.provider`, characters sent, and the last error. The starter plan
caps at 30,000 characters a month, roughly 300 narration sentences.

## Local component detector in front of Claude (boxes on the dashboard)

`detect.py` runs a small YOLOv8 ONNX on every incoming frame with plain onnxruntime (about 30 ms
on the Mac CPU, no Roboflow runtime, no key at run time). It is the Roboflow Universe model
`gab-qwrl3/arduino-lcxdx` v3 (3647 images, CC BY 4.0) with 14 Arduino-kit classes: arduino, lcd,
servomotor, sensor ultrasonico, 7-seg, pot, led, resistencia, diodo, transistor, pulsador, plus
drv8825, esp82 and ttl which are not in our kit.

What it does:

- Dashboard: every frame's boxes are pushed to `/ws/view` as
  `{"type":"detections","ts","ms","boxes":[{"label","name","conf","box":[x1,y1,x2,y2],"catalog_hint"}]}`
  (normalized coordinates) and drawn over the video, following the Rotate and Fit transforms.
  The box that triggered Claude turns green and carries Claude's answer. Toggle with the Boxes button.
- Optimistic trigger for reactive identify: as soon as the top box has held still for
  `stable_frames` (2) frames and is a new label, Claude is called with the padded crop and a hint
  ("local detector flagged servo 87%, likely catalog id servo-sg90"). Claude still reads the
  markings and picks the catalog id; the hint is only a prior. With no boxes in view the original
  motion-settle rule runs on the whole frame, so parts outside the 14 classes still get identified.
- `POST /identify` also uses the top box when there is one. `GET /detections` returns the latest
  boxes. `/health` reports `detector` (ms, boxes, frames) and `triggers` (how many Claude calls
  came from the detector vs the settle rule). `GET /debug/tasks` shows the background loops.
- Tuning via `POST /reactive`: `det_min_conf` (0.5), `stable_frames` (2). Env: `DETECT=0`
  disables the detector, `DETECT_CONF` (0.45) and `DETECT_IOU` (0.5) set the raw NMS thresholds,
  `DETECT_ONNX` points at another model.

Weights are gitignored. Fetch them once with the Roboflow key in the repo-root `.env`
(Roboflow publishes no `.pt`, only this ONNX bundle through its inference cache):

    cd <repo>
    .venv-inf/bin/python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); \
      from inference import get_model; get_model(model_id='arduino-lcxdx/3', api_key=os.environ['ROBOFLOW_API_KEY'])"
    d=$(ls -d /tmp/cache/models-cache/v2-arduino-lcxdx-3-*/*/ | head -1)
    cp -L "$d/weights.onnx" weights/components_yolov8.onnx
    cp -L "$d/class_names.txt" weights/components_yolov8.classes.txt

Without the files the relay logs `weights missing` and behaves exactly as before.
