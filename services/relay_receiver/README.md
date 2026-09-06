# Relay receiver (Mac)

Minimal receiver used during the hackathon to prove the glasses -> phone -> laptop pipeline.
Accepts frames from the iOS app over `WS /ws/ingest` (also `WS /`), fans them out to browsers on
`WS /ws/view`, serves a live dashboard on `/` (canvas, rotate/fit, latency HUD), and exposes
`POST /inspect` (Claude vision on the latest frame; needs `ANTHROPIC_API_KEY`). On top of that
sit the hands-free modes, the ElevenLabs voice, and the local component detector described below.

    ./run.sh          # venv + deps, listens on 0.0.0.0:8787, advertises _glassesrelay._tcp via Bonjour

The Bonjour TXT record carries the receiver's current URLs (refreshed every 10 s) so the phone
never needs a typed IP. `--ws-ping-timeout 90` matters: the phone's socket stalls otherwise.

## Claude -> glasses audio
`POST /inspect` and `POST /narrate {"enabled":true,"interval":8}` run Claude on the latest frame and stream each
finished sentence to the connected phone(s) as `{"type":"speak","text":...}` then `{"type":"speak_end"}`; the iOS
app speaks them into the glasses. Put `ANTHROPIC_API_KEY=...` in `services/relay_receiver/.env` (gitignored);
`INSPECT_FAKE=1 ./run.sh` streams canned sentences to test the audio path without a key.

## Hands-free modes and models

`POST /reactive {"enabled":true,"mode":"parts","preannounce":true}` (also the phone's gear menu
and the dashboard's Hands-free picker) starts the reactive loop in `identify.py`:

- `parts`: identify the component in view against `docs/components/components.json` and speak
  its catalog name the moment it changes. Two triggers feed the same Claude call: the local
  detector (a box stable for 2 frames, sent as a crop plus a hint) and the settle rule (the
  scene changed and the frame is sharp or motion stopped). A different part only cuts in on a
  running announcement above `interrupt_confidence`; otherwise it waits for the audio to end.
  Extra knobs in the same body: `min_confidence`, `settle_seconds`, `cooldown_seconds`,
  `interrupt_confidence`, `det_min_conf`, `stable_frames`, `short_spoken`.
- `scene`: free-text narration of what changed, spoken sentence by sentence.

Three Claude call sites, each with its own model, switchable at runtime with
`POST /models {"identify":"sonnet","scene":"sonnet","describe":"opus"}` (full model ids are
accepted too; env defaults are `IDENTIFY_MODEL`, `NARRATE_MODEL`, `INSPECT_MODEL`):

| Path | Default | Why |
|---|---|---|
| Parts (`identify`) | `claude-sonnet-5` | about 1.8 s to the id (Opus about 3.5 s); both read markings well |
| Scene (`scene`) | `claude-sonnet-5` | latency first |
| Describe button and `POST /inspect` (`describe`) | `claude-opus-5` | one shot, quality first |
| Voice questions (`ask`) | `claude-sonnet-5` | short spoken answers, latency first |

`POST /voice {"provider":"apple"}` or `"elevenlabs"` switches the voice. The phone re-sends its
saved mode, models, and voice every time it connects, so a change made on the dashboard is
undone by the next phone reconnect unless it is also changed in the phone's gear menu.
`GET /reactive` and `GET /health` report the active settings, `POST /catalog/reload` re-reads
the catalog after an edit, and `POST /identify` runs one identification of the latest frame
without speaking.

## Voice input: questions from the wearer

The phone recognizes speech (glasses mic over Bluetooth HFP, or the phone mic) and sends each
finished sentence as `{"type":"ask","text":...}`; `POST /ask {"text":...}` and the dashboard's
"Ask (as voice)" button take the same path. `handle_ask()` routes:

| Heard | Intent | What happens |
|---|---|---|
| "stop", "hush", "quiet" (3 words or fewer) | hush | the answer being streamed is cut, queued sentences dropped, phone told `speak_stop` |
| "what is this", "which part", "identify" | identify | fresh identification of the latest frame, catalog line spoken in full even if it was just announced |
| "describe", "what do you see", "hazards" | describe | the Describe path (Opus) on a 768 px frame |
| anything else | answer | `ASK_PROMPT` on `ASK_MODEL` (Sonnet) with the frame and `identify.context_for_question()`, the catalog record of the last identified part (pins, electrical, wiring, safety, troubleshooting) |

A question outranks narration: queued speech is dropped first, a running analysis is allowed to
finish (bounded), and the reactive loop is held off until the answer has played. Every ask lands
in the report with `kind: "ask"` and the question; `/health` reports `voice_in` (count, last
sentence, intents). `{"type":"hush"}` from the phone does the hush part alone.

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
    ELEVENLABS_SPEED=1.1                        # 0.7 to 1.2
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
- The "this is an electronic component" moment: the instant a box qualifies, the server sends
  `{"type":"detected","display","label","conf","box","catalog_hint","in_catalog"}` to dashboards
  and phones. The dashboard shows a pulsing banner, and with pre-announce on (default, checkbox
  next to Reactive identify, or `preannounce` in `POST /reactive`) the glasses hear the catalog's
  short name ("Servo Motor SG90.") right away. Claude's full line replaces it one or two seconds
  later. The banner then turns green when Claude's id matches the detector's catalog hint, orange
  when it disagrees. Measured in fake mode: cue at 0.9 s, Claude at 2.1 s.
- Agreement with the catalog: at startup `detect.bind_catalog` checks every label-to-id hint
  against `docs/components/components.json` and takes `name_on_kit` as the display name, so the
  boxes, the banner, the pre-announcement and Claude all use the same words. `/health` and
  `/reactive` report `agreement` (agree / disagree / no_hint) and each report entry stores
  `trigger`, `det` and `agrees`. 11 of the 14 classes map to kit parts; drv8825, esp82 and ttl
  do not and are announced by their generic names.
- Component-only by design. The PPE model (SH17: hands, helmet, face) is never loaded here.
  `DETECT_CLASSES=arduino,lcd,...` restricts the detector further if a broader ONNX is dropped in.
- The detector path identifies the exact frame the boxes were computed on (`latest_frame`), never
  a newer one, so a scene switch cannot announce the previous part over the new picture.
- `POST /identify` also uses the top box when there is one. `GET /detections` returns the latest
  boxes. `/health` reports `detector` (ms, boxes, frames) and `triggers` (how many Claude calls
  came from the detector vs the settle rule). `GET /debug/tasks` shows the background loops.
- Tuning via `POST /reactive`: `det_min_conf` (0.5), `stable_frames` (2). Env: `DETECT=0`
  disables the detector, `DETECT_CONF` (0.45) and `DETECT_IOU` (0.5) set the raw NMS thresholds,
  `DETECT_ONNX` points at another model.
- Runtime on/off: the phone's gear menu (Hands-free > Local part detector), the dashboard's
  "detector" checkbox, or `POST /detector {"enabled":false}`. Off clears the boxes, stops the
  early trigger and pre-announce, and lets the settle rule identify alone; `/health` reports
  `detector.enabled`. The phone re-sends its setting on every reconnect.

The weights are committed (`weights/components_yolov8.onnx`, 12 MB, plus the class list), so
a fresh clone runs the detector with no download. To re-export them from Roboflow (it
publishes no `.pt`, only this ONNX bundle through its inference cache), use the Roboflow key
in the repo-root `.env`:

    cd <repo>
    .venv-inf/bin/python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); \
      from inference import get_model; get_model(model_id='arduino-lcxdx/3', api_key=os.environ['ROBOFLOW_API_KEY'])"
    d=$(ls -d /tmp/cache/models-cache/v2-arduino-lcxdx-3-*/*/ | head -1)
    cp -L "$d/weights.onnx" weights/components_yolov8.onnx
    cp -L "$d/class_names.txt" weights/components_yolov8.classes.txt

With `DETECT=0`, or if the files are missing, the relay logs the reason in `/health` and
behaves exactly as before.
