# Relay receiver (Mac), reactive edition

The Mac side of Glasses Inspector. Accepts frames from the iOS app over `WS /ws/ingest` (also
`WS /`), fans them out to browsers on `WS /ws/view`, serves a live dashboard on `/`, and answers
the wearer's spoken questions. Nothing analyzes frames on its own: the relay keeps a ring buffer
of recent frames and only calls Claude when a question arrives.

    ./run.sh          # venv + deps, listens on 0.0.0.0:8787, advertises _glassesrelay._tcp via Bonjour

The Bonjour TXT record carries the receiver's current URLs (refreshed every 10 s) so the phone
never needs a typed IP. `--ws-ping-timeout 90` matters: the phone's socket stalls otherwise.
Put `ANTHROPIC_API_KEY=...` in `services/relay_receiver/.env` (gitignored); `INSPECT_FAKE=1 ./run.sh`
answers with canned text to test the audio path without a key.

## A question, end to end

1. The phone recognizes "hey inspector, how many pins does the part on my left have" (glasses mic
   over Bluetooth HFP, Apple's on-device recognizer) and sends `{"type":"ask","text":...,"heard_at":ms}`,
   where `heard_at` is when the wearer started speaking. `POST /ask {"text","heard_at"?}` and the
   dashboard's Ask button take the same path; `POST /inspect` is an ask with a default question
   ("What am I looking at? Name the parts you can see."), which is what the phone's What's here?
   button sends.
2. `select_frames()` picks the 3 sharpest frames (`FRAMES_PER_ASK`) spread over the window from
   `heard_at` to now, one per time slice by Laplacian variance, oldest first. The wearer was
   looking at the thing while asking, so that window is where the part is.
3. `_closeup()` runs the local detector (`CROP_CONF`, default 0.25) over those frames, sharpest
   first, and crops a padded close-up around the first box it finds. A part held at arm's length
   is about a hundred pixels wide in the full frame; the crop is what makes small parts match.
4. Claude Sonnet 5 gets the frames, the crop, the previous exchange as text (`MEMORY_SECONDS`,
   2 min, so "does it need a driver" resolves "it" and names are not repeated), the question, and
   the whole part catalog in a cached system prompt: `TEACH_PROMPT` (an apprentice's senior
   colleague; one or two sentences, four for a pinout or a comparison; the catalog is the ground
   truth; position words are image positions) + a quick index (one line per part with what it
   looks like) + every full record, about 58k tokens. `_warm()` re-reads the cache every 4 min so
   it never lapses. Thinking is off (`THINKING=off`): Sonnet 5 thinks by default, and a thinking
   block once ate the whole output budget so no text came out; `THINKING=adaptive EFFORT=low`
   measured no better on grounding.
5. Sentences are spoken as they stream (`speak` for the Apple voice on the phone, or ElevenLabs
   PCM rendered here). A stop word ("stop", "hush", "quiet", three words or fewer) or
   `{"type":"hush"}` bumps a stop generation that cuts the stream and the sentence being
   rendered. Every question lands in `report.jsonl` with its frames (`frames/ask-<ts>-<i>.jpg`
   plus `-crop.jpg`), first-word and total milliseconds; `/report.html` (behind the phone's
   report-page toggle) shows them.

Measured on the real API with 504x896 frames: first word 1.0 to 1.5 s, whole answer 2.5 to
3.5 s. Grounding depends on the frames and on the record: the same three questions about a
28BYJ-48 at arm's length came back right in about seven of nine runs, and the misses cited the
record's own visual description (it said white plug and white cap; the kit's motor has a blue
connector housing). Fix `visual_identification` in `docs/components/records/<id>.json`, run
`docs/components/merge_records.py`, then `POST /catalog/reload`. `GET /catalog/prompt` shows the
exact text Claude gets.

## Controls and endpoints

| What | Phone gear menu | Dashboard | HTTP |
|---|---|---|---|
| Ask | say it, or What's here? | Ask, What's here?, Hush | `POST /ask`, `POST /inspect` |
| Voice: Apple on the phone or ElevenLabs on the Mac | Voice | Voice select | `POST /voice {"provider"}` |
| Detector boxes on the dashboard (visual only, off by default) | Inspector > Detector boxes | detector boxes | `POST /detector {"enabled"}` |
| Judges' report page | Report | link appears | `POST /report_page {"enabled"}`, `GET /report.html` |
| Status | | HUD | `GET /health`, `GET /status`, `GET /debug/tasks`, `GET /detections`, `GET /report` |

The phone re-sends its saved voice, detector and report-page settings on every reconnect, so a
change made on the dashboard is undone by the next phone reconnect unless it is also changed in
the phone's gear menu. Env knobs: `MODEL` (`claude-sonnet-5`), `THINKING`, `EFFORT`,
`FRAMES_PER_ASK` (3), `FRAME_MAX_SIDE` (1280), `CROP_CONF` (0.25), `STALE_SECONDS` (8, an older newest
frame gets "no recent picture" instead of an answer), `DETECT_DEFAULT` (0),
`ADVERTISE=0` for a second test instance on another port, `SPEAK=1` to also `say` on the Mac.

## Voice: ElevenLabs on the Mac, Apple voice as fallback

With `ELEVENLABS_API_KEY` in `services/relay_receiver/.env` and the voice switched to ElevenLabs,
the relay renders each finished sentence with ElevenLabs (`eleven_flash_v2_5`, raw 24 kHz PCM)
and streams it to the phone as `{"type":"audio","id":n,"rate":24000,"pcm":"<base64>"}` chunks of
about 200 ms, ending with `{"type":"audio_end","id":n}`. The matching `speak` message carries
`"audio": true` so the phone shows the caption but does not synthesize. If ElevenLabs fails
before any audio is sent, the relay sends `{"type":"speak_fallback","text":...}` and the phone
reads it with Apple's voice. Without the key, behaviour is unchanged: plain `speak` messages,
Apple voice on the phone (the default).

    ELEVENLABS_API_KEY=sk_...
    ELEVENLABS_VOICE_ID=mqlDiDxS84MhnMijtd3t    # Christopher, Friendly American
    ELEVENLABS_MODEL=eleven_flash_v2_5          # optional
    ELEVENLABS_GAIN=2.0                         # louder for the glasses; 1.0 = as rendered, clips above ~2.4
    ELEVENLABS_SPEED=1.1                        # 0.7 to 1.2
    ELEVENLABS_STABILITY=0.5                    # lower = more expressive
    ELEVENLABS_STYLE=0.0                        # style exaggeration, adds latency above 0

`GET /health` reports `tts.provider`, characters sent, and the last error. The starter plan
caps at 30,000 characters a month, roughly 300 sentences. Rendered sentences are cached in
`tts_cache/` (gitignored); deleting it costs characters. While the glasses mic is active the
phone's audio session is Bluetooth HFP in both directions, so this voice sounds narrower than
over A2DP; the phone's "Microphone: Phone" setting keeps A2DP output.

## The local component detector

`detect.py` runs a YOLOv8n ONNX with plain onnxruntime. Default weights are
`weights/components_v3.onnx`: a single class "component", 960 px, fine-tuned on our own webcam
footage (see the fine-tune workflow below), about 105 ms per frame on the Mac CPU. The `.json`
sidecar next to it sets `resize_mode: letterbox`; ultralytics-trained models need letterbox and
Roboflow-trained ones stretch, and the wrong mode silently halves recall. The older 14-class
Roboflow model (`components_yolov8.onnx`, Spanish labels) is kept for comparison; select it with
`DETECT_ONNX`.

In this edition the detector never triggers Claude. It has two jobs: the close-up crop at
question time (always, regardless of the toggle), and boxes on the dashboard video for the
audience when "detector boxes" is on (`POST /detector`, off by default because it costs a tenth
of a second of CPU per frame). `GET /detections` returns the latest boxes and the best
sub-threshold candidate ("almost"), useful for tuning.

Env: `DETECT=0` disables the detector entirely, `DETECT_CONF` (0.45) and `DETECT_IOU` (0.5) set
the raw NMS thresholds for the dashboard boxes, `DETECT_CLASSES` restricts classes for
multi-class weights.

## Fine-tuning the component detector on your own footage (tools/components/)

The Universe model does not transfer to webcam or glasses footage (it boxes furniture and shirts,
misses small parts in hand). The fix is a single-class "component" detector trained on our own
frames.

1. Capture: run the capture page from the PPE branch (`git worktree add .worktrees/feat/gabe-dev feat/gabe-dev`,
   link `weights/` and `sessions/`, `uvicorn server:app --port 8000`), record 5 minutes of parts in hand
   plus empty hands, upload with keep-every 3. Or `webcam_feed.py --save sessions/<name>`.
2. Propose boxes: `.venv-inf/bin/python tools/components/gdino_label.py --session sessions/capture_X --every 3 --out datasets/<name>`
   (Grounding DINO, saves raw witnesses down to 0.2).
3. Verify with Claude: `services/relay_receiver/.venv/bin/python tools/components/claude_verify.py --raw datasets/<name>/raw_detections.json --out datasets/<name>_verified`
   Sonnet keeps or rejects each numbered box per frame and flags missed parts (frames dropped). Audit `preview/`.
   Then drop negatives adjacent to positives (a part in hand persists), see the field notes.
4. Train: `.venv/bin/python tools/components/train.py --own datasets/<name>_verified --universe datasets/universe_1class --name <run> --epochs 25`
   (YOLOv8n, MPS, exports `weights/<run>.onnx` + `.classes.txt`).
5. Gate: `services/relay_receiver/.venv/bin/python tools/components/eval_gate.py --onnx weights/<run>.onnx --data datasets/<name>_verified`
   passes at recall >= 0.8 and <= 0.05 false positives per negative frame at the trigger threshold.
6. Swap: `DETECT_ONNX=weights/<run>.onnx ./run.sh`.

## Testing without the glasses

`webcam_feed.py` (root `.venv`, needs OpenCV) feeds the Mac camera into the relay as if it were
the phone (`--rotate 90` for portrait frames). For the wire protocol, a second instance with
`INSPECT_FAKE=1 ADVERTISE=0 .venv/bin/uvicorn app:app --port 8790 --ws websockets` does not
fight the live one over Bonjour; it still appends to the shared `report.jsonl` and `frames/`,
so scrub its entries afterwards.
