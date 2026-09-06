# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DNHacks 2026 project: a hands-free copilot for Ray-Ban Meta glasses that identifies the
electronic part in the wearer's hand and speaks it into their ear. The pieces, and who talks to
whom:

| Path | What | Talks to |
|---|---|---|
| `clients/ios/GlassesInspector` | iPhone app (fork of Meta's DAT `CameraAccess` sample). Pulls the glasses camera stream, relays JPEGs to the Mac, hears the wearer's questions through the glasses mic and sends them as text, plays the Mac's speech into the glasses. | `services/relay_receiver` |
| `services/relay_receiver` | FastAPI receiver on the Mac (`:8787`): `app.py` (sockets, dashboard, Describe/Scene narration, voice questions, speech queue), `identify.py` (catalog-bound part identification, reactive loop, Q&A context), `detect.py` (local YOLOv8 ONNX in front of Claude), `tts_elevenlabs.py` (voice rendered on the Mac). | the iOS app, browsers, `docs/components/components.json`, `weights/` |
| `docs/components` | The part catalog: 47 researched records (`records/*.json`) merged into `components.json`, plus query/merge scripts and a small RAG eval. The relay loads `components.json` at startup. | `services/relay_receiver` |
| `services/vision_api` | Earlier, tested FastAPI scaffold: sessions, versioned workflow packages, bounded latest-frame queue, pluggable `VisionEngine`. Not wired to the phone app. | nothing yet |
| `docs/datacenter` | Curated vendor-doc links for data center hardware (future vector DB). Not wired to anything. | nothing yet |
| `firmware/context_node` | Arduino Uno sketch: a capacitive touch/proximity context signal over serial JSON. Not wired to anything; never a voltage sensor. | nothing yet |
| `tools/components` | Fine-tune workflow for the component detector: Grounding DINO proposals, Claude verification of every box, YOLOv8 training, an eval gate. Needs the gitignored `.venv-inf` (inference, GDINO) and the root `.venv`. Steps are in the relay README. | `weights/` |
| `weights/` | The committed detectors: `components_v3.onnx` (default; single class, 960 px, letterbox, with a `.json` sidecar) and the older 14-class `components_yolov8.onnx`, each with a `.classes.txt`. Everything else there is gitignored. | `detect.py` |

Docs that matter as much as the code:

- `docs/meta-glasses-field-notes.md` is the running log: hardware/firmware facts, SDK
  constraints, measurements, root causes of past outages, and the backlog. Read it before
  touching the iOS or relay code, and append dated entries there when something changes.
- `docs/TEAM_PLAN.md` splits the demo into three tracks (phone, Mac brain, demo/story) and pins
  the phone<->Mac interfaces. Change an interface only with the other side's owner in the loop,
  and record the change there. The protocol section below is the current superset of it.
- `services/relay_receiver/README.md` documents the hands-free modes, the three models, the
  voice path, and the detector in depth, including measured timings.
- `docs/components/AGENT_BRIEF.md` and `spec_extraction_prompt.md` define the record schema and
  the research rules (every number sourced, never guess a pinout, plain English, no em dashes).
- `docs/pipeline.html` is the source of the architecture diagram in the README (made with the
  `diagram-design` skill, default skin). Edit the HTML and re-export `docs/pipeline.svg` from
  it; never hand-edit the SVG.

Hardware fact that shapes everything: the glasses are Ray-Ban Meta (camera + BT headset,
**no display**), so Meta's Web Apps / display path does not apply despite the
`meta-glasses-display-access` branch name. All feedback to the wearer is audio.

Things you may see that are not part of this branch: `clients/ios/StockSample/` (gitignored)
is Meta's untouched sample kept only for A/B lag comparisons against the fork. The remote
`feat/gabe-*` and `feat/live-webcam-demo` branches carry a separate PPE-detection server
(`app.py`, `server.py`, `core/` at the repo root) that speaks the same `/ws/ingest` wire
format; they diverge from `main` (vision_api tests removed), so do not merge them blindly. The
PPE model is deliberately never loaded in the relay (it fires on hands and faces).

## Commands

### relay_receiver (separate venv, separate requirements, no tests)

```bash
services/relay_receiver/run.sh            # creates .venv, installs requirements.txt, sources .env, starts :8787
INSPECT_FAKE=1 services/relay_receiver/run.sh   # canned Claude output for every path, no API key
SPEAK=1 services/relay_receiver/run.sh          # also `say` Describe results on the Mac
DETECT=0 services/relay_receiver/run.sh         # skip the local detector (settle-rule triggers only)
cd services/relay_receiver && INSPECT_FAKE=1 ADVERTISE=0 .venv/bin/uvicorn app:app --port 8790 --ws websockets   # second instance for tests, no Bonjour
```

A second instance still appends to the shared `report.jsonl` and `frames/`; scrub its entries
afterwards or the judges' report page shows them.

Secrets and knobs go in `services/relay_receiver/.env` (gitignored): `ANTHROPIC_API_KEY`,
`ELEVENLABS_API_KEY` (+ `ELEVENLABS_VOICE_ID`, `_MODEL`, `_GAIN`, `_SPEED`, `_STABILITY`, `_STYLE`),
`INSPECT_MODEL` (Describe, default `claude-opus-5`), `NARRATE_MODEL` (Scene, default
`claude-sonnet-5`), `IDENTIFY_MODEL` (Parts, default `claude-sonnet-5`), `ASK_MODEL` (voice
questions, default `claude-sonnet-5`), `IDENTIFY_MAX_SIDE` (512),
`DETECT_CONF` (0.45), `DETECT_IOU` (0.5), `DETECT_ONNX` (a sibling `.json` sidecar sets
`resize_mode`), `DETECT_CLASSES`. Keep the
`--ws websockets --ws-ping-timeout 90` flags in `run.sh`; the phone socket stalls without them.
Ask before restarting it on the demo Mac; it holds the phone's socket.

Everything else is switchable at runtime from the dashboard (http://localhost:8787), the phone's
gear menu, or curl:

```bash
curl -s localhost:8787/health | python3 -m json.tool          # models, detector, tts, triggers, agreement
curl -s localhost:8787/debug/tasks                             # detector / reactive / narration / speech loops alive?
curl -X POST localhost:8787/reactive -H 'content-type: application/json' -d '{"enabled":true,"mode":"parts","preannounce":true}'
curl -X POST localhost:8787/models   -H 'content-type: application/json' -d '{"identify":"opus","scene":"sonnet"}'
curl -X POST localhost:8787/voice    -H 'content-type: application/json' -d '{"provider":"apple"}'
curl -X POST localhost:8787/identify                           # one-shot identification of the latest frame, no speech
curl -X POST localhost:8787/ask -H 'content-type: application/json' -d '{"text":"how many pins does this have"}'   # same path as a spoken question
curl -X POST localhost:8787/catalog/reload                     # after editing docs/components/components.json
curl -X POST localhost:8787/frame --data-binary @photo.jpg     # laptop/webcam demo mode: any JPEG is a frame
.venv/bin/python services/relay_receiver/webcam_feed.py --rotate 90   # Mac camera into the relay as if it were the phone (root .venv, needs OpenCV); --save sessions/<name> records training frames
```

### Component catalog (stdlib unless noted)

```bash
python3 docs/components/query.py --search relay              # substring over all fields; --id hc-sr04 --json for one record
python3 docs/components/merge_records.py                      # records/*.json -> components.json + REPORT.md (validates pins, sources, qty)
cd docs/components/rag && python3 search.py --mode bm25 "blue cube five legs"   # bm25 is stdlib; embed/hybrid need torch+transformers
cd docs/components/rag && python3 build_index.py              # re-embed after the catalog changes (MiniLM, local)
python3 docs/datacenter/query_links.py --search "hot swap"    # the data center link set
```

Edit `records/<id>.json`, never `components.json` by hand; merge regenerates it. Per
`rag/EVAL_RESULTS.md`, BM25 beats embeddings for camera-style descriptions, which is why the relay
gives Claude the whole catalog index in a cached prompt instead of retrieving.

### vision_api (root pyproject, root `.venv`)

```bash
cp .env.example .env                      # Settings reads .env from the CWD
python3 -m venv .venv
.venv/bin/pip install fastapi httpx pydantic-settings 'uvicorn[standard]' pytest pytest-asyncio
PYTHONPATH=services uvicorn vision_api.main:app --reload
docker compose up --build                 # same service, needs .env
```

`pip install -e '.[dev]'` (what the README says) currently fails: setuptools sees both
`clients/` and `services/` at the root and refuses flat-layout auto-discovery. Install the
dependencies directly as above, or add a `[tool.setuptools.packages.find] where = ["services"]`
table to `pyproject.toml` if you want the editable install back. The Dockerfile is unaffected
because it copies only `pyproject.toml` before `pip install .`.

Tests are `unittest`-style (`IsolatedAsyncioTestCase`); pyproject configures pytest with
`pythonpath = ["services"]`, so both runners work (7 tests, all passing):

```bash
pytest                                    # all
pytest tests/test_pipeline.py -k failure  # one test by keyword
PYTHONPATH=services python3 -m unittest tests.test_vlm -v   # one module
```

Use the root `.venv` for these, not `services/relay_receiver/.venv`; the relay venv lacks
`httpx` and `pydantic-settings`, so every test module fails to import there.

### iOS app

Needs full Xcode (26.4+; 26.6 is installed), not Command Line Tools. Compile check from the CLI:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -project clients/ios/GlassesInspector/CameraAccess.xcodeproj -scheme CameraAccess \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
```

Running on the phone is done from Xcode (free personal team is enough; see entitlements note
below). Scheme and target are still named `CameraAccess`; display name is "Glasses Inspector",
bundle id `com.eddiehu.glassesinspector.CameraAccess`, URL scheme `glassesinspector://`.
The DAT SDK is an SPM dependency pinned in `Package.resolved` (0.9.0). Meta ships Claude Code
plugins for it (`claude plugin marketplace add facebook/meta-wearables-dat-ios`) and an MCP
endpoint at https://mcp.developer.meta.com/wearables.

### Firmware

```bash
cd firmware/context_node && pio run                          # PlatformIO, env `uno` (`uno-legacy` for 57600 upload)
pio run --target upload --upload-port /dev/cu.usbmodemXXXX && pio device monitor --baud 115200
```

`TOUCH_SENSOR_DEMO.md` records the circuit, readings, and the current blocker (the Uno on hand
will not accept uploads).

## Architecture

### Phone <-> Mac wire protocol (FrameRelay.swift <-> relay_receiver/app.py)

Changing either side means changing both. Everything rides one WebSocket from the phone to
`/ws/ingest` (`/` is an alias because Bonjour service endpoints carry no path).

- Phone -> Mac binary: 8-byte big-endian capture timestamp (ms since epoch) + JPEG bytes. The
  server treats a message starting with `FF D8` as a bare JPEG with no timestamp.
- Phone -> Mac text (JSON) commands, each handled in its own task so ingest never blocks:
  `{"type":"inspect","question"?}`, `{"type":"narrate","enabled","interval"}`,
  `{"type":"reactive","enabled","mode":"parts"|"scene","preannounce"}`,
  `{"type":"models","identify","scene","ask"?}` (values `sonnet`/`opus` or full model ids),
  `{"type":"voice","provider":"apple"|"elevenlabs"}`, `{"type":"ask","text"}` (a sentence the
  wearer said, recognized on the phone), `{"type":"hush"}` (stop talking, drop queued sentences),
  `{"type":"detector","enabled"}` (runtime switch for the local YOLO detector).
- Mac -> Phone speech: `{"type":"speak","text"}` per finished sentence, then `{"type":"speak_end"}`.
  With ElevenLabs active the `speak` carries `"id"` and `"audio":true` (caption only, no local
  synthesis) and is followed by `{"type":"audio","id","rate":24000,"pcm":<base64 s16 mono>}` chunks
  of about 200 ms and `{"type":"audio_end","id"}`. `{"type":"speak_fallback","id","text"}` means
  ElevenLabs failed before any audio played, so the phone reads it with Apple's voice.
  `{"type":"speak_stop"}` cuts playback and clears the caption.
- Mac -> Phone status: `{"type":"narration","enabled","interval"}` and `{"type":"reactive",...}`
  (the full `_reactive_public()` dict: mode, status, voice, models, catalog size, triggers,
  agreement, preannounce) on connect and after every change. The first `reactive` after a
  connect is the Mac's hello; the phone answers by re-sending its saved voice, models, and
  reactive preferences. So the phone's settings win over anything set on the dashboard whenever
  it reconnects, and the Mac holds no preferences across restarts. `{"type":"detected",...}` is
  also sent to phones but ignored by the app today.
- Mac -> Browser (`/ws/view`): binary frames from a latest-only queue (maxsize 1), plus
  `{"type":"stats",...}` (the `/health` dict) every 0.5 s, `{"type":"caption","text","final"}` while
  Claude streams, `{"type":"detections","ts","ms","boxes"}` per detected frame (normalized
  x1,y1,x2,y2), `{"type":"detected",...}` the instant a box qualifies as a trigger, and
  `{"type":"identified",...}` with Claude's parsed answer and the catalog record, and
  `{"type":"heard","text"}` when a voice question arrives.
- Mac HTTP: `POST /inspect {question?}`, `POST /ask {text}`, `POST`/`GET /narrate`, `POST`/`GET /reactive`,
  `POST /models`, `POST /voice`, `POST /detector {enabled}`, `POST /identify`, `GET /detections`, `POST /catalog/reload`,
  `GET /report`, `GET /health`, `GET /debug/tasks`, `GET /latest.jpg`, `GET /frames/{name}`,
  dashboard on `/`. `POST /frame` (raw JPEG body, optional `x-capture-ts` ms header) is the ingest
  fallback and the laptop-webcam demo mode.
- Discovery: the Mac advertises `_glassesrelay._tcp` via zeroconf with TXT `urls` (comma list of
  `http://ip:8787`, refreshed every 10 s) and `server="<host>.local."`. The SRV target must be a
  plain hostname; iOS refuses to resolve the zeroconf default and the connection sits in
  `.preparing` forever.

### iOS additions on top of Meta's sample

All custom code is marked "Glasses Inspector" (file headers or `// MARK:` comments) and lives
in two files plus small hooks:

- `CameraAccess/Media/FrameRelay.swift`: `RelaySocket` (actor; `NWConnection` +
  `NWProtocolWebSocket`; tries the USB link-local `169.254.x` URL on the wired interface first
  with a 3 s budget, then falls back to LAN IP / service name, retries cable after 60 s),
  `RelayBrowser` (`NWBrowser` for Bonjour, `includePeerToPeer=false`), `RelayEngine` (actor;
  throttle to `targetFPS`, JPEG encode, single latest-frame slot, up to `maxInFlight` sends),
  `MainThreadMeter`, and `FrameRelay`, the `@Observable @MainActor` facade the views bind to.
  `FrameRelay.shared` is a singleton with a private init: the sample creates `CameraViewModel`
  more than once per run (`FrameRelay.viewModelsCreated` counts them), so anything with a
  lifetime (socket, browser, speaker, stats task) hangs off the shared instance, never the view
  model. Settings persist in `UserDefaults` (`relay*` keys for transport, plus `handsFreeMode`
  off/parts/scene, `voiceProvider`, `partsModel`, `sceneModel`, `preannounce`); each setter
  sends the matching command, and `announcePrefs()` replays them on the Mac's hello. Server
  text messages are parsed in `handleServerText`.
- `CameraAccess/Media/VoiceInput.swift`: the wearer's voice. The DAT SDK has no audio API, so
  the glasses mic is reached through the phone's audio session (`.playAndRecord` +
  `.allowBluetoothHFP`, the same route Meta's sample records sound-in-video through) or the
  phone's own mic (`.allowBluetoothA2DP`, keeps high-quality output). An `SFSpeechRecognizer`
  (on-device when available) streams partials; an utterance ends after 0.9 s without new words
  and a fresh request starts. Modes: off, wake word (default "inspector", alone arms the next
  sentence), always. Stop words are handled on the phone (`FrameRelay.hush()`), everything else
  goes out as `ask`. Utterances that end while the glasses are talking are dropped unless they
  are stop words, and the recognizer context resets when playback ends (echo guard). Settings
  persist under `voiceInputMode`, `voiceMic`, `wakeWord`. Turning it off restores `.playback`.
- `CameraAccess/Media/Speaker.swift`: `AVSpeechSynthesizer` for `speak`/`speak_fallback` and
  `PCMStreamPlayer` (`AVAudioEngine` + `AVAudioPlayerNode`) for ElevenLabs PCM, both on a
  `.playback/.spokenAudio` session so audio routes to the glasses over A2DP. It deliberately
  leaves a `.playAndRecord` session alone when the sample is recording.
- `CameraViewModel` holds `frameRelay = FrameRelay.shared`. Inside the toolkit's
  `videoFramePublisher` callback (off the main actor) it decodes the HEVC frame, calls
  `frameRelay.push(previewImage)`, and drops the image into a single-slot `previewSlot` so at
  most one main-actor hop is in flight. Do not reintroduce a per-frame `Task { @MainActor }`;
  that unbounded backlog is what froze the UI at 720p. Stream resolution and fps are read from
  the `streamResolution` / `streamFPS` `UserDefaults` keys in `beginStream`, so they apply on
  the next Preview (default Low, 24 fps).
- `VideoFrameDecoder` prefers hardware VideoToolbox decode and a GPU `CIContext`; software HEVC
  decode was the dominant CPU cost.
- `CameraView` shows the relay chip, caption overlay, Describe/Hush buttons, and the
  `RelaySettingsView` sheet (hands-free mode, model pickers, voice, pre-announce, test voice).
  `WearablesViewModel.deviceStatusText` is the per-device link/compat line shown there.
- `Info.plist` must keep `UISupportedExternalAccessoryProtocols = [com.meta.ar.wearable]` and
  the `external-accessory` background mode. That is what lets DAT stream over Bluetooth Classic
  on a free Apple team. The Wi-Fi (SoftAP) path needs HotspotConfiguration and wifi-info
  entitlements that a free team cannot sign, so `CameraAccess.entitlements` intentionally
  omits them. Also keep `NSBonjourServices` (`_glassesrelay._tcp`) and `NSAllowsLocalNetworking`.

### relay_receiver internals

Three Claude call sites, three independently switchable models, one prompt each:

| Path | Trigger | Model (default) | Prompt / code |
|---|---|---|---|
| Describe | phone `inspect`, `POST /inspect` | `models["describe"]` (Opus) | `SYSTEM_PROMPT`, `analyze()` in `app.py` |
| Scene | hands-free `scene` mode, or legacy `narrate` loop | `models["scene"]` (Sonnet) | `NARRATION_PROMPT`, `analyze(narrate=True)`; previous narration passed as context, literal `no change` suppressed |
| Parts | hands-free `parts` mode, `POST /identify` | `identify.state["model"]` (Sonnet) | `identify.system_prompt()`: catalog index in a `cache_control` system block |
| Ask | phone `ask` (voice), `POST /ask`, dashboard "Ask (as voice)" | `models["ask"]` (Sonnet) | `ASK_PROMPT`; `handle_ask()` routes stop words -> `hush()`, "what is this" -> `_identify_and_speak()`, "describe" -> Describe path, else `analyze()` with `identify.context_for_question()` (the catalog record of the last identified part) and a 768 px frame |

`hush()` bumps `speech["stop_gen"]`; `analyze()` and `_tts_worker` compare against it and stop
mid-stream, so a "stop" cuts the answer within a sentence instead of draining only the queue. The
phone side drops late PCM chunks for stopped utterance ids (`PCMStreamPlayer.stoppedThrough`).

A voice question outranks narration: `handle_ask()` drops queued speech, waits (bounded) for a
running analysis, and afterwards pushes `identify.state["announce_until"]` past the answer so the
reactive loop does not talk over it. `voice_in` in `/health` counts asks by intent.

`app.py` keeps module-level dicts: `state` (latest JPEG, fps/latency windows, `report` reloaded
from `report.jsonl` at startup), `phones`, `viewers` + `viewer_sockets`, `narration`, `detector`,
`speech`, `voice`, `models`. `analyze()` streams Claude, splits on sentence boundaries, and
`_say()`s each sentence as it lands so speech starts early. The server-side-fallback beta is only
sent to Opus/Fable models (`_supports_fallbacks`); do not add it to Sonnet calls. `narration["busy"]` is the
single guard for every `analyze()` path; identification has its own `_busy` in `identify.py`.

Speech: `_say()` estimates playback seconds (exact for cached ElevenLabs audio) so the reactive
loop knows when the glasses go quiet (`announce_until`). With ElevenLabs active, sentences enter
`speech["queue"]` and one `_tts_worker` streams them in order; `tts_elevenlabs.py` renders
`eleven_flash_v2_5` as raw 24 kHz PCM with gain applied, caches by text hash in `tts_cache/`
(gitignored), and `_warm()` pre-renders every catalog spoken line at startup so announcements
are cache hits. `_drop_pending_speech()` + `speak_stop` is how an interrupt keeps stale audio
from following. The starter plan is 30,000 characters a month; `/health` reports usage.

`identify.py`: `load_catalog()` reads `docs/components/components.json` (falls back to a sibling
`components.json`) and builds one index line per part from `id`, `canonical_name`,
`details.visual_identification` (printed text, shape, color, easily confused with). Claude
answers in exactly two lines, `<catalog id or none> <confidence>` then evidence; `_stream_model`
fires `on_first_line` as soon as line 1 is complete so the glasses hear the name before the
evidence finishes streaming. Frames are cropped to the detector box with 25% padding when there
is one, then downscaled to 512 px. `resolve()` tolerates near-miss ids. `spoken_short()` (name +
pins/voltage) is what the glasses hear by default; `spoken_for()` adds one caution, is spoken
when `short_spoken` is off, and is what the report and dashboard store as `spoken`.
`warm_cache()` sends a tiny call per model at startup so the first real call reads the cached
prompt.

`reactive_loop()` runs at 20 Hz while hands-free is on and has two triggers feeding the same
call: the detector path (only with `det_trigger` on, default off: a `usable_box()`, conf >= 0.6,
at least 2 % of the frame and clear of the edges, persisted `stable_frames`=3 frames with
IoU > 0.4 and it is a new label or the scene changed; identifies the exact frame the boxes came
from) and the settle path (thumbnail diff vs the last identified scene exceeds
`change_threshold`, motion stopped for `settle_seconds`, and the frame is sharp by Laplacian
variance; after a miss the wait doubles, 4, 8, 15 s via `miss_backoff`, so an empty view is not
a Claude call every two seconds). With the trigger off, boxes are a visual only and the
"box trigger" checkbox on the dashboard turns them back into a trigger.
`_interrupt_policy` decides per early id whether to announce, interrupt (only above
`interrupt_confidence`), hold as `pending` until audio ends, or ignore (same id, low
confidence). With `preannounce` on, `on_detected` speaks the detector's catalog display name
before Claude is called, and `agreement` counts whether Claude's id matched the hint. Scene mode
reuses the same settle detection but calls `narrate_scene` instead. The thresholds live in
`identify.state`; `POST /reactive` exposes the main ones (`min_confidence`, `settle_seconds`,
`cooldown_seconds`, `interrupt_confidence`, `det_trigger`, `det_min_conf`, `det_min_area`,
`det_edge_margin`, `stable_frames`, `short_spoken`, `preannounce`, `mode`), the
motion/change/sharpness thresholds only via `identify.set_enabled`.

`detect.py`: a YOLOv8n ONNX on `CPUExecutionProvider` (CoreML is not faster) in `_detect_loop`
via `asyncio.to_thread`, latest frame wins. Default weights are `weights/components_v3.onnx`:
single class `component`, 960 px, fine-tuned on our own webcam frames (Sept 6; about 105 ms
per frame, so the loop follows roughly 8 fps of the stream). Its `.json` sidecar sets
`resize_mode: letterbox`; ultralytics-trained models need letterbox and Roboflow ones stretch,
and the wrong mode silently halves recall. The older 14-class Roboflow model
(`components_yolov8.onnx`, Spanish labels, select with `DETECT_ONNX`) is kept for comparison;
`CATALOG_HINT` and `bind_catalog()` (display name = the record's `name_on_kit`) only matter for
multi-class weights. `state["almost"]` carries the best sub-threshold candidate for tuning. If
weights are missing or inference throws, the detector marks itself unavailable and the relay
falls back to the settle path.
`set_detector()` is the runtime switch (`detector["enabled"]`, driven by the phone's gear menu,
the dashboard checkbox, or `POST /detector`); off clears the boxes and the reactive loop's
stale-box state so the settle rule takes over immediately.
`/debug/tasks` shows whether the loop died.

Every analysis appends to `report.jsonl` and saves the frame under `frames/` (both gitignored;
reactive frames are `identify-<ts>.jpg` and the entry records `trigger`, `det`, `agrees`).
Bonjour registration retries for about 90 s because a just-killed instance's record lingers as
a name conflict.

### Component catalog contract

`components.json` is `{"_meta", "components": [...]}`; each record has flat seed fields (`id`,
`name_on_kit`, `canonical_name`, `mpn`, `category`, `kit`, `qty`, `pin_count`, `voltage`,
`key_specs`, `price_usd`, `status`, `confidence`) and the full researched record under `details`
(`identity`, `function`, `pins`, `electrical`, `visual_identification`, `wiring_to_uno`, `safety`,
`troubleshooting`, `market`, `sources`, `confidence`). The relay depends on `id`, `name_on_kit`,
`canonical_name`, `pin_count`, `voltage`, `details.visual_identification`, `details.safety.hazards`,
and `details.wiring_to_uno.example_connections`; `detect.CATALOG_HINT` depends on the ids. Renaming
any of those means touching `identify._index_line` / `_spoken_line` and `detect.bind_catalog`.
`merge_records.py` refuses records without a datasheet/manufacturer source and flags pin-count
mismatches, so keep new records in that schema rather than patching the merged file.

### vision_api internals

`main.py` builds module-level singletons at import: `VisionPipeline`, `WorkflowRegistry`
(loaded from `workflow_definitions/*.json`, or `WORKFLOW_DEFINITIONS_DIR`), an in-memory
`sessions` dict mapping session id -> workflow id, and `AdvisorySignalStore`.

Request flow: `POST /v1/sessions` binds a workflow -> frames arrive on the WebSocket as
alternating `FrameMetadata` JSON + binary image (or `POST .../frames` with the metadata in an
`x-frame-metadata` header) -> `VisionPipeline.submit` drops the oldest queued frame when full
and returns the per-session dropped count -> one worker task calls `VisionEngine.analyze(Frame)`
-> `SegmentationResult` lands in a per-session bounded deque -> `GET .../results` / `.../metrics`.

`VisionEngine` is a `Protocol`; `build_vision_engine` picks `MockSegmentationEngine` or
`OpenAICompatibleVLM` from `VISION_BACKEND`. `vlm.instruction_for` merges the safety
instruction with the session's workflow instructions and checkpoints. New engines implement
`analyze` and register in `build_vision_engine`.

Design invariant across the whole repo: nothing here authorizes work. Workflows shape model
context only, `ActionProposal.requires_confirmation` defaults to true, advisory field signals
(and the Arduino context node) are observations that can warn but never establish a
deenergized state (see README for the OSHA basis), and the relay's detector stays
component-only. Keep that boundary when adding endpoints, workflow definitions, or models.

## Environment quirks

- `Settings` (pydantic-settings) reads `.env` relative to the working directory, so run
  `uvicorn` and tests from the repo root.
- `INSPECT_FAKE=1` fakes all three Claude paths (identify cycles `hc-sr04`, `rtc-ds3231`,
  `relay-5v`, or follows the detector hint) and skips prompt warming, but the detector and
  ElevenLabs still run for real. Use it to test the audio path and the dashboard without a key.
- The phone re-asserts its saved preferences every time it connects, so a model/voice/mode
  change made on the dashboard is undone by the next phone reconnect unless you also change it
  in the phone's gear menu.
- Deleting `tts_cache/` costs ElevenLabs characters on the next start (every catalog line is
  re-rendered).
- Voice input with the glasses mic puts the whole session on Bluetooth HFP, so ElevenLabs output
  is 16 kHz mono while listening and the SCO link shares Bluetooth Classic with the DAT video
  stream. Meta's own sample records HFP audio while streaming, so it should hold, but fps under
  load and the echo guard are not yet measured on the hardware (see the field notes).
- The Mac's LAN and USB link-local addresses change during the day; always pick the receiver
  from the Bonjour list on the phone rather than typing an IP.
- Do not let the Mac hold the glasses over Bluetooth (headset pairing) while testing the phone
  app; the stream will not start. Another app with the same URL scheme breaks DAT registration.
