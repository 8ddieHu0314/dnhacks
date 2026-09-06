# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DNHacks 2026 project: a voice-first trainer for Ray-Ban Meta glasses. An apprentice on a workshop
bench says "hey inspector, how many pins does the part on my left have?", the Mac answers from
the frames of that moment and the workshop's part catalog, and the answer is spoken in the
glasses. This branch (`feat/reactive-voice`) is the reactive edition: nothing analyzes frames on
its own. The proactive edition (frame differencing, detector-triggered identification, scene
narration) lives on `main`.

| Path | What | Talks to |
|---|---|---|
| `clients/ios/GlassesInspector` | iPhone app (fork of Meta's DAT `CameraAccess` sample). Pulls the glasses camera stream, relays JPEGs to the Mac, hears the wearer through the glasses mic and sends sentences as text, plays the Mac's speech into the glasses. | `services/relay_receiver` |
| `services/relay_receiver` | FastAPI receiver on the Mac (`:8787`): `app.py` (sockets, dashboard, the question path, speech queue), `catalog.py` (the part catalog rendered as Claude's cached ground-truth prompt), `detect.py` (local YOLOv8 ONNX: dashboard boxes and a close-up crop at question time), `tts_elevenlabs.py` (voice rendered on the Mac). | the iOS app, browsers, `docs/components/components.json`, `weights/` |
| `docs/components` | The part catalog: 47 researched records (`records/*.json`) merged into `components.json`, plus query/merge scripts and a small RAG eval. The relay loads `components.json` at startup and puts all of it in the prompt. | `services/relay_receiver` |
| `tools/components` | Fine-tune workflow for the component detector: Grounding DINO proposals, Claude verification of every box, YOLOv8 training, an eval gate. Needs the gitignored `.venv-inf` (inference, GDINO) and the root `.venv`. Steps are in the relay README. | `weights/` |
| `weights/` | The committed detectors: `components_v3.onnx` (default; single class, 960 px, letterbox, with a `.json` sidecar) and the older 14-class `components_yolov8.onnx`, each with a `.classes.txt`. Everything else there is gitignored. | `detect.py` |
| `services/vision_api` | Earlier, tested FastAPI scaffold: sessions, versioned workflow packages, bounded latest-frame queue, pluggable `VisionEngine`. Not wired to the phone app. | nothing yet |
| `docs/datacenter` | Curated vendor-doc links for data center hardware (future vector DB). Not wired to anything. | nothing yet |
| `firmware/context_node` | Arduino Uno sketch: a capacitive touch/proximity context signal over serial JSON. Not wired to anything; never a voltage sensor. | nothing yet |

Docs that matter as much as the code:

- `docs/meta-glasses-field-notes.md` is the running log: hardware/firmware facts, SDK
  constraints, measurements, root causes of past outages, and the backlog. Read it before
  touching the iOS or relay code, and append dated entries there when something changes.
- `docs/TEAM_PLAN.md` splits the demo into three tracks (phone, Mac brain, demo/story) and pins
  the phone<->Mac interfaces. Change an interface only with the other side's owner in the loop,
  and record the change there. The protocol section below is the current superset of it.
- `services/relay_receiver/README.md` documents the question path, the voice path, and the
  detector in depth, including measured timings.
- `docs/components/AGENT_BRIEF.md` and `spec_extraction_prompt.md` define the record schema and
  the research rules (every number sourced, never guess a pinout, plain English, no em dashes).
- `docs/pipeline.html` is the source of the architecture diagram in the README (made with the
  `diagram-design` skill, default skin). Edit the HTML and re-export `docs/pipeline.svg` from
  it; never hand-edit the SVG.

Hardware fact that shapes everything: the glasses are Ray-Ban Meta (camera + BT headset,
**no display**), so Meta's Web Apps / display path does not apply despite the
`meta-glasses-display-access` branch name. All feedback to the wearer is audio, and the only
way to the glasses microphone is the phone's Bluetooth HFP audio session (the DAT SDK has no
audio API).

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
INSPECT_FAKE=1 services/relay_receiver/run.sh   # canned answers, no API key
SPEAK=1 services/relay_receiver/run.sh          # also `say` answers on the Mac
cd services/relay_receiver && INSPECT_FAKE=1 ADVERTISE=0 .venv/bin/uvicorn app:app --port 8790 --ws websockets   # second instance for tests, no Bonjour
```

A second instance still appends to the shared `report.jsonl` and `frames/`; scrub its entries
afterwards or the judges' report page shows them.

Secrets and knobs go in `services/relay_receiver/.env` (gitignored): `ANTHROPIC_API_KEY`,
`ELEVENLABS_API_KEY` (+ `ELEVENLABS_VOICE_ID`, `_MODEL`, `_GAIN`, `_SPEED`, `_STABILITY`, `_STYLE`),
`MODEL` (default `claude-sonnet-5`, one model for everything), `THINKING` (`off`, the default,
or `adaptive`) and `EFFORT` (`low` when adaptive), `FRAMES_PER_ASK` (3), `FRAME_MAX_SIDE` (1280),
`CROP_CONF` (0.25, detector threshold for the close-up), `STALE_SECONDS` (8; an older newest frame gets
"no recent picture" instead of an answer), `DETECT_DEFAULT` (0; dashboard boxes off),
`DETECT_ONNX` (a sibling `.json` sidecar sets `resize_mode`), `DETECT_CONF`, `DETECT_IOU`,
`DETECT_CLASSES`. Keep the `--ws websockets --ws-ping-timeout 90` flags in `run.sh`; the phone
socket stalls without them. Ask before restarting it on the demo Mac; it holds the phone's socket.

Useful from curl (the dashboard at http://localhost:8787 has buttons for the same):

```bash
curl -s localhost:8787/health | python3 -m json.tool          # fps, buffer, model, thinking, detector, tts, voice_in
curl -s localhost:8787/status                                  # what the phone gets as its hello
curl -X POST localhost:8787/ask -H 'content-type: application/json' -d '{"text":"how many pins does the part on the left have"}'
curl -X POST localhost:8787/ask -H 'content-type: application/json' -d '{"text":"stop"}'          # hush
curl -X POST localhost:8787/inspect                             # "What am I looking at? Name the parts you can see."
curl -X POST localhost:8787/detector -H 'content-type: application/json' -d '{"enabled":true}'    # boxes on the dashboard
curl -s localhost:8787/catalog/prompt | head -60               # the exact ground-truth text Claude gets
curl -X POST localhost:8787/catalog/reload                     # after editing docs/components/components.json
curl -X POST localhost:8787/frame --data-binary @photo.jpg     # laptop/webcam demo mode: any JPEG is a frame
.venv/bin/python services/relay_receiver/webcam_feed.py --rotate 90   # Mac camera into the relay as if it were the phone (root .venv, needs OpenCV)
```

### Component catalog (stdlib unless noted)

```bash
python3 docs/components/query.py --search relay              # substring over all fields; --id hc-sr04 --json for one record
python3 docs/components/merge_records.py                      # records/*.json -> components.json + REPORT.md (validates pins, sources, qty)
cd docs/components/rag && python3 search.py --mode bm25 "blue cube five legs"   # bm25 is stdlib; embed/hybrid need torch+transformers
python3 docs/datacenter/query_links.py --search "hot swap"    # the data center link set
```

Edit `records/<id>.json`, never `components.json` by hand; merge regenerates it, then
`POST /catalog/reload` or restart the relay. The relay does no retrieval: the whole catalog is in
Claude's cached prompt (about 58k tokens), so a record's `visual_identification` text is what
decides whether a part on the bench gets matched. When the model keeps calling a kit part "not in
the catalog", fix that record's description of this kit's variant (the stepper's blue connector
housing was such a case) before touching the prompt.

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
endpoint at https://mcp.developer.meta.com/wearables. New Swift files must be registered in
`project.pbxproj` (four entries, copy the `Speaker.swift` pattern); the groups are not
filesystem-synchronized.

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
  `{"type":"ask","text","heard_at"?}` (a sentence the wearer said, recognized on the phone;
  `heard_at` is the epoch ms when they started speaking and anchors the frame window),
  `{"type":"hush"}` (stop talking, drop queued sentences), `{"type":"inspect","question"?}`
  (the What's here? button: an ask with a default question), `{"type":"detector","enabled"}`
  (dashboard boxes), `{"type":"report_page","enabled"}`, `{"type":"voice","provider":"apple"|"elevenlabs"}`.
- Mac -> Phone speech: `{"type":"speak","text"}` per finished sentence, then `{"type":"speak_end"}`.
  With ElevenLabs active the `speak` carries `"id"` and `"audio":true` (caption only, no local
  synthesis) and is followed by `{"type":"audio","id","rate":24000,"pcm":<base64 s16 mono>}` chunks
  of about 200 ms and `{"type":"audio_end","id"}`. `{"type":"speak_fallback","id","text"}` means
  ElevenLabs failed before any audio played, so the phone reads it with Apple's voice.
  `{"type":"speak_stop"}` cuts playback.
- Mac -> Phone status: `{"type":"status","voice","detector_enabled","detector","report_page",
  "model","thinking","effort","catalog_parts","busy","prompt_version"}` on connect and after
  every change. The first `status` after a connect is the Mac's hello; the phone answers by
  re-sending its saved voice, report-page and detector preferences. So the phone's settings win
  over anything set on the dashboard whenever it reconnects, and the Mac holds no preferences
  across restarts.
- Mac -> Browser (`/ws/view`): binary frames from a latest-only queue (maxsize 1), plus
  `{"type":"stats",...}` (the `/health` dict) every 0.5 s, `{"type":"heard","text"}` when a
  question arrives, `{"type":"caption","text","final"}` while Claude streams (the first caption
  is `You: <question>`), `{"type":"detections","ts","ms","boxes","almost"}` per detected frame
  when the detector is on (normalized x1,y1,x2,y2), `{"type":"report_page","enabled"}`.
- Mac HTTP: `POST /ask {text, heard_at?}`, `POST /inspect {question?}`, `POST /detector {enabled}`,
  `POST /voice {provider}`, `POST /report_page {enabled}`, `POST /catalog/reload`,
  `GET /catalog/prompt`, `GET /status`, `GET /health`, `GET /debug/tasks`, `GET /detections`,
  `GET /report`, `GET /report.html`, `GET /latest.jpg`, `GET /frames/{name}`, dashboard on `/`.
  `POST /frame` (raw JPEG body, optional `x-capture-ts` ms header) is the ingest fallback and the
  laptop-webcam demo mode.
- Discovery: the Mac advertises `_glassesrelay._tcp` via zeroconf with TXT `urls` (comma list of
  `http://ip:8787`, refreshed every 10 s) and `server="<host>.local."`. The SRV target must be a
  plain hostname; iOS refuses to resolve the zeroconf default and the connection sits in
  `.preparing` forever.

### iOS additions on top of Meta's sample

All custom code is marked "Glasses Inspector" (file headers or `// MARK:` comments) and lives
in three files plus small hooks:

- `CameraAccess/Media/FrameRelay.swift`: `RelaySocket` (actor; `NWConnection` +
  `NWProtocolWebSocket`; tries the USB link-local `169.254.x` URL on the wired interface first
  with a 3 s budget, then falls back to LAN IP / service name, retries cable after 60 s; opening
  is serialized and a new connection cancels the old one, because two live sockets once made the
  Mac speak every sentence twice), `RelayBrowser` (`NWBrowser` for Bonjour,
  `includePeerToPeer=false`), `RelayEngine` (actor; throttle to `targetFPS`, JPEG encode, single
  latest-frame slot, up to `maxInFlight` sends), `MainThreadMeter`, and `FrameRelay`, the
  `@Observable @MainActor` facade the views bind to. `FrameRelay.shared` is a singleton with a
  private init: the sample creates `CameraViewModel` more than once per run
  (`FrameRelay.viewModelsCreated` counts them), so anything with a lifetime (socket, browser,
  speaker, voice input, stats task) hangs off the shared instance, never the view model.
  Settings persist in `UserDefaults` (`relay*` keys for transport, `relaySpeak`, `voiceProvider`,
  `detectorEnabledV2`, `reportPage`, `voiceInputMode`, `voiceMic`, `wakeWord`); a one-time
  migration keyed on `relayDefaultsVersion` moves older installs to High resolution, 15 fps from
  the glasses, a 5 fps relay cap and JPEG quality 0.8, because frames now feed questions rather
  than a live analysis. Each setter sends the matching command and `announcePrefs()` replays them
  on the Mac's hello. `ask(_:heardAt:)` sends the question with `heard_at`; `hush()` stops the
  speaker and tells the Mac. Server text messages are parsed in `handleServerText`.
- `CameraAccess/Media/VoiceInput.swift`: the wearer's voice. The glasses mic is reached through
  the phone's audio session (`.playAndRecord` + `.allowBluetoothHFP`, the same route Meta's
  sample records sound-in-video through) or the phone's own mic (`.allowBluetoothA2DP`, keeps
  high-quality output). An `SFSpeechRecognizer` (on-device when available) streams partials; an
  utterance ends after 0.9 s without new words and a fresh request starts; `lastUtteranceStart`
  is when the sentence began. Modes: off, wake word (default "inspector", alone arms the next
  sentence for 8 s), always. Stop words are handled on the phone, everything else goes out as
  `ask`. The wake word barges in over playback; other utterances that end while the glasses are
  talking are dropped, and the recognizer context resets when playback ends (echo guard). The
  engine is rebuilt around Bluetooth route changes after waiting for the route to settle, so the
  tap never installs against a stale format (that was an uncatchable crash).
- `CameraAccess/Media/Speaker.swift`: `AVSpeechSynthesizer` for `speak`/`speak_fallback` and
  `PCMStreamPlayer` (`AVAudioEngine` + `AVAudioPlayerNode`) for ElevenLabs PCM, both on a
  `.playback/.spokenAudio` session so audio routes to the glasses over A2DP. It deliberately
  leaves a `.playAndRecord` session alone when voice input or the sample's recording owns it.
  Late PCM chunks for a stopped utterance id are dropped; the watermark resets on reconnect.
- `CameraViewModel` holds `frameRelay = FrameRelay.shared`. Inside the toolkit's
  `videoFramePublisher` callback (off the main actor) it decodes the HEVC frame, calls
  `frameRelay.push(previewImage)`, and drops the image into a single-slot `previewSlot` so at
  most one main-actor hop is in flight. Do not reintroduce a per-frame `Task { @MainActor }`;
  that unbounded backlog is what froze the UI at 720p. Stream resolution and fps are read from
  the `streamResolution` / `streamFPS` `UserDefaults` keys in `beginStream`, so they apply on
  the next Preview (default High, 15 fps).
- `VideoFrameDecoder` prefers hardware VideoToolbox decode and a GPU `CIContext`; software HEVC
  decode was the dominant CPU cost.
- `CameraView` shows the relay and mic chips, the live "You: …" transcript, the caption overlay,
  the What's here? and Hush buttons, and the `RelaySettingsView` sheet (Inspector section with the
  detector toggle, Stream, Voice, Voice input, Report, Diagnostics).
- `Info.plist` must keep `UISupportedExternalAccessoryProtocols = [com.meta.ar.wearable]` and
  the `external-accessory` background mode. That is what lets DAT stream over Bluetooth Classic
  on a free Apple team. The Wi-Fi (SoftAP) path needs HotspotConfiguration and wifi-info
  entitlements that a free team cannot sign, so `CameraAccess.entitlements` intentionally
  omits them. Also keep `NSBonjourServices` (`_glassesrelay._tcp`), `NSAllowsLocalNetworking`,
  `NSMicrophoneUsageDescription` and `NSSpeechRecognitionUsageDescription`.

### relay_receiver internals

One Claude call per question, no background analysis. The pieces, in order of a question:

1. `_ingest()` keeps `state["latest"]` and a ring buffer `state["recent"]` of about 200
   `(receive_ts, jpeg)` pairs (at 5 fps that is 40 s; at 15 fps about 13 s).
2. `handle_ask()` is the single entry for the phone's `ask`, `inspect`, the dashboard and
   `POST /ask`. Three-word-or-shorter stop words become `hush()`; anything else first hushes
   whatever is playing, waits (bounded) for `state["busy"]` to clear, then calls `answer()`.
3. `select_frames(since)` returns `FRAMES_PER_ASK` frames spread over the window from
   `heard_at` (or the last 2 s) to now, the sharpest one per time slice by Laplacian variance,
   oldest first. `_closeup()` runs the detector at `CROP_CONF` over those frames, sharpest first,
   and returns a padded crop of the first box it finds: a part held at arm's length is about a
   hundred pixels wide in the full frame, and the crop is what makes small-part matches hold.
4. `_claude_stream()` sends the frames, the crop, the previous exchange (text only, if under
   `MEMORY_SECONDS` old, so "does it need a driver" resolves "it" and names are not repeated)
   and the question. The system prompt is `TEACH_PROMPT` + a quick index (one line per part:
   id, names, what it looks like) + every full record, from `catalog.py`, about 58k tokens with
   `cache_control`; `_warm()` re-reads it every four minutes so the cache never lapses.
   `THINKING` is off by default: Sonnet 5 thinks by default, and a thinking block once consumed
   the entire output budget so no text came out; adaptive thinking at low effort measured no
   better on grounding and slightly slower.
5. Sentences are split as they stream and `_say()` sends each to the phone (plain `speak` for the
   Apple voice, or a queued ElevenLabs render). `hush()` bumps `speech["stop_gen"]`; the stream
   loop and `_tts_worker` compare against it and stop mid-answer. The report entry records the
   question, answer, the frame files (`ask-<ts>-<i>.jpg` plus `-crop.jpg`), first-token and total
   milliseconds.

`catalog.py`: `load()` reads `components.json`, `render_record()` clips each record to about
3 KB (identity, function, looks-like, pins with pinout, electrical, key specs, wiring, safety,
troubleshooting, price, uncertain fields), `index_line()` makes the quick index, `record()`
tolerates near-miss ids. Measured on the real API: first word in 1.0 to 1.5 s, whole answer in
2.5 to 3.5 s, cache read 57,863 tokens plus about 1.8k new tokens per question.

`detect.py` is unchanged from `main` except for its role: a YOLOv8n ONNX on
`CPUExecutionProvider`, default weights `weights/components_v3.onnx` (single class `component`,
960 px, letterbox via the `.json` sidecar, about 105 ms per frame). `_detect_loop` draws boxes
for the dashboard only when `detector["enabled"]` (off by default, `POST /detector` or the phone
toggle); `_closeup()` uses the model on demand regardless. `state["almost"]` carries the best
sub-threshold candidate for tuning.

Speech: `_say()` estimates playback seconds; with ElevenLabs active, sentences enter
`speech["queue"]` and one `_tts_worker` streams them in order; `tts_elevenlabs.py` renders
`eleven_flash_v2_5` as raw 24 kHz PCM with gain applied and caches by text hash in `tts_cache/`
(gitignored). The starter plan is 30,000 characters a month; `/health` reports usage.

Every answer appends to `report.jsonl` and saves its frames under `frames/` (both gitignored).
`/report.html` (behind the phone's report-page toggle) shows every question with its frames.
Bonjour registration retries for about 90 s because a just-killed instance's record lingers as
a name conflict. `ws_ingest` closes any older socket from the same client host when a new one
connects.

### Component catalog contract

`components.json` is `{"_meta", "components": [...]}`; each record has flat seed fields (`id`,
`name_on_kit`, `canonical_name`, `mpn`, `category`, `kit`, `qty`, `pin_count`, `voltage`,
`key_specs`, `price_usd`, `status`, `confidence`) and the full researched record under `details`
(`identity`, `function`, `pins`, `electrical`, `visual_identification`, `wiring_to_uno`, `safety`,
`troubleshooting`, `market`, `sources`, `confidence`). `catalog.render_record()` and
`index_line()` read all of those; renaming a field means touching them. The prompt tells Claude
to match what it sees against `visual_identification` first, so that section must describe the
variant actually in this kit (connector color, housing, fittings), not a generic datasheet
picture. `merge_records.py` refuses records without a datasheet/manufacturer source and flags
pin-count mismatches, so keep new records in that schema rather than patching the merged file.

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
- `INSPECT_FAKE=1` fakes the Claude call (a canned three-sentence answer) and skips the prompt
  warm; the detector and ElevenLabs still run for real. Use it to test the audio path and the
  dashboard without a key.
- The phone re-asserts its saved preferences every time it connects, so a voice or detector
  change made on the dashboard is undone by the next phone reconnect unless you also change it
  in the phone's gear menu.
- Deleting `tts_cache/` costs ElevenLabs characters on the next start.
- Voice input with the glasses mic puts the whole session on Bluetooth HFP, so ElevenLabs output
  is 16 kHz mono while listening and the SCO link shares Bluetooth Classic with the DAT video
  stream. Meta's own sample records HFP audio while streaming, so it should hold, but fps under
  load and the echo guard are not yet measured on the hardware (see the field notes).
- Grounding is only as good as the frames and the record: on 504 px frames with the part a
  hundred pixels wide, the same three questions about the stepper motor came back right in about
  seven of nine runs. Sonnet 5 accepts no temperature, so run-to-run variation is inherent; the
  levers are the 720p stream, holding the part closer, the detector crop, and accurate
  `visual_identification` text.
- The Mac's LAN and USB link-local addresses change during the day; always pick the receiver
  from the Bonjour list on the phone rather than typing an IP.
- Do not let the Mac hold the glasses over Bluetooth (headset pairing) while testing the phone
  app; the stream will not start. Another app with the same URL scheme breaks DAT registration.
