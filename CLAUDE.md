# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DNHacks 2026 project: a hands-free field-inspection copilot for Ray-Ban Meta glasses. Three
independent pieces live here, and only two of them talk to each other:

| Path | What | Talks to |
|---|---|---|
| `clients/ios/GlassesInspector` | iPhone app (fork of Meta's DAT `CameraAccess` sample). Pulls the glasses camera stream, relays JPEGs to the Mac, speaks Claude's answers into the glasses. | `services/relay_receiver` |
| `services/relay_receiver` | Single-file FastAPI receiver on the Mac (`:8787`). Live dashboard, Claude vision on the latest frame, spoken results back to the phone. | the iOS app, browsers |
| `services/vision_api` | Earlier, tested FastAPI scaffold: sessions, versioned workflow packages, bounded latest-frame queue, pluggable `VisionEngine`. Not wired to the phone app. | nothing yet |

Two docs matter as much as the code:

- `docs/meta-glasses-field-notes.md` is the running log: hardware/firmware facts, SDK
  constraints, measurements, root causes of past outages, and the backlog. Read it before
  touching the iOS or relay code, and append dated entries there when something changes.
- `docs/TEAM_PLAN.md` splits the demo into three tracks (phone, Mac brain, demo/story) on
  branches `track-phone`, `track-brain`, `track-demo` off `meta-glasses-display-access`, and
  pins the phone<->Mac interfaces. Change an interface only with the other side's owner in
  the loop, and record the change there.

Hardware fact that shapes everything: the user's glasses are Ray-Ban Meta (camera + BT
headset, **no display**), so Meta's Web Apps / display path does not apply despite the branch
name. All feedback to the wearer is audio.

Things you may see that are not part of this branch: `clients/ios/StockSample/` (gitignored)
is Meta's untouched sample kept only for A/B lag comparisons against the fork. The remote
`feat/gabe-*` and `feat/live-webcam-demo` branches carry a separate PPE-detection server
(`app.py`, `server.py`, `core/` at the repo root) that speaks the same `/ws/ingest` wire
format; they diverge from `main` (vision_api tests removed), so do not merge them blindly.

## Commands

### vision_api (root pyproject)

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
`pythonpath = ["services"]`, so both runners work:

```bash
PYTHONPATH=services python3 -m unittest discover -s tests -v
pytest                                    # all
pytest tests/test_pipeline.py -k failure  # one test by keyword
PYTHONPATH=services python3 -m unittest tests.test_vlm -v   # one module
```

Use the root `.venv` for these, not `services/relay_receiver/.venv`; the relay venv lacks
`httpx` and `pydantic-settings`, so every test module fails to import there.

### relay_receiver (separate venv, separate requirements)

```bash
services/relay_receiver/run.sh            # creates .venv, installs requirements.txt, sources .env, starts :8787
INSPECT_FAKE=1 services/relay_receiver/run.sh   # canned Claude output, no API key
SPEAK=1 services/relay_receiver/run.sh          # also `say` results on the Mac
```

`ANTHROPIC_API_KEY` goes in `services/relay_receiver/.env` (gitignored). `INSPECT_MODEL`
overrides the model (default `claude-opus-5`). Dashboard: http://localhost:8787. Keep the
`--ws websockets --ws-ping-timeout 90` flags in `run.sh`; the phone socket stalls without them.
There are no tests for this service. Ask before restarting it on the demo Mac; it holds the
phone's socket.

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

## Architecture

### Phone <-> Mac wire protocol (FrameRelay.swift <-> relay_receiver/app.py)

Changing either side means changing both. Everything rides one WebSocket from the phone to
`/ws/ingest` (`/` is an alias because Bonjour service endpoints carry no path).

- Phone -> Mac binary: 8-byte big-endian capture timestamp (ms since epoch) + JPEG bytes. The
  server treats a message starting with `FF D8` as a bare JPEG with no timestamp.
- Phone -> Mac text (JSON): `{"type":"inspect","question"?:...}`, `{"type":"narrate","enabled":bool,"interval":s}`.
- Mac -> Phone text: `{"type":"speak","text":sentence}` per finished sentence, then
  `{"type":"speak_end"}`; `{"type":"narration","enabled","interval"}` on connect and on change.
- Mac -> Browser (`/ws/view`): binary frames from a latest-only queue (maxsize 1), plus
  `{"type":"stats",...}` every 0.5 s and `{"type":"caption","text","final"}` while Claude streams.
- Mac HTTP: `POST /inspect {question?}`, `POST`/`GET /narrate {enabled, interval}`, `GET /report`,
  `GET /health`, `GET /latest.jpg`, `GET /frames/{name}`, dashboard on `/`. `POST /frame` (raw
  JPEG body, optional `x-capture-ts` ms header) is the ingest fallback and the laptop-webcam
  demo mode.
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
  model. Settings persist in `UserDefaults` under `relay*` keys; server text messages are parsed
  in `handleServerText`.
- `CameraAccess/Media/Speaker.swift`: `AVSpeechSynthesizer` on a `.playback/.spokenAudio`
  session so speech routes to the glasses over A2DP. It deliberately leaves a `.playAndRecord`
  session alone when the sample is recording.
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
  `RelaySettingsView` sheet. `WearablesViewModel.deviceStatusText` is the per-device
  link/compat line shown there.
- `Info.plist` must keep `UISupportedExternalAccessoryProtocols = [com.meta.ar.wearable]` and
  the `external-accessory` background mode. That is what lets DAT stream over Bluetooth Classic
  on a free Apple team. The Wi-Fi (SoftAP) path needs HotspotConfiguration and wifi-info
  entitlements that a free team cannot sign, so `CameraAccess.entitlements` intentionally
  omits them. Also keep `NSBonjourServices` (`_glassesrelay._tcp`) and `NSAllowsLocalNetworking`.

### relay_receiver internals

Module-level `state` dict (latest JPEG, fps/latency windows, report), `phones` and
`viewer_sockets` sets. `analyze()` streams Claude (`AsyncAnthropic`, image + prompt, beta
server-side fallbacks), splits on sentence boundaries, and pushes each sentence to phones as it
lands so speech starts early. Prompt tuning lives in the `SYSTEM_PROMPT` and `NARRATION_PROMPT`
constants. `_narration_loop` runs only while the latest frame is under 3 s old, re-runs with the
previous narration as context, suppresses the literal reply `no change`, and clamps the interval
to at least 3 s. Every analysis appends to `report.jsonl` and saves the frame under `frames/`
(both gitignored). Only one analysis runs at a time: `narration["busy"]` guards `/inspect`, phone
commands, and the loop. Bonjour registration retries for about 90 s because a just-killed
instance's record lingers as a name conflict.

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

Design invariant across this service: nothing here authorizes work. Workflows shape model
context only, `ActionProposal.requires_confirmation` defaults to true, and advisory field signals
are observations that can warn but never establish a deenergized state (see README for the OSHA
basis). Keep that boundary when adding endpoints or workflow definitions.

## Environment quirks

- `Settings` (pydantic-settings) reads `.env` relative to the working directory, so run
  `uvicorn` and tests from the repo root.
- The Mac's LAN and USB link-local addresses change during the day; always pick the receiver
  from the Bonjour list on the phone rather than typing an IP.
- Do not let the Mac hold the glasses over Bluetooth (headset pairing) while testing the phone
  app; the stream will not start. Another app with the same URL scheme breaks DAT registration.
