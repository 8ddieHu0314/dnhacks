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

`docs/meta-glasses-field-notes.md` is the running log: hardware/firmware facts, SDK
constraints, measurements, root causes of past outages, and the backlog. Read it before
touching the iOS or relay code, and append dated entries there when something changes.

Hardware fact that shapes everything: the user's glasses are Ray-Ban Meta (camera + BT
headset, **no display**), so Meta's Web Apps / display path does not apply despite the branch
name. All feedback to the wearer is audio.

## Commands

### vision_api (root pyproject)

```bash
cp .env.example .env                      # Settings reads .env from the CWD
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
PYTHONPATH=services uvicorn vision_api.main:app --reload
docker compose up --build                 # same service, needs .env
```

Tests are `unittest`-style (`IsolatedAsyncioTestCase`); pyproject configures pytest with
`pythonpath = ["services"]`, so both runners work:

```bash
PYTHONPATH=services python3 -m unittest discover -s tests -v
pytest                                    # all
pytest tests/test_pipeline.py -k failure  # one test by keyword
PYTHONPATH=services python3 -m unittest tests.test_vlm -v   # one module
```

### relay_receiver (separate venv, separate requirements)

```bash
services/relay_receiver/run.sh            # creates .venv, installs requirements.txt, sources .env, starts :8787
INSPECT_FAKE=1 services/relay_receiver/run.sh   # canned Claude output, no API key
SPEAK=1 services/relay_receiver/run.sh          # also `say` results on the Mac
```

`ANTHROPIC_API_KEY` goes in `services/relay_receiver/.env` (gitignored). `INSPECT_MODEL`
overrides the model (default `claude-opus-5`). Dashboard: http://localhost:8787. Keep the
`--ws websockets --ws-ping-timeout 90` flags in `run.sh`; the phone socket stalls without them.
There are no tests for this service.

### iOS app

Needs full Xcode (26.4+), not Command Line Tools. Compile check from the CLI:

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
- Discovery: the Mac advertises `_glassesrelay._tcp` via zeroconf with TXT `urls` (comma list of
  `http://ip:8787`, refreshed every 10 s) and `server="<host>.local."`. The SRV target must be a
  plain hostname; iOS refuses to resolve the zeroconf default and the connection sits in
  `.preparing` forever.

### iOS additions on top of Meta's sample

All custom code is marked "Glasses Inspector addition" and lives in two files plus small hooks:

- `CameraAccess/Media/FrameRelay.swift`: `RelaySocket` (actor; `NWConnection` +
  `NWProtocolWebSocket`; tries the USB link-local `169.254.x` URL on the wired interface first
  with a 3 s budget, then falls back to LAN IP / service name, retries cable after 60 s),
  `RelayBrowser` (`NWBrowser` for Bonjour, `includePeerToPeer=false`), `RelayEngine` (actor;
  throttle to `targetFPS`, JPEG encode, single latest-frame slot, up to `maxInFlight` sends),
  `MainThreadMeter`, and `FrameRelay` (the `@Observable @MainActor` facade the views bind to;
  persists settings in `UserDefaults` under `relay*` keys; parses server text messages).
- `CameraAccess/Media/Speaker.swift`: `AVSpeechSynthesizer` on a `.playback/.spokenAudio`
  session so speech routes to the glasses over A2DP. It deliberately leaves a `.playAndRecord`
  session alone when the sample is recording.
- Hooks: `CameraViewModel` owns `frameRelay` and calls `frameRelay.push(previewImage)` per
  decoded frame off the main actor; `CameraView` shows the relay chip, caption overlay,
  Describe button, and `RelaySettingsView` sheet.
- `Info.plist` must keep `UISupportedExternalAccessoryProtocols = [com.meta.ar.wearable]` and
  the `external-accessory` background mode. That is what lets DAT stream over Bluetooth Classic
  on a free Apple team. The Wi-Fi (SoftAP) path needs HotspotConfiguration and wifi-info
  entitlements that a free team cannot sign, so `CameraAccess.entitlements` intentionally
  omits them. Also keep `NSBonjourServices` (`_glassesrelay._tcp`) and `NSAllowsLocalNetworking`.

### relay_receiver internals

Module-level `state` dict (latest JPEG, fps/latency windows, report), `phones` and
`viewer_sockets` sets. `analyze()` streams Claude (`AsyncAnthropic`, image + prompt, beta
server-side fallbacks), splits on sentence boundaries, and pushes each sentence to phones as it
lands so speech starts early. `_narration_loop` re-runs with the previous narration as context
and suppresses the literal reply `no change`. Every analysis appends to `report.jsonl` and saves
the frame under `frames/` (both gitignored). Only one analysis runs at a time (`narration["busy"]`).

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
