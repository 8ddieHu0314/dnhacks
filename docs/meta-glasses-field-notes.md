# Meta glasses developer playground

Purpose: experiment with Ray-Ban Meta / Meta Ray-Ban Display glasses and push the
developer surface as far as it goes. Notes gathered 2026-09-05.

## Official developer surface (Meta Wearables Developer Center)

Docs hub: https://wearables.developer.meta.com/
FAQ: https://developers.meta.com/wearables/faq/
Wearables MCP endpoint (for Claude Code / Cursor etc.): https://mcp.developer.meta.com/wearables

Two build paths, both in Developer Preview (publishing not yet open; share
with up to 100 testers via release channels / password-protected URLs):

### 1. Device Access Toolkit (DAT) — native iOS (Swift) / Android (Kotlin)
- Repos: https://github.com/facebook/meta-wearables-dat-ios (SPM)
         https://github.com/facebook/meta-wearables-dat-android (Maven, GitHub Packages, needs PAT with read:packages)
- Latest SDK 0.9.0. Android artifacts: mwdat-core, mwdat-camera, mwdat-display, mwdat-mockdevice.
- Capabilities: session/device lifecycle, camera video streaming (configurable
  res/fps), photo capture, mic + speakers over Bluetooth, display output
  (Display glasses only). Mock Device Kit for testing without hardware
  (does not mock the display).
- Supported glasses: Ray-Ban Meta Gen 1/Gen 2, Meta Ray-Ban Display,
  Oakley Meta HSTN, Oakley Meta Vanguard.
- Requirements: iOS 15.2+ / Android 10+; Meta AI app V282 and glasses
  firmware V126 (Display: V125) for DAT 0.9.0.
- Developer Mode: Meta AI app > Settings > App Info > tap App version 5x > toggle Developer Mode.
- Official Claude Code plugins:
    claude plugin marketplace add facebook/meta-wearables-dat-ios
    claude plugin marketplace add facebook/meta-wearables-dat-android
    claude plugin install mwdat-android@mwdat-android-marketplace

### 2. Web Apps — HTML/CSS/JS running on Meta Ray-Ban Display (no phone app)
- Docs: https://wearables.developer.meta.com/docs/develop/webapps
        https://wearables.developer.meta.com/docs/develop/webapps/build/
- Must be hosted on public HTTPS; loaded on the glasses by URL.
- Display: fixed 600x600 viewport, additive waveguide (black = transparent),
  use dark backgrounds, no scrolling.
- Input: Neural Band / captouch gestures arrive as keyboard events
  ArrowUp/Down/Left/Right, Enter (index pinch), Escape (middle pinch).
  No pointer. Everything must be reachable by focus navigation; ~88px targets.
- Sensors: DeviceMotionEvent / DeviceOrientationEvent (IMU on glasses),
  navigator.geolocation (proxied from the phone), localStorage/sessionStorage (5 MB).
- NOT available in web apps: camera, microphone, text input, offline,
  notifications, back navigation, continuous cursor, custom Neural Band gestures.
- Requires Display glasses firmware v125+, Meta AI app v272+, Developer Mode.

## Unofficial / research angles
- BLE reverse engineering toolkit (scanner, GATT explorer, notification
  monitor, packet analyzer): https://github.com/lingster/meta-rayban-bluetooth
  Only standard services (0x180A device info, 0x180F battery) documented;
  camera control and voice assistant services still undocumented.
- BLE fingerprinting / detection of Meta glasses: https://github.com/NullPxl/banrays
- GitHub topic: https://github.com/topics/ray-ban-meta
- Livestreaming to Instagram/Facebook/WhatsApp video calls is built in and
  can be a poor man's video feed on any device without DAT.

## Local toolchain (checked 2026-09-05)
- claude CLI present, node v25.2.1, xcodebuild on PATH (full Xcode not verified).

## Experiment backlog
1. Enable Developer Mode; record which glasses model + firmware we have.
2. Web app: 600x600 dark-theme page with focus navigation, deploy to Vercel/GitHub Pages,
   load on Display glasses. Try IMU + geolocation.
3. DAT iOS sample app: camera stream into a local vision pipeline (Claude API).
4. Mock Device Kit run without hardware.
5. Register the Wearables MCP endpoint in Claude Code for live docs/constraints.
6. BLE sniffing session with the lingster toolkit to map proprietary services.

## Simple project ideas (2026-09-05)

Web Apps (Display glasses, 600x600, arrow/Enter/Escape input, no camera):
1. Pomodoro timer — swipe to pick duration, pinch to start
2. Teleprompter — script in localStorage, swipe to scroll
3. Heads-up compass — DeviceOrientationEvent heading
4. Walking pace/distance — navigator.geolocation.watchPosition
5. Recipe stepper — one step per screen
6. Transit countdown — nearest stop, next 3 departures
7. Flashcards — flip with pinch, progress in localStorage
8. Head-tilt maze game — accelerometer steering

Device Access Toolkit (native, camera + mic + speakers):
9. "What am I looking at" — photo -> Claude vision -> spoken answer
10. Receipt / business-card scanner — photo -> OCR
11. Timelapse recorder — sample frames from the stream
12. Live captioning — mic -> speech-to-text
13. Barcode lookup — frame stream -> barcode detector -> product info
14. Remote viewer — camera stream over WebRTC
15. Motion-triggered snapshot — frame diffing

No SDK (BLE / built-in features):
16. Battery + presence widget — standard BLE battery service
17. Livestream-to-pipeline — built-in IG/WhatsApp streaming captured on another device

NOTE 2026-09-05: user's glasses have NO display (Ray-Ban Meta / Oakley, not Display).
Web Apps path (1-8) does not apply. Everything goes through DAT: the glasses are a
camera + Bluetooth headset peripheral, the phone app is where the logic and UI live.
Feedback to the wearer is audio only (speakers) or on the phone screen.
First three: 9 (Claude vision, voice answer), 12 (live captioning), 16 (BLE battery).

## DNHacks 2026 (Sept 5-6, Washington DC) — Energy & Industrialization track (Deterrence)
Track scope: critical minerals, smart energy, agriculture; mining site ID, hazard detection,
drilling optimization, ag robotics. Judging: technical quality, real-world impact, execution.
Special award for AI usage.

Glasses advantage: hands-free first-person camera + voice in/out for workers with busy/gloved hands.

Ideas (strongest first):
1. Hands-free field inspection copilot — "inspect": frame -> Claude vision -> component ID,
   nameplate/gauge read, checklist check, spoken result, geotagged report on web dashboard  <-- RECOMMENDED
2. Continuous hazard watch — stream frames, warn on missing PPE, open panel, spill, LOTO, exposed conductor
3. Gauge/meter reader with trending dashboard
4. Critical minerals field logger — rock/core photo + GPS + voice note + vision guess -> site map
5. Voice-guided maintenance with visual step verification (LOTO compliance)
6. Crop/livestock scouting heatmap
7. Electrical panel -> digital schematic
8. Remote expert POV stream (WebRTC, risky overnight)

Plan: build 1 with 2 as a mode. Voice commands: inspect / read / watch.
Hour-1 risks: Meta AI app V282 + firmware V126 + Developer Mode; Xcode signing for DAT iOS build.
Fallback: built-in photo capture synced via Meta AI app, processed on laptop.

## Hardware on hand (2026-09-05)
- Ray-Ban Meta Wayfarer, shows on the Mac as "RB Meta 0CB4", BT addr 80:AA:1C:92:AE:86,
  vendor 0x01AB (Meta), product 0x010A, firmware 20.3.6. Paired with this Mac as a headset.
- No SSH / ADB / shell: closed firmware on Qualcomm AR1. Connection surfaces are
  (a) Bluetooth Classic headset profile (HFP/A2DP), (b) BLE GATT, (c) Meta AI app -> DAT.
- Tools: blueutil installed (brew); bleak venv + scan.py in the session scratchpad.
- First attempt: BLE scan (252 devices) did not see the glasses, blueutil connect failed.
  Glasses were likely in the case or held by the phone. Retry with glasses on and phone BT off.
- 2026-09-05 12:59: glasses firmware release 127.14.0.220.436 (>= V126 required by DAT 0.9.0). OK.
- Classic BT connect from Mac works (blueutil --connect 80-aa-1c-92-ae-86). Mac gets the glasses as
  default mic (HFP, 16 kHz mono) and speaker (A2DP, 44.1 kHz stereo). `say` plays through them.
  Python: sounddevice device 1 = glasses mic, device 2 = glasses speaker (venv in scratchpad).
  => Laptop-side voice loop (mic -> STT -> Claude -> TTS -> glasses) works with zero SDK.
- BLE: glasses do not advertise while connected over classic BT; scan them with everything disconnected.
- Camera still requires DAT via the Meta AI app on the phone.

## Status 2026-09-05 13:10 — iOS path
- Glasses firmware 128, Meta AI app 287.0.0.11.156, Developer Mode ON. All above DAT 0.9.0 minimums.
- Cloned facebook/meta-wearables-dat-ios into ./meta-wearables-dat-ios (samples/CameraAccess is the base).
- Sample requirements: iOS 17.2+, Xcode 26.4+, Swift 6.3+. Developer Mode => MetaAppID can be 0,
  no Wearables Developer Center registration needed for testing.
- BLOCKER: this Mac has only Command Line Tools, no Xcode, no signing identity.
  Need: Xcode 26.4+ from App Store, Apple ID added in Xcode > Settings > Accounts (free personal team),
  iPhone in Developer Mode (Settings > Privacy & Security > Developer Mode).
- Also need ANTHROPIC_API_KEY (not in env) for the Claude vision / report step.
- DAT API shape (from sample): Wearables.configure(); startRegistration() -> Meta AI callback via URL scheme;
  AutoDeviceSelector; DeviceSession.start(); Stream(StreamConfiguration(resolution:.low, frameRate:24));
  stream.videoFramePublisher.listen { frame in frame.makeUIImage() }; stream.capturePhoto(format:.jpeg).
- Plan A (camera): fork CameraAccess -> "Inspector": on tap / voice, grab current frame -> Claude vision
  -> spoken result (AVSpeechSynthesizer routed to glasses) -> append to report (JSON) -> push to laptop dashboard.
- Plan B (no Xcode in time): laptop voice loop through glasses mic/speaker + photos taken with the
  glasses' capture button, synced via Meta AI app, dropped into a watch folder on the Mac.

## Video streaming pipeline (2026-09-05, built)
Glasses camera -> (Meta AI app / DAT, BT+WiFi) -> iOS app GlassesInspector -> HTTP POST JPEG @4fps -> Mac server :8787
  -> live view + Claude vision (/inspect) + report.jsonl

- GlassesInspector/ = fork of Meta's CameraAccess sample. Bundle id com.eddiehu.glassesinspector.CameraAccess.
  Added CameraAccess/Media/FrameRelay.swift (throttled JPEG POST, UserDefaults "relayURL"),
  relay status chip + relay URL/toggle in the gear menu (Views/CameraView.swift),
  frameRelay.push(image) in CameraViewModel after each decoded preview frame,
  NSAllowsLocalNetworking in Info.plist. Compiles (xcodebuild, CODE_SIGNING_ALLOWED=NO).
- server/ = FastAPI receiver. ./run.sh starts it on 0.0.0.0:8787. Endpoints: POST /frame, GET /latest.jpg,
  GET / (live dashboard), POST /inspect {question}, GET /report, GET /health. Model claude-opus-5 with
  server-side fallbacks. SPEAK=1 speaks results with `say`. Needs ANTHROPIC_API_KEY for /inspect only.
- Mac LAN IP at build time: 192.168.1.7 (default relay URL in the app; editable in the gear menu).
- Build from CLI: DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -project GlassesInspector/CameraAccess.xcodeproj -scheme CameraAccess -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
- Remaining to run on the phone: Apple ID in Xcode > Settings > Accounts, pick the team in Signing & Capabilities, Run.
- 2026-09-05 10:50 DIAGNOSIS of "Device is not connected" at stream start (session started, link=connected, compat=Undefined):
  MWDATCamera needs a "medium (BTC) or high (WiFi) bandwidth link". High = SoftAP via NEHotspotConfiguration
  (needs HotspotConfiguration + wifi-info entitlements; NOT available on a free personal team, and I removed them
  to sign). Medium = Bluetooth Classic via ExternalAccessory; the SDK demands
  UISupportedExternalAccessoryProtocols = ["com.meta.ar.wearable"] in Info.plist (sample omits it). Added that
  plus external-accessory background mode. If BTC path still fails, fallback = a paid Apple developer team
  (restore the two entitlements in CameraAccess.entitlements).
- On-glasses DAT app confirmed installed: Meta AI > App info shows "RB Meta 0CB4 DAT SDK version 0.9.0.26.0".
- 2026-09-05 10:50 STREAM WORKING end to end. Glasses -> phone (BT Classic via ExternalAccessory) -> Mac relay.
  Receiver saw 112 frames at ~2.4 fps, 360x640 JPEG (~50 KB each). compat still "Undefined" but harmless.
  Next: Claude /inspect (needs ANTHROPIC_API_KEY), spoken results on the phone, hackathon demo polish.
- 2026-09-05 11:10 Relay v2: phone pushes frames over WebSocket (ws://mac:8787/ws/ingest, 8-byte capture ts + JPEG),
  server fans out to browsers on /ws/view (latest-frame-only queues), dashboard draws on a canvas (rotate, fit/fill)
  with HUD: frame size, relay fps, shown fps, phone->mac ms, KB/frame. Phone gear menu: resolution low/medium/high
  (applies on next Preview), relay fps 2-24 (default 15), JPEG quality (default 0.8). Server must run with
  `--ws websockets` (run.sh does). Self-test: 30 frames @15fps delivered, ~7 ms server latency.
- 2026-09-05 11:40 Measurements over venue Wi-Fi: Medium 504x896 = 4.6 fps in/out, 16 ms/send, 92 KB, phone->mac 31 ms
  (glasses->phone BT Classic is the ceiling). High 720x1280 = 1 fps, 1003 ms/send, 143 KB, 1863 ms latency
  (Wi-Fi collapses while BT Classic streams; venue Wi-Fi RTT 130 ms avg). Mac is on 5 GHz ch 128.
- USB path: the iPhone exposes a USB Ethernet link without Personal Hotspot. Mac en10 = 169.254.88.41,
  phone = 169.254.8.239 (iphone-15.local), RTT < 1 ms. Relay URL for the phone: http://169.254.88.41:8787.
  (Mac's 169.254 address can change if the cable is re-plugged: `ipconfig getifaddr en10`.)
- Phone Relay chip now shows: in <toolkit fps> · out <relay fps> · ms/send · KB · sent.
- 2026-09-05 11:50 Gear icon now opens a Settings sheet (the old popover put the URL field off-screen). Receiver
  advertises itself over Bonjour (_glassesrelay._tcp, TXT "urls"=comma list, refreshed every 10 s); the sheet lists
  discovered receivers as one-tap buttons, USB link-local (169.254.x) first. Mac addresses DO change during the day
  (seen 192.168.1.7 -> 192.168.8.47, 169.254.88.41 -> 169.254.89.99), so always pick from the list.
  Dashboard on the Mac itself: http://localhost:8787
- 2026-09-05 12:05 Relay v3: Network.framework NWConnection + NWProtocolWebSocket. Target = Bonjour service name
  (_glassesrelay._tcp, resolved at connect time) with "Prefer USB cable" = prohibitedInterfaceTypes [wifi, cellular],
  auto-fallback to any interface if the cable-only attempt has no path. Manual URL kept as fallback toggle.
  Server: websocket at "/" aliases /ws/ingest (service endpoints carry no path); Bonjour refresh task fixed
  (was garbage-collected). USB link-local IP changed 3x in an hour (every Xcode install cycles the interface).
- 2026-09-05 14:55 Relay v3.1: cable attempt = NWPathMonitor finds the wired/other interface, NWConnection with
  requiredInterface to the advertised 169.254 URL (else <name>.local), 3 s budget; timeout/.waiting/.failed marks
  cable failed -> next attempt uses any interface (service endpoint or manual URL); cable retried after 60 s.
  Prohibiting Wi-Fi alone left the connection stuck in .preparing forever (no .waiting), so v3 never sent anything.
  Browser: includePeerToPeer=false (true hid the Mac's LAN advertisement). Default manual URL = Eddies-MacBook-Pro.local.
- 2026-09-05 15:20 STREAMING AGAIN via discovered receiver. Root cause of the v3 outage: python-zeroconf SRV target
  defaulted to "<name>._glassesrelay._tcp.local." which iOS refuses to resolve -> NWConnection stuck in .preparing.
  Fixed with ServiceInfo(server="<host>.local."); phone now also connects straight to the advertised LAN IP.
  Readings (relay cap 24): Medium 504x896 14.8 fps, 171 ms, 72 KB; High 720x1280 12.1 fps, 252 ms, 139 KB.
  Bytes*fps ~ 8-13 Mbps => phone->Mac hop is bandwidth/latency bound (looks like Wi-Fi, not cable).
  Added: 3 frames in flight (pipelining), glasses frame-rate picker 15/24/30, relay cap up to 30.
- 2026-09-05 16:30 Claude -> glasses audio loop built. Mac: analyze() streams Claude (AsyncAnthropic, claude-opus-5,
  server-side fallbacks) on the latest frame, pushes each finished sentence to the phone over the ingest WebSocket
  as {"type":"speak","text":...} then {"type":"speak_end"}; viewers get live "caption" messages. Endpoints:
  POST /inspect, POST/GET /narrate {enabled, interval} (continuous loop, "no change" suppressed, previous
  narration passed as context). INSPECT_FAKE=1 streams canned text without a key. run.sh sources server/.env.
  Phone: Speaker.swift (AVSpeechSynthesizer, playback/spokenAudio session -> glasses over A2DP), Describe button,
  caption overlay, settings: speak toggle, continuous narration + interval, test voice. Phone->Mac commands:
  {"type":"inspect"}, {"type":"narrate"}. Verified fake round trip: first sentence at 0.7 s, speak_end at 2.6 s.
