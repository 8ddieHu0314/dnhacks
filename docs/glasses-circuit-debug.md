# Meta Ray-Ban circuit-debug test

The glasses do not run Claude directly. The Meta DAT iOS app receives their
camera stream, sends timestamped JPEG frames to the Mac, and plays the Mac's
`speak` replies over the iPhone's active glasses audio route.

## Prerequisites

- Ray-Ban Meta glasses paired with the Meta AI app, not with the Mac.
- Developer Mode enabled in Meta AI.
- An iPhone on iOS 17.2 or newer and the Mac on the same local network.
- Full Xcode 26.4 or newer. Apple Command Line Tools alone cannot build the app.
- A valid Anthropic API key exported only in the server terminal.

## Start the Mac service

From the repository root, install dependencies once:

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Enter the temporary key without echoing or saving it, then launch:

```zsh
read -s "VLM_API_KEY?Anthropic API key: "; export VLM_API_KEY; echo
services/vision_api/run_glasses.sh
```

The launcher selects the fast catalog-identification backend, Claude Haiku,
one inference per second, glasses speech, and port 8000. Check it from another
terminal with `curl http://127.0.0.1:8000/health`.

## Install the iPhone app

1. Open `clients/ios/GlassesInspector/CameraAccess.xcodeproj` in full Xcode.
2. Select the `CameraAccess` target and your signing team. Debug builds use Meta
   App ID `0`; production builds need Wearables Developer Center credentials.
3. Select the physical iPhone as the run destination and press Run.
4. In Meta AI, confirm Developer Mode is still enabled.
5. In Glasses Inspector, tap Connect and complete the Meta AI registration.

## Connect and run

1. Find the Mac's LAN address with `ipconfig getifaddr en0` (try `en1` if it is
   blank). Do not use `127.0.0.1`; that means the iPhone itself.
2. Open the app's gear menu. Enable **Use manual URL** and enter
   `http://MAC_ADDRESS:8000`.
3. Enable **Relay frames** and **Speak results through glasses**. Set the relay
   cap to 2 fps and JPEG quality near 70 percent for a stable first test.
4. Tap **Start Session**, then **Preview**. Accept camera and local-network
   prompts. The relay diagnostics should show a WebSocket connection and a
   rising sent-frame count.
5. Tap **Test voice on glasses**. Fix the iPhone's Bluetooth audio route before
   debugging if that sentence does not play through the glasses.
6. Under **Voice input**, choose Wake word and the glasses or phone microphone.
   Say “Inspector, what is wrong with this circuit?”; the question is applied to
   the first full debug pass even when that frame initially detects the breadboard.

Point at an ordinary kit part to test identification. Point at a breadboard to
switch that socket's session automatically into circuit debugging. The first
reply identifies the breadboard; the following frame reconstructs and compares
the circuit. The phone shows **POWER LOCKED** until every blocking check passes.

Claude can request a top-down, side, relay-pin, or breadboard-row view. Move your
head to supply that view and hold still through the next reply. Keep both USB and
the external motor supply disconnected while the gate is locked.

Use **Circuit evidence** in the gear menu for facts a camera cannot prove. Enter
the actual resistance or continuity result for meter checks; do not submit a
guess. The server retains accepted evidence across later frames. Submit powered
behavior only after the app says **READY TO TEST**.

## Fast fault isolation

- No sent frames: Preview is not streaming, Relay frames is off, or the socket
  address is wrong. Inspect the app's Diagnostics section.
- Sent frames but no response: check the Mac terminal and `/health`; confirm the
  API key is exported as `VLM_API_KEY`, not `ANTHROPIC_API_KEY`.
- Identification never changes to debug: fill most of the view with the
  breadboard under even light, then tap Describe once.
- Text but no sound: enable speech, use Test voice, and select the glasses as the
  iPhone's media output. Server-side `SPEECH_MODE` must be `glasses` or `both`.
- Slow response: use low glasses resolution, a 2 fps relay cap, and keep the
  one-second inference interval. Raising frame rate does not make Claude faster.

Frames sent in this mode leave the local network for Anthropic inference. This
prototype is an observation aid, not an electrical safety authority.
