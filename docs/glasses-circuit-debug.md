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
2. Select the `CameraAccess` target and your signing team. Keep the configured
   Meta App ID, client token, URL scheme, and wearable entitlement intact.
3. Select the physical iPhone as the run destination and press Run.
4. In Meta AI, confirm Developer Mode is still enabled.
5. In Glasses Inspector, tap Connect and complete the Meta AI registration.
