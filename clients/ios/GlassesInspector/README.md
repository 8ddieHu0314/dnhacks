# Glasses Inspector (iOS)

Companion app that pulls the live camera stream from Ray-Ban Meta glasses via Meta's
Device Access Toolkit (DAT 0.9.0) and relays JPEG frames to a receiver on the LAN.
Forked from Meta's `samples/CameraAccess` in
[facebook/meta-wearables-dat-ios](https://github.com/facebook/meta-wearables-dat-ios);
the toolkit is pulled in by Swift Package Manager on first build.

## Requirements
- Xcode 26.4+, iOS 17.2+ phone, glasses firmware >= v126, Meta AI app >= v282, Developer Mode on.
- Free personal Apple team is enough. See "Signing on a personal team" below.

## Build
Open `CameraAccess.xcodeproj`, pick your team under Signing & Capabilities, run on the phone.
In the app: Connect (approve in Meta AI) -> Start Session -> Preview. Gear icon = relay settings.

## Signing on a personal team (important)
Meta's sample carries two Wi-Fi entitlements (Hotspot Configuration, Access Wi-Fi Information)
that only paid teams can sign. They are removed here. Without them the toolkit cannot use its
Wi-Fi video path, so this app declares `UISupportedExternalAccessoryProtocols = [com.meta.ar.wearable]`
in Info.plist, which enables the toolkit's Bluetooth Classic video path instead. Streaming works
at 360p/504p/720p, ~24 fps at the phone; "compat=Undefined" in the status line is harmless.

## Relay
- `Media/FrameRelay.swift`: WebSocket push (8-byte capture timestamp + JPEG) over Network.framework.
  Finds the receiver via Bonjour (`_glassesrelay._tcp`), prefers the USB cable when the phone has a
  169.254.x.x interface, falls back to Wi-Fi automatically. Latest-frame slot, up to 3 in flight.
- Receiver: `services/relay_receiver` in this repo (dashboard + `/ws/ingest`).
- To target the vision API instead, point the relay at `WS /v1/sessions/{id}/frames` and send the
  `FrameMetadata` JSON before each JPEG (see repo README).
