# Ray-Ban Meta live vision scaffold

This repository starts the server side of a hands-free industrial/field-vision
prototype: receive a wearer’s point-of-view frames, keep latency bounded, run a
segmentation model, and return structured regions to a mobile client.

The [Meta Wearables Device Access Toolkit](https://developers.meta.com/blog/introducing-meta-wearables-device-access-toolkit/)
gives preview developers access to glasses camera and audio functionality through
their mobile apps. This project does **not** assume a private transport or API
shape from Meta. Instead, it defines the server boundary immediately after the
native companion app has obtained and decoded a camera frame.

## Pipeline

```text
Meta glasses camera
      │  Device Access Toolkit / native iOS or Android companion app
      ▼
Video transport adapter (WebRTC/H.264, when SDK details are available)
      │  timestamped JPEG, PNG, or WebP frames
      ▼
FastAPI ingest service ──► bounded latest-frame queue ──► segmentation engine
      │                                                        │
      └──────── ACK / metrics ◄──── structured masks ◄─────────┘
```

The current service receives **decoded image frames**, not a raw video codec.
That isolates vision and segmentation from the eventual glasses transport. A
mobile bridge can initially sample video at 4–10 FPS, encode a JPEG/WebP frame,
and send it over the WebSocket. A later WebRTC/H.264 adapter simply needs to
emit the same `FrameMetadata + bytes` pair.

## What is implemented

- `POST /v1/sessions` creates a short-lived stream session.
- `WS /v1/sessions/{session_id}/frames` accepts alternating metadata JSON and
  binary image messages.
- `POST /v1/sessions/{session_id}/frames` is an HTTP fallback for native bridges.
- A bounded queue drops old frames under load, keeping live guidance fresh.
- A segmentation-engine protocol and deterministic mock backend exercise the
  end-to-end contract without a GPU or model checkpoint.
- Result and metrics endpoints expose regions, latency, and dropped-frame counts.

## Run locally

```bash
cp .env.example .env
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
PYTHONPATH=services .venv/bin/uvicorn vision_api.main:app --reload
```

Create a session:

```bash
curl -X POST http://localhost:8000/v1/sessions
```

Then open a WebSocket to the returned session’s `/frames` URL. Send one
`FrameMetadata` JSON message, then the matching JPEG/PNG/WebP bytes. The server
responds with an acknowledgement; read `/results` for segmentation output.

Run the initial test:

```bash
PYTHONPATH=services .venv/bin/python -m unittest discover -s tests -v
```

Docker is also available after creating `.env`:

```bash
docker compose up --build
```
