# Relay receiver (Mac)

Minimal receiver used during the hackathon to prove the glasses -> phone -> laptop pipeline.
Accepts frames from the iOS app over `WS /ws/ingest` (also `WS /`), fans them out to browsers on
`WS /ws/view`, serves a live dashboard on `/` (canvas, rotate/fit, latency HUD), and exposes
`POST /inspect` (Claude vision on the latest frame; needs `ANTHROPIC_API_KEY`).

    ./run.sh          # venv + deps, listens on 0.0.0.0:8787, advertises _glassesrelay._tcp via Bonjour

The Bonjour TXT record carries the receiver's current URLs (refreshed every 10 s) so the phone
never needs a typed IP. `--ws-ping-timeout 90` matters: the phone's socket stalls otherwise.
