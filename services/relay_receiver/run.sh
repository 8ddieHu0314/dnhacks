#!/bin/sh
# Start the Glasses Inspector receiver. Needs ANTHROPIC_API_KEY in the environment for /inspect.
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install -q -r requirements.txt
echo "Mac LAN address: $(ipconfig getifaddr en0)  -> set this in the iOS app relay URL as http://$(ipconfig getifaddr en0):8787"
SPEAK=${SPEAK:-0} exec ./.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8787 --ws websockets --ws-ping-interval 20 --ws-ping-timeout 90 --ws-max-size 33554432
