# PPE Sentinel: agent instructions

Read `docs/ARCHITECTURE.md` before changing anything. It is the contract: layers,
canonical class vocabulary, rule viewpoints, how to add a rule, a detector, a plugin,
the tracker, and how recording and the session record feed the report agent and the
knowledge graph. If you change a boundary, update that file in the same commit.

Quick rules that bite:
- Rules in `core/rules.py` are pure: no model imports, no numpy, no network, no clock.
- SH17 labels a gloved hand `gloves` with no `hands` box under it. `hands` means bare.
- Nothing hosted on the per-frame path. The tools model is a local ONNX in weights/ run by onnxruntime.
- Do not install Roboflow's `inference` package into .venv. Scratch venv only, see docs/ARCHITECTURE.md.
- Slow detectors go behind `CachedPlugin` on a cadence, never inline.
- No batching in the realtime path. The mailbox drops stale frames on purpose.
- No em dashes in code, comments, docs, or commit messages.

Run things:
- `.venv/bin/python smoke.py test_images/<file>.jpg` one image through the spine
- `.venv/bin/python stream_demo.py --no-speak` fake 10 fps feed, proves the debounce
- `.venv/bin/python stream_demo.py --fps 30 --show` smooth overlay window
- `.venv/bin/python app.py` Gradio playground on http://127.0.0.1:7860
- `.venv/bin/uvicorn server:app --port 8000` endpoint the iPhone posts frames to

Setup: `./setup.sh` on a fresh clone (venv, pinned deps from requirements-lock.txt, SH17 weights).
Secrets: `ROBOFLOW_API_KEY` lives in `.env` (gitignored), only needed to fetch a model bundle. Never print it.
