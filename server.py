"""Step 1 server: thin FastAPI shim so a phone (Step 2) can POST frames and
poll for the latest verdict. Underneath it is the same Mailbox + StreamRunner
the demo uses, so a phone sending frames faster than the spine can process
never backs the server up; /latest always reflects the newest frame.

Launch: .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.detectors import SH17Detector
from core.spine import Spine, roboflow_tools_plugin
from core.stream import CachedPlugin, Mailbox, Speaker, StreamRunner

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "yolo8s")
CONF = float(os.environ.get("CONF", "0.4"))
DEBOUNCE = float(os.environ.get("DEBOUNCE", "2.0"))
# SPEAK: server-side text-to-speech stand-in (a Mac plays the audio) while a
# phone client would relay it over A2DP to glasses in a later step.
SPEAK = os.environ.get("SPEAK", "1") not in ("0", "false", "False")

app = FastAPI(title="PPE compliance server (Step 1)")

mailbox = Mailbox()
speaker = Speaker(enabled=SPEAK)

_latest_lock = threading.Lock()
_latest = {"result": None, "ts": None}

sh17 = SH17Detector(f"weights/{MODEL_NAME}.pt", device="mps", conf=CONF)
_plugin = CachedPlugin(roboflow_tools_plugin(conf=CONF), every_n_frames=5, ttl_seconds=1.0)
spine = Spine(sh17=sh17, plugins=[(_plugin, 1)], debounce_seconds=DEBOUNCE)


def _on_result(result, ts, latency_ms, dropped) -> None:
    with _latest_lock:
        _latest["result"] = result
        _latest["ts"] = ts
    if result.sentence:
        speaker.say(result.sentence)


runner = StreamRunner(spine, mailbox, _on_result)
_runner_thread = None


@app.on_event("startup")
def _startup() -> None:
    global _runner_thread
    _runner_thread = threading.Thread(target=runner.run, daemon=True)
    _runner_thread.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    runner.stop()


@app.post("/frame")
async def post_frame(request: Request):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("frame")
        if upload is None:
            return JSONResponse({"accepted": False, "error": "missing 'frame' field"}, status_code=400)
        raw = await upload.read()
    else:
        raw = await request.body()

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"accepted": False, "error": "could not decode image"}, status_code=400)

    mailbox.put(frame, time.time())
    return {"accepted": True, "dropped": mailbox.dropped}


@app.get("/latest")
def get_latest():
    with _latest_lock:
        result = _latest["result"]
        ts = _latest["ts"]

    if result is None:
        return JSONResponse({"error": "no frame processed yet"}, status_code=404)

    return {
        "detections": [
            {"class": d.class_name, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box], "source": d.source}
            for d in result.detections
        ],
        "rule_results": [
            {"rule": r.rule, "active": r.active, "applicable": r.applicable, "detail": r.detail}
            for r in result.rule_results
        ],
        "fired": result.fired,
        "sentence": result.sentence,
        "timers": result.timers,
        "latency_ms": result.latency_ms,
        "ts": ts,
    }


@app.get("/health")
def get_health():
    return {"status": "ok", "model": MODEL_NAME}
