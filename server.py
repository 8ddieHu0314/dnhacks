"""Step 1 server: thin FastAPI shim so a phone (Step 2) can POST frames and
poll for the latest verdict. Underneath it is the same Mailbox + StreamRunner
the demo uses, so a phone sending frames faster than the spine can process
never backs the server up; /latest always reflects the newest frame.

Launch: .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import re
import shutil
import threading
import time
import zipfile

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.detectors import SH17Detector
from core.spine import Spine, roboflow_tools_plugin
from core.stream import CachedPlugin, Mailbox, Recorder, Speaker, StreamRunner

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "yolo8s")
CONF = float(os.environ.get("CONF", "0.4"))
DEBOUNCE = float(os.environ.get("DEBOUNCE", "2.0"))
# SPEAK: server-side text-to-speech stand-in (a Mac plays the audio) while a
# phone client would relay it over A2DP to glasses in a later step.
SPEAK = os.environ.get("SPEAK", "1") not in ("0", "false", "False")
# RECORD: keep a ring buffer of raw frames and dump a clip + event on every
# violation, for the session's audit trail. See docs/ARCHITECTURE.md.
RECORD = os.environ.get("RECORD", "1") not in ("0", "false", "False")
BUFFER_SECONDS = float(os.environ.get("BUFFER_SECONDS", "20.0"))
TAIL_SECONDS = float(os.environ.get("TAIL_SECONDS", "3.0"))
# RECORD_ALL: also save every ingested frame as a JPEG under
# sessions/<id>/frames/ for a training data capture. Off by default.
RECORD_ALL = os.environ.get("RECORD_ALL", "0") not in ("0", "false", "False")

app = FastAPI(title="PPE compliance server (Step 1)")

mailbox = Mailbox()
speaker = Speaker(enabled=SPEAK)
recorder = (
    Recorder(buffer_seconds=BUFFER_SECONDS, tail_seconds=TAIL_SECONDS, record_all=RECORD_ALL)
    if RECORD
    else None
)
if recorder is not None:
    # Lets a browser (or an end-of-session report) hit /clips/<file>.mp4 and
    # /keyframes/<file>.jpg directly. StaticFiles serves HTTP Range requests,
    # which a <video> tag needs to seek/scrub instead of just downloading.
    app.mount("/clips", StaticFiles(directory=recorder.clips_dir), name="clips")
    app.mount("/keyframes", StaticFiles(directory=recorder.keyframes_dir), name="keyframes")

_latest_lock = threading.Lock()
_latest = {"result": None, "ts": None}

sh17 = SH17Detector(f"weights/{MODEL_NAME}.pt", device="mps", conf=CONF)
_plugin = CachedPlugin(roboflow_tools_plugin(conf=CONF), every_n_frames=5, ttl_seconds=1.0)
spine = Spine(sh17=sh17, plugins=[(_plugin, 1)], debounce_seconds=DEBOUNCE)


def _on_result(result, ts, latency_ms, dropped) -> None:
    with _latest_lock:
        _latest["result"] = result
        _latest["ts"] = ts
    if recorder is not None:
        recorder.handle_result(result, ts)
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
    if recorder is not None:
        recorder.shutdown()


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

    ts = time.time()
    if recorder is not None:
        recorder.add_frame(frame, ts)
    mailbox.put(frame, ts)
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


@app.get("/events")
def get_events():
    """The session's fired-event queue so far, oldest first, each with a
    clickable clip_url/keyframe_url for an end-of-session report to replay.
    A clip only finishes encoding once it closes (rule cleared + tail, or
    the stream stopped), so clip_url may 404 for the most recent event(s)
    until then.
    """
    if recorder is None:
        return {"recording": False, "events": []}

    events = []
    for e in recorder.list_events():
        item = dict(e)
        if item.get("clip_path"):
            item["clip_url"] = f"/clips/{os.path.basename(item['clip_path'])}"
        if item.get("keyframe_path"):
            item["keyframe_url"] = f"/keyframes/{os.path.basename(item['keyframe_path'])}"
        events.append(item)
    return {"recording": True, "session_dir": recorder.session_dir, "events": events}


SESSIONS_DIR = "sessions"
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-T]+$")


@app.get("/")
def get_capture_page():
    """Browser capture page: record from a camera, upload the clip, download
    sessions. Camera access needs a secure context, so open it on the Mac at
    http://127.0.0.1:8000/ (an iPhone can join as a Continuity Camera)."""
    return FileResponse(os.path.join(_STATIC_DIR, "capture.html"))


def _extract_frames(video_path: str, frames_dir: str, every_n: int) -> dict:
    """Decode a video and write every Nth frame as frames/<index>.jpg. Runs
    on the request threadpool (sync endpoint), never on the live frame path."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open the uploaded video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total % every_n == 0:
            cv2.imwrite(os.path.join(frames_dir, f"{total:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += 1
        total += 1
    cap.release()
    return {"total_frames": total, "frames_written": written, "fps": fps}


@app.post("/upload")
def post_upload(video: UploadFile = File(...), every_n: int = Form(3)):
    """Save an uploaded clip under sessions/capture_<t>/ and extract frames
    from it for training data. Accepts whatever the browser's MediaRecorder
    or a phone produced (webm, mp4, mov)."""
    every_n = max(1, int(every_n))
    ext = os.path.splitext(video.filename or "")[1].lower() or ".webm"
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
        ext = ".webm"
    session_id = time.strftime("capture_%Y%m%dT%H%M%S")
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    source_path = os.path.join(session_dir, f"source{ext}")
    with open(source_path, "wb") as out:
        shutil.copyfileobj(video.file, out)

    try:
        stats = _extract_frames(source_path, os.path.join(session_dir, "frames"), every_n)
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "session_dir": session_dir}, status_code=400)

    return {
        "session_id": session_id,
        "session_dir": session_dir,
        "source": source_path,
        "every_n": every_n,
        **stats,
        "download_url": f"/sessions/{session_id}/download",
    }


def _dir_stats(path: str) -> tuple:
    n_bytes = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                n_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n_bytes


def _count_files(path: str) -> int:
    return len(os.listdir(path)) if os.path.isdir(path) else 0


@app.get("/sessions")
def list_sessions():
    """Every session folder on disk, newest first, with frame and clip counts."""
    if not os.path.isdir(SESSIONS_DIR):
        return {"sessions": []}
    live_id = recorder.session_id if recorder is not None else None
    out = []
    for sid in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        p = os.path.join(SESSIONS_DIR, sid)
        if not os.path.isdir(p) or not _SESSION_ID_RE.match(sid):
            continue
        out.append(
            {
                "id": sid,
                "live": sid == live_id,
                "frames": _count_files(os.path.join(p, "frames")),
                "clips": _count_files(os.path.join(p, "clips")),
                "bytes": _dir_stats(p),
            }
        )
    return {"sessions": out}


@app.get("/sessions/{session_id}/download")
def download_session(session_id: str):
    """Zip a session folder (frames, clips, keyframes, events, source clip)
    and send it. The zip is rebuilt each time, next to the folder."""
    if not _SESSION_ID_RE.match(session_id):
        return JSONResponse({"error": "bad session id"}, status_code=400)
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return JSONResponse({"error": "no such session"}, status_code=404)
    zip_path = os.path.join(SESSIONS_DIR, f"{session_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(session_dir):
            for f in sorted(files):
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, SESSIONS_DIR))
    return FileResponse(zip_path, media_type="application/zip", filename=f"{session_id}.zip")


@app.get("/health")
def get_health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "session_dir": recorder.session_dir if recorder is not None else None,
        "record_all": bool(recorder is not None and recorder.record_all),
        "frames_written": recorder.frames_written if recorder is not None else 0,
    }
