"""Follow Claude's box between identifications.

Claude returns one box per answer, about two seconds after the wearer holds still, and that box
is a snapshot. This module keeps it alive: an OpenCV CSRT tracker is started on the frame Claude
looked at, catches up through the frames that arrived while Claude was thinking, then follows the
part on every new frame at camera rate. The next Claude answer re-anchors it; a "no part" answer
or a run of failed updates stops it.

All functions are blocking (decode + track, roughly 10 to 40 ms); call them via a thread.
"""
import threading
import time

import cv2
import numpy as np

state = {
    "active": False,
    "ok": False,          # last update succeeded
    "box": None,          # normalized [x1, y1, x2, y2] in frame coordinates
    "id": None,           # catalog id being followed
    "name": None,
    "conf": 0.0,
    "since": 0.0,         # when tracking started
    "updates": 0,
    "lost": 0,            # consecutive failed updates
    "ms": 0.0,
    "catchup_frames": 0,
}
MAX_LOST = 4
_tracker = None
_size = (0, 0)            # (w, h) of the frame the tracker was started on
_lock = threading.Lock()


def _decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bad jpeg")
    return img


def _to_pixels(box, w, h):
    x1, y1, x2, y2 = box
    return (int(x1 * w), int(y1 * h), max(2, int((x2 - x1) * w)), max(2, int((y2 - y1) * h)))


def _to_norm(rect, w, h):
    x, y, rw, rh = rect
    return [round(max(0.0, x / w), 4), round(max(0.0, y / h), 4),
            round(min(1.0, (x + rw) / w), 4), round(min(1.0, (y + rh) / h), 4)]


def start(frame: bytes, box: list[float], later_frames: list[bytes], *, pid: str, name: str, conf: float) -> bool:
    """Anchor on `frame` at `box`, then step through `later_frames` (the ones that arrived while
    Claude was answering) so the box is current when the dashboard first sees it."""
    global _tracker, _size
    t0 = time.perf_counter()
    with _lock:
        img = _decode(frame)
        h, w = img.shape[:2]
        tr = cv2.TrackerCSRT_create()
        tr.init(img, _to_pixels(box, w, h))
        _tracker, _size = tr, (w, h)
        state.update(active=True, ok=True, box=list(box), id=pid, name=name, conf=conf, since=time.time(),
                     updates=0, lost=0, catchup_frames=len(later_frames))
        cur = list(box)
        # catch up on every other frame; CSRT copes with the larger steps and it halves the delay
        for data in later_frames[::2]:
            ok, rect = tr.update(_decode(data))
            if ok:
                cur = _to_norm(rect, w, h)
            else:
                state["lost"] += 1
        state["box"] = cur
        state["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return state["lost"] < MAX_LOST


def update(frame: bytes) -> dict:
    """Advance the tracker one frame. Returns the public state (copy)."""
    t0 = time.perf_counter()
    with _lock:
        if not state["active"] or _tracker is None:
            return dict(state)
        img = _decode(frame)
        h, w = img.shape[:2]
        if (w, h) != _size:
            # frame size changed (phone vs webcam); the tracker's model no longer applies
            _stop_locked()
            return dict(state)
        ok, rect = _tracker.update(img)
        state["updates"] += 1
        if ok:
            state["box"], state["ok"], state["lost"] = _to_norm(rect, w, h), True, 0
        else:
            state["lost"] += 1
            state["ok"] = False
            if state["lost"] >= MAX_LOST:
                _stop_locked()
        state["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return dict(state)


def _stop_locked():
    global _tracker
    _tracker = None
    state.update(active=False, ok=False, box=None)


def stop():
    with _lock:
        _stop_locked()


def public() -> dict:
    return {k: state[k] for k in ("active", "ok", "box", "id", "name", "conf", "updates", "lost", "ms", "catchup_frames")}
