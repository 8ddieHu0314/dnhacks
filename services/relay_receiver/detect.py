"""Local component detector that runs in front of Claude.

A small YOLOv8 ONNX (Roboflow Universe `arduino-lcxdx/3`, 14 Arduino-kit classes, CC BY 4.0)
runs on every incoming frame with plain onnxruntime. Its boxes are drawn on the dashboard and
used as an early trigger for the Claude identification: as soon as a box is stable for a couple
of frames, Claude gets the crop plus a hint, instead of waiting for the whole scene to settle.

Weights live at <repo>/weights/components_yolov8.onnx with a sibling .classes.txt (one class per
line). Both are gitignored; see README for how to fetch them. If either is missing the detector
disables itself and the relay behaves as before.
"""
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("DETECT_ONNX", REPO_ROOT / "weights" / "components_yolov8.onnx"))
CLASSES = WEIGHTS.with_suffix(".classes.txt")
CONF = float(os.environ.get("DETECT_CONF", "0.45"))
IOU = float(os.environ.get("DETECT_IOU", "0.5"))
ENABLED = os.environ.get("DETECT", "1") == "1"
# Component-only by design: this relay never loads the PPE (SH17 hands/helmet/face) model, and the
# allowlist below keeps the detector to electronic parts even if a broader ONNX is dropped in.
ALLOW = {c.strip() for c in os.environ.get("DETECT_CLASSES", "").split(",") if c.strip()} or None

# Detector label -> most likely catalog id in docs/components/components.json. Only a hint for
# Claude; the catalog match is still its call. Labels are the dataset's (Spanish) class names.
CATALOG_HINT = {
    "arduino": "uno-r3",
    "lcd": "lcd1602",
    "servomotor": "servo-sg90",
    "sensor ultrasonico": "hc-sr04",
    "7-seg": "7seg-1digit or 7seg-4digit",
    "pot": "pot-10k",
    "led": "led-5mm or led-rgb",
    "resistencia": "resistor-assorted",
    "diodo": "diode-1n4007",
    "transistor": "pn2222",
    "pulsador": "button",
    "drv8825": None,     # stepper driver, not in the kit (kit has uln2003-driver)
    "esp82": None,       # ESP8266, not in the kit
    "ttl": None,         # USB-TTL adapter, not in the kit
}
ENGLISH = {"resistencia": "resistor", "diodo": "diode", "pulsador": "button", "servomotor": "servo",
           "sensor ultrasonico": "ultrasonic sensor", "pot": "potentiometer", "7-seg": "7-segment display",
           "esp82": "ESP8266", "ttl": "USB-TTL adapter", "drv8825": "DRV8825 stepper driver"}

state = {
    "available": False,
    "reason": "not loaded",
    "model": WEIGHTS.name,
    "provider": None,
    "classes": [],
    "ms": 0.0,           # last inference time
    "frames": 0,
    "latest": [],        # list of Detection dicts for the latest frame
    "latest_ts": 0.0,    # frame timestamp the boxes belong to
    "latest_frame": None,  # the JPEG those boxes were computed on (so consumers never pair boxes with a newer frame)
    "allow": None,       # class allowlist in effect (None = every class in classes.txt)
    "hints": {},         # label -> {"catalog_id", "display", "in_catalog"} after bind_catalog()
}
_session = None
_input_name = None
_size = 640


def english(label: str) -> str:
    return ENGLISH.get(label, label)


def load() -> str:
    """Create the onnxruntime session. Returns a one-line status for /health."""
    global _session, _input_name, _size
    if not ENABLED:
        state.update(available=False, reason="disabled (DETECT=0)")
        return state["reason"]
    if not WEIGHTS.exists() or not CLASSES.exists():
        state.update(available=False, reason=f"weights missing: {WEIGHTS}")
        return state["reason"]
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]  # CoreML EP splits the graph into 3 partitions and is not faster for yolov8n
        _session = ort.InferenceSession(str(WEIGHTS), sess_options=opts, providers=providers)
        inp = _session.get_inputs()[0]
        _input_name = inp.name
        _size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else 640
        state["classes"] = [l.strip() for l in CLASSES.read_text().splitlines() if l.strip()]
        state["allow"] = sorted(ALLOW & set(state["classes"])) if ALLOW else None
        state["provider"] = _session.get_providers()[0]
        state.update(available=True, reason="ok")
        return f"{WEIGHTS.name} ({len(state['classes'])} classes, {state['provider']})"
    except Exception as e:  # pragma: no cover - environment dependent
        state.update(available=False, reason=f"load failed: {e}")
        return state["reason"]


def bind_catalog(catalog: dict) -> str:
    """Make detector output agree with the catalog: check every hinted id exists and pick the
    catalog's own short name (name_on_kit) as the label the dashboard and the glasses use."""
    hints, missing = {}, []
    for label in state["classes"] or CATALOG_HINT:
        guess = CATALOG_HINT.get(label)
        ids = [g.strip() for g in guess.split(" or ")] if guess else []
        found = [i for i in ids if i in catalog]
        missing += [i for i in ids if i not in catalog]
        if len(found) == 1:
            rec = catalog[found[0]]
            display = rec.get("name_on_kit") or rec.get("canonical_name") or found[0]
            display = str(display).split(" (")[0].strip()
        else:
            display = english(label)
            if found and english(label).lower() == label:      # e.g. "led": several catalog matches
                display = english(label).upper() if len(label) <= 3 else english(label)
        hints[label] = {"catalog_id": found[0] if len(found) == 1 else (" or ".join(found) or None),
                        "display": display, "in_catalog": bool(found)}
    state["hints"] = hints
    for d in state["latest"]:
        d["display"] = hints.get(d["label"], {}).get("display", d["name"])
    note = f"{sum(h['in_catalog'] for h in hints.values())}/{len(hints)} detector classes map to catalog parts"
    return note + (f"; unknown ids: {missing}" if missing else "")


def display_name(label: str) -> str:
    return state["hints"].get(label, {}).get("display") or english(label)


def _preprocess(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    x = np.asarray(img.resize((_size, _size), Image.BILINEAR), dtype=np.float32) / 255.0  # stretch, rgb, /255
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x), w, h


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        ov = inter / np.maximum(a + b - inter, 1e-9)
        order = rest[ov <= iou]
    return keep


def detect(data: bytes, conf: float = CONF) -> list[dict]:
    """Run the detector on one JPEG. Boxes are normalized (0..1) x1,y1,x2,y2 in image coordinates.

    Blocking (about 15 to 40 ms on CPU); call it through an executor from async code."""
    if not state["available"]:
        return []
    t0 = time.perf_counter()
    x, w, h = _preprocess(data)
    out = _session.run(None, {_input_name: x})[0]          # (1, 4+nc, 8400)
    pred = out[0].T                                         # (8400, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls)), cls]
    m = scores >= conf
    dets: list[dict] = []
    if m.any():
        p, cls, scores = pred[m], cls[m], scores[m]
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) / _size  # normalized, stretch undone
        boxes = np.clip(boxes, 0, 1)
        keep = _nms(boxes, scores, IOU)
        names = state["classes"]
        allow = state["allow"]
        for i in keep:
            label = names[cls[i]] if cls[i] < len(names) else str(cls[i])
            if allow is not None and label not in allow:
                continue
            h = state["hints"].get(label, {})
            dets.append({"label": label, "name": english(label), "display": h.get("display") or english(label),
                         "conf": round(float(scores[i]), 3), "box": [round(float(v), 4) for v in boxes[i]],
                         "catalog_hint": h.get("catalog_id", CATALOG_HINT.get(label)), "in_catalog": h.get("in_catalog", False)})
    state["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    state["frames"] += 1
    state["latest"] = dets
    state["latest_frame"] = data
    return dets


def crop(data: bytes, box: list[float], pad: float = 0.25, min_side: int = 224) -> bytes:
    """JPEG crop around a normalized box with padding, so Claude sees the part large and sharp."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) * w, (y2 - y1) * h
    px, py = max(bw * pad, (min_side - bw) / 2, 0), max(bh * pad, (min_side - bh) / 2, 0)
    l, t = int(max(0, x1 * w - px)), int(max(0, y1 * h - py))
    r, b = int(min(w, x2 * w + px)), int(min(h, y2 * h + py))
    if r - l < 32 or b - t < 32:
        return data
    out = io.BytesIO()
    img.crop((l, t, r, b)).save(out, format="JPEG", quality=88)
    return out.getvalue()


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def primary(dets: list[dict]) -> dict | None:
    """The box Claude should look at: highest confidence, ties broken by area (bigger = closer)."""
    if not dets:
        return None
    return max(dets, key=lambda d: (d["conf"], (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])))
