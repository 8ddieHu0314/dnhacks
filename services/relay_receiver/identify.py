"""Reactive part identification for the glasses.

Watches the incoming frame stream, detects when the wearer has settled on a new scene,
identifies the part in view against a component catalog (Claude vision with the catalog
index in a cached system prompt), and speaks only when the identified part changes.

Catalog: docs/components/components.json at the repo root (teammate's knowledge base).
The loader is tolerant of field names; see `_index_line`.
"""
import asyncio
import base64
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATHS = [REPO_ROOT / "docs" / "components" / "components.json",
                 Path(__file__).with_name("components.json")]

MODEL = os.environ.get("IDENTIFY_MODEL", os.environ.get("INSPECT_MODEL", "claude-sonnet-5"))
FAKE = os.environ.get("INSPECT_FAKE", "0") == "1"
MAX_SIDE = int(os.environ.get("IDENTIFY_MAX_SIDE", "512"))


def _supports_fallbacks(model: str) -> bool:
    """Server-side refusal fallbacks exist only on the largest models."""
    return model.startswith(("claude-opus-5", "claude-fable"))

state = {
    "enabled": False,
    "status": "idle",            # idle | moving | settled | identifying
    "min_confidence": 0.6,
    "settle_seconds": 0.4,
    "motion_threshold": 0.035,   # mean abs diff between consecutive thumbnails
    "change_threshold": 0.06,    # diff vs the last identified scene
    "cooldown_seconds": 1.5,
    "last_id": None,
    "last_result": None,
    "last_det_label": None,      # detector label of the box that triggered the last identification
    "stable_frames": 2,          # detector path: consecutive frames a box must persist before Claude is called
    "det_min_conf": 0.5,         # detector path: ignore boxes below this
    "triggers": {"detector": 0, "settle": 0},
    "last_spoken_at": 0.0,
    "calls": 0,
    "task": None,
}
_prev_thumb = None
_prev_frame_ts = 0.0
_last_motion_at = 0.0
_identified_thumb = None
_busy = False

catalog: dict[str, dict] = {}
catalog_index: str = ""
catalog_source: str = "none"


# ---------------------------------------------------------------- catalog

def _first(rec: dict, *keys, default=""):
    for k in keys:
        v = rec.get(k)
        if v:
            return v
    return default


def _sq(v, n):
    if isinstance(v, (list, tuple)):
        v = "; ".join(str(x) for x in v[:2])
    return re.sub(r"\s+", " ", str(v or "")).strip()[:n]


def _index_line(rec: dict) -> str:
    """One compact line per part for the cached system prompt."""
    d = rec.get("details") or {}
    vis = d.get("visual_identification") or {}
    pid = rec.get("id")
    name = _sq(rec.get("canonical_name") or rec.get("name") or pid, 50)
    text = vis.get("printed_text_to_look_for") or rec.get("mpn") or ""
    look = " ".join(x for x in (vis.get("shape_and_size"), vis.get("color_and_markings")) if x)
    confused = vis.get("easily_confused_with") or []
    line = f"- {pid}: {name} | text: {_sq(text, 70) or 'none'} | looks: {_sq(look, 130) or 'n/a'}"
    if confused:
        line += f" | not: {_sq(confused[:1], 50)}"
    return line


def _words(v, n):
    """Trim to n chars on a word boundary."""
    t = _sq(v, 10_000)
    if len(t) <= n:
        return t
    cut = t[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:(") 


def _first_sentence(v, n=120):
    t = _sq(v, 10_000)
    t = re.split(r"(?<=[.;])\s", t, 1)[0]
    return _words(t, n)


def _spoken_line(rec: dict) -> str:
    """Two short sentences for the wearer: name, pins/voltage/key spec, one caution."""
    d = rec.get("details") or {}
    name = rec.get("canonical_name") or rec.get("name") or rec.get("id")
    name = _words(str(name).split(",")[0].split(" (")[0], 60)
    bits = []
    pc = rec.get("pin_count") or (d.get("pins") or {}).get("pin_count")
    if pc:
        bits.append(f"{pc} pins")
    v = _sq(rec.get("voltage") or (d.get("electrical") or {}).get("operating_voltage"), 200)
    if v and len(v) <= 30 and not v.lower().startswith(("not applicable", "depends", "n/a")):
        bits.append(v)
    ks = rec.get("key_specs")
    if isinstance(ks, str) and ks:
        bits.append(_words(ks, 60))
    caution = ""
    safety = d.get("safety") or {}
    for cand in (safety.get("hazards") or []) + (safety.get("common_mistakes") or []):
        c = str(cand)
        if c and not c.lower().startswith(("none", "not a ", "no ")):
            caution = _first_sentence(c, 120)
            break
    line = f"{name}."
    if bits:
        line += " " + ", ".join(bits) + "."
    if caution:
        caution = caution.rstrip(" .;,:")
        line += " " + caution[0].upper() + caution[1:] + "."
    return line


def load_catalog() -> str:
    global catalog, catalog_index, catalog_source
    for p in CATALOG_PATHS:
        if p.exists():
            raw = json.loads(p.read_text())
            recs = raw.get("components") or raw.get("parts") or raw if isinstance(raw, dict) else raw
            if isinstance(recs, dict):
                recs = [dict(v, id=v.get("id", k)) for k, v in recs.items()]
            catalog = {str(_first(r, "id", "slug", "part_id", "name")): r for r in recs}
            catalog_index = "\n".join(_index_line(r) for r in recs)
            catalog_source = str(p)
            return f"{len(catalog)} parts from {p}"
    catalog, catalog_index, catalog_source = {}, "", "none"
    return "no catalog file found; generic identification"


def spoken_for(rec: dict, fallback: str) -> str:
    s = _first(rec, "spoken", "spoken_summary", "summary_spoken")
    if s:
        return str(s)
    try:
        return _spoken_line(rec)
    except Exception:
        return fallback


# ---------------------------------------------------------------- prompts

def system_prompt() -> str:
    base = ("You identify electronic components seen through a technician's smart glasses, matching against a catalog. "
            "Read any printed markings first; markings beat shape. Never invent specifications. "
            "Reply with JSON only, keys in this order: "
            '{"id": "<catalog id, or null if no catalog part is clearly in view>", "confidence": <0..1>, '
            '"name": "<short name>", "evidence": "<at most 12 words>"}')
    if catalog_index:
        return base + "\n\nCatalog:\n" + catalog_index
    return base + "\n\nNo catalog is loaded: set id to null and name the part generically."


def _downscale(data: bytes, max_side: int = MAX_SIDE) -> bytes:
    img = Image.open(io.BytesIO(data))
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=80)
    return out.getvalue()


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0) if m else text)
    except Exception:
        obj = {"id": None, "name": "unknown", "confidence": 0.0, "evidence": text[:200], "spoken": ""}
    obj.setdefault("id", None)
    obj.setdefault("confidence", 0.0)
    obj.setdefault("spoken", "")
    return obj


def _hint_text(det: dict | None) -> str:
    if not det:
        return "Identify the component in view."
    guess = det.get("catalog_hint")
    hint = f"A local detector flagged this crop as: {det.get('name')} ({det.get('conf', 0):.0%})."
    if guess:
        hint += f" Likely catalog id: {guess}."
    else:
        hint += " That class is not in the catalog; it may be a non-kit part."
    return hint + " Confirm or correct by reading the markings; the detector is only a hint."


async def _call_model(model: str, data: bytes, max_tokens: int = 160, det: dict | None = None):
    """One vision call with the cached catalog prompt. Kwargs are assembled so the
    fallbacks parameter is only sent to models that accept it."""
    import anthropic
    from typing import Any
    client = anthropic.AsyncAnthropic()
    b64 = base64.standard_b64encode(_downscale(data)).decode()
    kwargs: dict[str, Any] = {
        "model": model, "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": _hint_text(det)}]}],
    }
    if _supports_fallbacks(model):
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return await client.beta.messages.create(**kwargs)


_fake_cycle = ["hc-sr04", "rtc-ds3231", "relay-5v"]


async def identify_frame(data: bytes, det: dict | None = None) -> dict:
    """One identification call on a frame. Returns the parsed JSON plus the catalog record.

    With `det` (a detector box, see detect.py) Claude sees the padded crop around the box plus a
    hint with the detector's label and likely catalog id. Without it, the whole frame."""
    import detect
    state["calls"] += 1
    image = detect.crop(data, det["box"]) if det else data
    if FAKE:
        await asyncio.sleep(1.2)
        if det and det.get("catalog_hint") and det["catalog_hint"] in catalog:
            pid = det["catalog_hint"]
        else:
            pid = _fake_cycle[state["calls"] % len(_fake_cycle)]
        obj = {"id": pid, "name": pid.upper(), "confidence": 0.9, "evidence": "fake mode" + (" via detector" if det else ""),
               "spoken": f"This looks like the {pid.upper()} module. Fake mode."}
    else:
        import anthropic
        resp = await _call_model(MODEL, image, det=det)
        text = "".join(b.text for b in resp.content if b.type == "text")
        obj = _parse(text) if resp.stop_reason != "refusal" else {"id": None, "name": "declined", "confidence": 0, "spoken": ""}
        obj["usage"] = {"cache_read": getattr(resp.usage, "cache_read_input_tokens", 0), "out": resp.usage.output_tokens}
    rec = catalog.get(str(obj.get("id"))) if obj.get("id") else None
    if not rec and obj.get("id"):
        # tolerate near-miss ids (e.g. "ds3231" for "rtc-ds3231")
        want = str(obj["id"]).lower()
        for k in catalog:
            if want in k or k in want:
                rec = catalog[k]; obj["id"] = k; break
    obj.setdefault("spoken", "")
    if rec:
        obj["spoken"] = spoken_for(rec, obj.get("spoken") or f"This is the {rec.get('name', obj['id'])}.")
        d = rec.get("details") or {}
        obj["record"] = {k: rec.get(k) for k in ("id", "canonical_name", "name_on_kit", "mpn", "category", "pin_count",
                                                  "pins", "interface", "voltage", "key_specs", "function", "price_usd",
                                                  "confidence") if rec.get(k) is not None}
        obj["record"]["safety"] = (d.get("safety") or {}).get("hazards")
        obj["record"]["wiring"] = ((d.get("wiring_to_uno") or {}).get("example_connections"))
    obj["ts"] = time.time()
    obj["trigger"] = "detector" if det else "settle"
    if det:
        obj["det"] = {k: det[k] for k in ("label", "name", "conf", "box")}
    return obj


# ---------------------------------------------------------------- reactive loop

def _thumb(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L").resize((32, 32))
    return np.asarray(img, dtype=np.float32) / 255.0


def _diff(a, b) -> float:
    if a is None or b is None:
        return 1.0
    return float(np.abs(a - b).mean())


async def reactive_loop(get_latest, on_result, speak, speak_stop, set_caption, get_dets=None):
    """get_latest() -> (bytes, ts) | (None, 0); on_result(obj); speak(text); speak_stop(); set_caption(text);
    get_dets() -> (list[detection], frame_ts) from detect.py, or None when no detector is loaded.

    Two triggers feed the same Claude call:
    - detector: the top box has persisted for `stable_frames` frames and is a different label than the
      last identification (or the scene changed). Fires early, before the wearer holds still, with
      the crop and a hint. This is the optimistic path.
    - settle: no boxes, so fall back to the original rule, identify the whole frame once motion stops.
    """
    global _prev_thumb, _prev_frame_ts, _last_motion_at, _identified_thumb, _busy
    stable_det, stable_n = None, 0
    while state["enabled"]:
        await asyncio.sleep(0.1)
        data, ts = get_latest()
        now = time.time()
        if not data or ts == _prev_frame_ts or now - ts > 2:
            continue
        _prev_frame_ts = ts
        thumb = _thumb(data)
        motion = _diff(thumb, _prev_thumb)
        _prev_thumb = thumb
        scene_changed = _diff(thumb, _identified_thumb) >= state["change_threshold"]

        # ---- detector path
        det = None
        if get_dets is not None:
            import detect
            dets, dts = get_dets()
            dets = [d for d in dets if d["conf"] >= state["det_min_conf"]] if now - dts < 1.0 else []
            top = detect.primary(dets)
            if top and stable_det and top["label"] == stable_det["label"] and detect.iou(top["box"], stable_det["box"]) > 0.4:
                stable_n += 1
            else:
                stable_n = 1 if top else 0
            stable_det = top
            if top and stable_n >= state["stable_frames"]:
                new_part = top["label"] != state["last_det_label"] or scene_changed
                if new_part and not _busy and now - state["last_spoken_at"] >= state["cooldown_seconds"]:
                    det = top
        if det is None:
            # ---- settle path (original behaviour), only when the detector has nothing to offer
            if motion > state["motion_threshold"]:
                _last_motion_at = now
                state["status"] = "moving"
                continue
            if now - _last_motion_at < state["settle_seconds"]:
                continue
            state["status"] = "settled"
            if stable_det is not None:
                continue   # a box is in view but already identified; wait for a new part
            if _busy or not scene_changed:
                continue
            if now - state["last_spoken_at"] < state["cooldown_seconds"]:
                continue
        _busy = True
        state["status"] = "identifying"
        state["triggers"]["detector" if det else "settle"] += 1
        await set_caption(f"Looking… ({det['name']}?)" if det else "Looking…", False)
        try:
            obj = await identify_frame(data, det)
            _identified_thumb = thumb
            state["last_det_label"] = det["label"] if det else None
            state["last_result"] = obj
            obj["_frame"] = data
            await on_result(obj)
            pid = obj.get("id")
            conf = float(obj.get("confidence") or 0)
            if pid and conf >= state["min_confidence"] and pid != state["last_id"]:
                state["last_id"] = pid
                state["last_spoken_at"] = time.time()
                await speak_stop()
                await speak(obj.get("spoken") or f"This is {obj.get('name')}.")
                await set_caption(obj.get("spoken") or obj.get("name", ""), True)
            elif pid and pid == state["last_id"]:
                await set_caption(f"Still {obj.get('name')}.", True)
            else:
                await set_caption(f"Not sure ({obj.get('name')}, {conf:.0%}). {obj.get('evidence', '')}", True)
        except Exception as e:
            await set_caption(f"identify error: {e}", True)
        finally:
            _busy = False
            state["status"] = "settled"
    state["status"] = "idle"


def set_enabled(enabled: bool, **kw):
    global _identified_thumb
    for k in ("min_confidence", "settle_seconds", "motion_threshold", "change_threshold", "cooldown_seconds", "det_min_conf"):
        if kw.get(k) is not None:
            state[k] = float(kw[k])
    if kw.get("stable_frames") is not None:
        state["stable_frames"] = max(1, int(kw["stable_frames"]))
    state["enabled"] = bool(enabled)
    if enabled:
        state["last_id"] = None
        state["last_det_label"] = None
        _identified_thumb = None
