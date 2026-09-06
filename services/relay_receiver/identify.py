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
    "settle_seconds": 0.3,       # wait this long after motion unless the frame is already sharp
    "motion_threshold": 0.035,   # mean abs diff between consecutive thumbnails
    "change_threshold": 0.06,    # diff vs the last identified scene
    "sharp_threshold": 0.010,    # Laplacian variance on a 96px gray; above = usable without settling
    "cooldown_seconds": 0.8,
    "short_spoken": True,        # announce name + one spec; full line stays on Describe
    "mode": "parts",             # parts = catalog lookup; scene = free-text narration on change
    "model": MODEL,              # identification model; switchable at runtime (sonnet/opus)
    "interrupt_confidence": 0.85,  # a different id must be this sure to cut an announcement in progress
    "announce_until": 0.0,       # when the current announcement's audio ends
    "pending": None,             # (pid, rec, conf) waiting for the audio to end
    # detector path (detect.py): a local YOLO box can trigger identification before the wearer holds still
    "last_det_label": None,      # detector label of the box that triggered the last identification
    "stable_frames": 2,          # consecutive frames a box must persist before Claude is called
    "det_min_conf": 0.5,         # ignore boxes below this
    "triggers": {"detector": 0, "settle": 0},
    "preannounce": False,        # speak the detector's catalog name into the glasses before Claude answers (off: rarely fires, can contradict Claude)
    "agreement": {"agree": 0, "disagree": 0, "no_hint": 0},   # Claude's id vs the detector's catalog hint
    "last_detected": None,       # the box announced most recently (for the dashboard banner)
    "last_id": None,
    "last_result": None,
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
            "Read any printed markings first; markings beat shape. Never invent specifications.\n"
            "Answer in exactly two lines and nothing else:\n"
            "line 1: <catalog id, or none> <confidence 0-1>\n"
            "line 2: <at most 12 words of evidence: markings, shape, color, pins>\n"
            "line 3: one spoken sentence for the wearer, under 16 words, natural and conversational, like a "
            "colleague glancing over. The name is already being spoken, so never state what the part is called or "
            "what kind of part it is; begin with what it does, or one rating, or one caution, or something notable "
            "about this particular unit. Say units in words (five volts). If line 1 is none, leave line 3 empty.")
    if catalog_index:
        return base + "\n\nCatalog:\n" + catalog_index
    return base + "\n\nNo catalog is loaded: answer 'none 0' and name the part on line 2."


def _downscale(data: bytes, max_side: int = MAX_SIDE) -> bytes:
    img = Image.open(io.BytesIO(data))
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=80)
    return out.getvalue()


_LINE1 = re.compile(r"^\s*([A-Za-z0-9_\-\.]+)\s+([01](?:\.\d+)?)\s*$")


def _parse_line1(line: str):
    m = _LINE1.match(line)
    if not m:
        return None
    pid, conf = m.group(1), float(m.group(2))
    if pid.lower() in ("none", "null", "unknown", "-"):
        pid = None
    return pid, conf


def _parse(text: str) -> dict:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    first = _parse_line1(lines[0]) if lines else None
    if first is None:
        # tolerate the old JSON shape
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0) if m else text)
            obj.setdefault("id", None); obj.setdefault("confidence", 0.0); obj.setdefault("evidence", "")
            return obj
        except Exception:
            return {"id": None, "confidence": 0.0, "name": "unknown", "evidence": text[:120]}
    pid, conf = first
    return {"id": pid, "confidence": conf, "evidence": " ".join(lines[1:2]).strip()[:160], "remark": remark[:240]}


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


def _kwargs(model: str, data: bytes, max_tokens: int = 60, det: dict | None = None):
    """Request kwargs; the fallbacks parameter is only sent to models that accept it.
    With `det` (a detector box, see detect.py) Claude sees the padded crop plus a hint."""
    from typing import Any
    if det:
        import detect
        data = detect.crop(data, det["box"])
    b64 = base64.standard_b64encode(_downscale(data)).decode()
    kwargs: dict[str, Any] = {
        "model": model, "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": _hint_text(det)}]}],
    }
    remark = " ".join(lines[2:]).strip() if len(lines) > 2 else ""
    if _supports_fallbacks(model):
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return kwargs


async def _call_model(model: str, data: bytes, max_tokens: int = 60, det: dict | None = None):
    import anthropic
    client = anthropic.AsyncAnthropic()
    return await client.beta.messages.create(**_kwargs(model, data, max_tokens, det))


_SENT_END = re.compile(r"(?<=[.!?])\s+")


async def _stream_model(model: str, data: bytes, on_first_line, max_tokens: int = 110, det: dict | None = None,
                        on_sentence=None) -> tuple[str, object]:
    """Stream the answer; call on_first_line(pid, conf) as soon as line 1 is complete, and
    on_sentence(text) for each completed sentence of line 3 (the conversational remark)."""
    import anthropic
    client = anthropic.AsyncAnthropic()
    text, fired, spoken_sent = "", False, 0
    async with client.beta.messages.stream(**_kwargs(model, data, max_tokens, det)) as stream:
        async for piece in stream.text_stream:
            text += piece
            if not fired and "\n" in text:
                parsed = _parse_line1(text.split("\n", 1)[0])
                if parsed:
                    fired = True
                    await on_first_line(*parsed)
        final = await stream.get_final_message()
    if not fired:
        parsed = _parse_line1(text.strip().split("\n", 1)[0]) if text.strip() else None
        if parsed:
            await on_first_line(*parsed)
    return text, final


_fake_cycle = ["hc-sr04", "rtc-ds3231", "relay-5v"]


def resolve(pid):
    """Catalog record for an id, tolerating near misses ("ds3231" -> "rtc-ds3231")."""
    if not pid:
        return None, None
    rec = catalog.get(str(pid))
    if rec:
        return pid, rec
    want = str(pid).lower()
    for k in catalog:
        if want in k or k in want:
            return k, catalog[k]
    return pid, None


def spoken_short(rec: dict) -> str:
    d = rec.get("details") or {}
    name = _words(str(rec.get("canonical_name") or rec.get("name") or rec.get("id")).split(",")[0].split(" (")[0], 50)
    bits = []
    pc = rec.get("pin_count") or (d.get("pins") or {}).get("pin_count")
    if pc:
        bits.append(f"{pc} pins")
    v = _sq(rec.get("voltage") or (d.get("electrical") or {}).get("operating_voltage"), 200)
    if v and len(v) <= 20 and not v.lower().startswith(("not applicable", "depends", "n/a")):
        bits.append(v)
    return f"{name}." + (f" {', '.join(bits)}." if bits else "")


async def identify_frame(data: bytes, on_early=None, det: dict | None = None, on_sentence=None) -> dict:
    """One identification call on a frame. `on_early(pid, conf, rec)` fires as soon as the id
            if on_sentence and text.count("\n") >= 2:
                tail = text.split("\n", 2)[2]
                parts = _SENT_END.split(tail)
                while len(parts) - 1 > spoken_sent and spoken_sent < 2:
                    sent = parts[spoken_sent].strip()
                    spoken_sent += 1
                    if sent:
                        await on_sentence(sent)
    is known (before the evidence line finishes). With `det` Claude sees the box crop plus a
        if on_sentence and text.count("\n") >= 2:
            tail = text.split("\n", 2)[2]
            parts = [x.strip() for x in _SENT_END.split(tail) if x.strip()]
            for sent in parts[spoken_sent:2]:
                await on_sentence(sent)
    hint (detector path). Returns the parsed result plus the record."""
    state["calls"] += 1
    t0 = time.time()
    early = {"pid": None, "conf": 0.0, "rec": None, "at": None}

    async def first_line(pid, conf):
        pid, rec = resolve(pid)
        early.update(pid=pid, conf=conf, rec=rec, at=time.time() - t0)
        if on_early:
            await on_early(pid, conf, rec)

    if FAKE:
        await asyncio.sleep(0.6)
        if det and det.get("catalog_hint") and det["catalog_hint"] in catalog:
            pid = det["catalog_hint"]
        else:
            pid = _fake_cycle[state["calls"] % len(_fake_cycle)]
        await first_line(pid, 0.9)
        obj = {"id": pid, "confidence": 0.9, "evidence": "fake mode" + (" via detector" if det else "")}
        usage = None
    else:
        text, final = await _stream_model(state["model"], data, first_line, det=det, on_sentence=on_sentence)
        obj = _parse(text) if final.stop_reason != "refusal" else {"id": None, "confidence": 0, "evidence": "declined"}
        usage = {"cache_read": getattr(final.usage, "cache_read_input_tokens", 0), "out": final.usage.output_tokens}
    pid, rec = resolve(obj.get("id"))
    obj["id"] = pid
    obj["name"] = (rec or {}).get("canonical_name") or (rec or {}).get("name") or (pid or "no catalog part")
    obj["spoken"] = spoken_for(rec, "") if rec else ""
    obj["spoken_short"] = spoken_short(rec) if rec else ""
    if rec:
        d = rec.get("details") or {}
        obj["record"] = {k: rec.get(k) for k in ("id", "canonical_name", "name_on_kit", "mpn", "category", "pin_count",
                                                  "pins", "interface", "voltage", "key_specs", "function", "price_usd",
                                                  "confidence") if rec.get(k) is not None}
        obj["record"]["safety"] = (d.get("safety") or {}).get("hazards")
        obj["record"]["wiring"] = ((d.get("wiring_to_uno") or {}).get("example_connections"))
    obj["usage"] = usage
    obj["timing"] = {"id_at_ms": int((early["at"] or 0) * 1000), "total_ms": int((time.time() - t0) * 1000)}
    obj["ts"] = time.time()
    obj["trigger"] = "detector" if det else "settle"
    if det:
        obj["det"] = {k: det.get(k) for k in ("label", "name", "display", "conf", "box", "catalog_hint")}
        hint = det.get("catalog_hint")
        if not hint:
            obj["agrees"] = None
            state["agreement"]["no_hint"] += 1
        else:
            obj["agrees"] = bool(obj.get("id")) and str(obj["id"]) in {h.strip() for h in str(hint).split(" or ")}
            state["agreement"]["agree" if obj["agrees"] else "disagree"] += 1
    return obj


# ---------------------------------------------------------------- reactive loop

def _gray(data: bytes, size: int) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


def _thumb(data: bytes) -> np.ndarray:
    return _gray(data, 32)


def _sharpness(data: bytes) -> float:
    """Variance of a Laplacian on a 96px gray image; motion blur drives it toward zero."""
    g = _gray(data, 96)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _diff(a, b) -> float:
    if a is None or b is None:
        return 1.0
    return float(np.abs(a - b).mean())


def _interrupt_policy(pid, conf, now):
    """Decide what to do with a fresh id while an announcement may still be playing.
    Returns 'announce', 'interrupt', 'pending' or 'ignore'."""
    if not pid or conf < state["min_confidence"]:
        return "ignore"
    if pid == state["last_id"]:
        return "ignore"
    if now >= state["announce_until"]:
        return "announce"          # nothing playing: say it
    if conf >= state["interrupt_confidence"]:
        return "interrupt"         # very sure it's a different part: cut in
    return "pending"               # probably a misread mid-motion: wait for the audio to end


async def reactive_loop(get_latest, on_result, speak, speak_stop, set_caption, narrate=None,
                        get_dets=None, on_detected=None):
    """get_latest() -> (bytes, ts) | (None, 0); on_result(obj); speak(text) -> seconds of audio;
    speak_stop(); set_caption(text, final); narrate(data) -> spoken text or None (scene mode);
    get_dets() -> (list[detection], frame_ts, frame_bytes) from detect.py, or None without a detector;
    on_detected(det) fires the instant a box becomes the trigger, before Claude is called.

    Parts mode has two triggers feeding the same Claude call:
    - detector: the top box has persisted `stable_frames` frames and is a new label (or the scene
      changed). Fires early, before the wearer holds still, with the crop and a hint.
    - change: no usable box, so identify the whole frame once it is sharp or settled.
    The detector path always identifies the frame its boxes were computed on."""
    global _prev_thumb, _prev_frame_ts, _last_motion_at, _identified_thumb, _busy
    stable_det, stable_n, last_dts = None, 0, 0.0
    while state["enabled"]:
        await asyncio.sleep(0.05)
        now = time.time()
        # A pending part whose announcement was deferred: say it once the audio has ended,
        # unless something else has been announced since.
        if state["pending"] and now >= state["announce_until"] and not _busy:
            pid, rec, conf = state["pending"]
            state["pending"] = None
            if pid != state["last_id"]:
                state["last_id"] = pid
                state["last_spoken_at"] = now
                line = spoken_short(rec) if state["short_spoken"] else spoken_for(rec, "")
                secs = await speak(line)
                state["announce_until"] = time.time() + (secs or 0)
                await set_caption(line, True)
        data, ts = get_latest()
        if not data or ts == _prev_frame_ts or now - ts > 2:
            continue
        _prev_frame_ts = ts
        thumb = _thumb(data)
        motion = _diff(thumb, _prev_thumb)
        _prev_thumb = thumb
        if motion > state["motion_threshold"]:
            _last_motion_at = now
            state["status"] = "moving"
        changed = _diff(thumb, _identified_thumb) >= state["change_threshold"]

        # ---- detector path (parts mode): a stable box triggers before the wearer holds still
        det = None
        if get_dets is not None and state["mode"] == "parts":
            import detect
            dets, dts, det_frame = get_dets()
            if dts != last_dts and det_frame is not None and now - dts < 1.0:
                last_dts = dts
                dets = [d for d in dets if d["conf"] >= state["det_min_conf"]]
                top = detect.primary(dets)
                if top and stable_det and top["label"] == stable_det["label"] and detect.iou(top["box"], stable_det["box"]) > 0.4:
                    stable_n += 1
                else:
                    stable_n = 1 if top else 0
                stable_det = top
                if top and stable_n >= state["stable_frames"]:
                    det_thumb = thumb if dts == ts else _thumb(det_frame)
                    det_changed = _diff(det_thumb, _identified_thumb) >= state["change_threshold"]
                    new_part = top["label"] != state["last_det_label"] or det_changed
                    if new_part and not _busy and now - state["last_spoken_at"] >= state["cooldown_seconds"]:
                        det = top
                        data, thumb = det_frame, det_thumb   # identify the frame the box belongs to
        if det is None:
            if not changed or _busy or now - state["last_spoken_at"] < state["cooldown_seconds"]:
                if motion <= state["motion_threshold"]:
                    state["status"] = "settled"
                continue
            if stable_det is not None and state["mode"] == "parts":
                continue   # a box is in view but already identified; wait for a new part
            # New scene: go immediately if the frame is sharp, otherwise wait for it to settle.
            settled = now - _last_motion_at >= state["settle_seconds"]
            if not settled and _sharpness(data) < state["sharp_threshold"]:
                continue
        _busy = True
        state["status"] = "identifying" if state["mode"] == "parts" else "describing"
        state["triggers"]["detector" if det else "settle"] += 1
        if det:
            state["last_detected"] = {**{k: det.get(k) for k in ("label", "display", "conf", "box", "catalog_hint", "in_catalog")}, "ts": now}
            if on_detected is not None:
                await on_detected(det)
        await set_caption(f"Looking… ({det['display']}?)" if det else "Looking…", False)
        try:
            if state["mode"] == "scene":
                _identified_thumb = thumb
                text = await narrate(data) if narrate else None
                state["last_result"] = {"mode": "scene", "text": text, "ts": time.time()}
                if text:
                    state["last_spoken_at"] = time.time()
                    state["announce_until"] = time.time() + len(text) / 12.0
                    await set_caption(text, True)
                else:
                    await set_caption("no change", True)
                continue

            decided = {"action": None}

            async def on_early(pid, conf, rec):
                # Hybrid announcement: the part's name the instant the id is known (instant, fixed),
                # then Claude's short conversational remark streams in right behind it.
                action = _interrupt_policy(pid, conf, time.time())
                decided["action"] = action
                if action in ("announce", "interrupt") and rec:
                    if action == "interrupt":
                        await speak_stop()
                    state["last_id"] = pid
                    state["last_spoken_at"] = time.time()
                    state["pending"] = None
                    name = str(rec.get("name_on_kit") or rec.get("canonical_name") or pid).split(",")[0].split(" (")[0]
                    line = f"{name}." if state["short_spoken"] else spoken_for(rec, "")
                    secs = await speak(line)
                    state["announce_until"] = time.time() + (secs or 0)
                    decided["spoken"].append(line)
                    await set_caption(line, False)
                elif action == "pending" and rec:
                    state["pending"] = (pid, rec, conf)

            async def on_sentence(sent):
                # Only after we announced this part; never for repeats, pending, or "none".
                if decided["action"] in ("announce", "interrupt") and len(decided["spoken"]) < 3:
                    secs = await speak(sent)
                    state["announce_until"] = max(state["announce_until"], time.time()) + (secs or 0)
                    decided["spoken"].append(sent)
                    await set_caption(" ".join(decided["spoken"]), False)

            obj = await identify_frame(data, on_early=on_early, det=det, on_sentence=on_sentence)
            _identified_thumb = thumb
            state["last_det_label"] = det["label"] if det else None
            state["last_result"] = obj
            obj["_frame"] = data
            obj["action"] = decided["action"]
            await on_result(obj)
            pid, conf = obj.get("id"), float(obj.get("confidence") or 0)
            if decided["action"] in ("announce", "interrupt"):
                await set_caption(" ".join(decided["spoken"]), True)
            elif decided["action"] == "pending":
                await set_caption(f"Maybe {obj.get('name')} ({conf:.0%}), waiting for audio to finish.", True)
            elif pid and pid == state["last_id"]:
                await set_caption(f"Still {obj.get('name')}.", True)
            else:
                await set_caption(f"Not sure ({obj.get('name')}, {conf:.0%}). {obj.get('evidence', '')}", True)
        except Exception as e:
            await set_caption(f"error: {e}", True)
        finally:
            _busy = False
            state["status"] = "settled"
    state["status"] = "idle"


def set_enabled(enabled: bool, **kw):
    global _identified_thumb
    for k in ("min_confidence", "settle_seconds", "motion_threshold", "change_threshold", "cooldown_seconds", "sharp_threshold"):
        if kw.get(k) is not None:
            state[k] = float(kw[k])
    if kw.get("short_spoken") is not None:
        state["short_spoken"] = bool(kw["short_spoken"])
    if kw.get("mode") in ("parts", "scene"):
        state["mode"] = kw["mode"]
    if kw.get("model"):
        state["model"] = str(kw["model"])
    if kw.get("interrupt_confidence") is not None:
        state["interrupt_confidence"] = float(kw["interrupt_confidence"])
    if kw.get("det_min_conf") is not None:
        state["det_min_conf"] = float(kw["det_min_conf"])
    if kw.get("stable_frames") is not None:
        state["stable_frames"] = max(1, int(kw["stable_frames"]))
    if kw.get("preannounce") is not None:
        state["preannounce"] = bool(kw["preannounce"])
    state["enabled"] = bool(enabled)
    if enabled:
        state["last_id"] = None
        state["last_det_label"] = None
        state["pending"] = None
        _identified_thumb = None


async def warm_cache(models=("claude-sonnet-5", "claude-opus-5")):
    """One tiny call per model so the first real identification finds the catalog prompt cached."""
    if FAKE or not os.environ.get("ANTHROPIC_API_KEY") or not catalog_index:
        return "skipped"
    img = Image.new("RGB", (64, 64), (0, 0, 0)); out = io.BytesIO(); img.save(out, "JPEG")
    parts = []
    for m in models:
        try:
            resp = await _call_model(m, out.getvalue(), max_tokens=8)
            parts.append(f"{m}: write={getattr(resp.usage, 'cache_creation_input_tokens', 0)} read={getattr(resp.usage, 'cache_read_input_tokens', 0)}")
        except Exception as e:
            parts.append(f"{m}: failed {e}")
    return "; ".join(parts)
