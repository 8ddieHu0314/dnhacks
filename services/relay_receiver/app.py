"""Glasses Inspector — Mac-side receiver for the Ray-Ban Meta camera stream.

The iOS app (GlassesInspector) pushes JPEG frames over a WebSocket to /ws/ingest
(8-byte big-endian capture timestamp in ms, then JPEG bytes). POST /frame is kept as a
fallback. Browsers subscribe to /ws/view and receive every frame as it arrives, plus
JSON stats. POST /inspect runs Claude vision on the latest frame (needs ANTHROPIC_API_KEY).
"""
import asyncio
import base64
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path

import identify
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

import tts_elevenlabs as tts

MODEL = os.environ.get("INSPECT_MODEL", "claude-opus-5")
SPEAK = os.environ.get("SPEAK", "0") == "1"           # also say results on the Mac speaker
FAKE = os.environ.get("INSPECT_FAKE", "0") == "1"      # stream canned text (no API key needed) to test the audio path
REPORT_PATH = Path(__file__).with_name("report.jsonl")
FRAMES_DIR = Path(__file__).with_name("frames")
FRAMES_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a field inspection assistant looking through a technician's smart glasses at an energy or industrial site.
Given a single camera frame, do the following, briefly and in plain language suitable for being read aloud:
1. Name what you are looking at (equipment, component, gauge, panel, material).
2. Read any visible labels, nameplates, gauge values, or warnings verbatim.
3. Flag any hazards you can see: missing PPE, exposed conductors, open panels, leaks, corrosion, damage, unsafe positioning.
4. Give one recommended next action.
Keep it under 80 words. If the frame is unclear, say what is needed (closer, more light, hold still)."""

NARRATION_PROMPT = """You are a live narrator speaking into a technician's smart glasses as they walk an energy or industrial site.
You receive one camera frame every few seconds. Speak as if to the wearer, in one or two short sentences, in plain language that sounds natural read aloud.
Name what is in view, read any labels, gauges, or warnings, and call out hazards first. Mention only what is new or changed compared with your previous narration.
If nothing meaningful changed, reply with exactly: no change"""

app = FastAPI()
print("Catalog:", identify.load_catalog())


def _lan_ips() -> list[str]:
    """All non-loopback IPv4 addresses (Wi-Fi, USB link to the phone, ...)."""
    import subprocess, re
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    return [ip for ip in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out) if not ip.startswith("127.")]


_zc = None
_refresh_task = None
_warm_task = None


async def _warm():
    """Warm the identification prompt cache and pre-render every catalog spoken line."""
    print("Prompt cache:", await identify.warm_cache(), flush=True)
    if tts.enabled():
        lines = []
        for rec in identify.catalog.values():
            lines.append(identify.spoken_short(rec))
            lines.append(identify.spoken_for(rec, ""))
        r, c = await tts.prerender(lines)
        print(f"Voice cache: rendered {r}, already cached {c}", flush=True)


@app.on_event("startup")
async def _advertise():
    global _warm_task
    _warm_task = asyncio.create_task(_warm())
    """Advertise this receiver over Bonjour so the phone app can find it without typing an IP."""
    global _zc
    try:
        import socket
        from zeroconf import ServiceInfo, Zeroconf
        ips = _lan_ips()
        urls = ",".join(f"http://{ip}:8787" for ip in ips)
        _zc = Zeroconf()
        host = socket.gethostname().split('.')[0]
        info = ServiceInfo("_glassesrelay._tcp.local.",
                           f"{host}._glassesrelay._tcp.local.",
                           addresses=[socket.inet_aton(ip) for ip in ips], port=8787,
                           properties={"urls": urls},
                           server=f"{host}.local.")   # SRV target must be a plain hostname for iOS to resolve it
        # A just-killed instance's record can linger and look like a name conflict; retry until it expires.
        for attempt in range(30):
            try:
                await asyncio.to_thread(_zc.register_service, info)
                print(f"Bonjour: advertising _glassesrelay._tcp with {urls}", flush=True)
                break
            except Exception as e:
                print(f"Bonjour register attempt {attempt + 1} failed: {type(e).__name__} {e}", flush=True)
                await asyncio.sleep(3)
        else:
            print("Bonjour: giving up; use the manual URL on the phone", flush=True)
            return

        async def refresh():
            # Addresses change when Wi-Fi hops or the USB cable is replugged; keep the TXT record current.
            current = urls
            while True:
                await asyncio.sleep(10)
                try:
                    ips2 = _lan_ips()
                    urls2 = ",".join(f"http://{ip}:8787" for ip in ips2)
                    if urls2 != current and ips2:
                        info2 = ServiceInfo(info.type, info.name, addresses=[socket.inet_aton(ip) for ip in ips2],
                                            port=8787, properties={"urls": urls2}, server=info.server)
                        await asyncio.to_thread(_zc.update_service, info2)
                        current = urls2
                        print(f"Bonjour: updated to {urls2}")
                except Exception as e:
                    print("Bonjour refresh failed:", e)
        global _refresh_task
        _refresh_task = asyncio.create_task(refresh())
    except Exception as e:  # discovery is a convenience, never fatal
        print("Bonjour advertise failed:", type(e).__name__, e, flush=True)


@app.on_event("shutdown")
async def _unadvertise():
    if _zc:
        _zc.close()

state = {
    "latest": None,        # bytes (JPEG)
    "latest_ts": 0.0,      # server receive time
    "capture_ts": 0.0,     # phone capture time (epoch seconds), 0 if unknown
    "frames": 0,
    "bytes": 0,
    "fps_window": [],
    "lat_window": [],      # phone->mac latency samples (ms)
    "report": [],
}
viewers: set[asyncio.Queue] = set()
viewer_sockets: set[WebSocket] = set()   # for text (stats/caption) pushes
phones: set[WebSocket] = set()           # ingest sockets: frames in, speech text out
caption = {"text": "", "final": True}
narration = {"enabled": False, "interval": 8.0, "busy": False, "last": "", "task": None}
# ElevenLabs sentences queue up here and a single worker streams them to the phone in order.
speech = {"queue": None, "task": None, "next_id": 0, "chars": 0, "errors": 0, "last_error": ""}

if REPORT_PATH.exists():
    state["report"] = [json.loads(l) for l in REPORT_PATH.read_text().splitlines() if l.strip()]


def _ingest(data: bytes, capture_ts: float | None):
    now = time.time()
    state["latest"] = data
    state["latest_ts"] = now
    state["capture_ts"] = capture_ts or 0.0
    state["frames"] += 1
    state["bytes"] += len(data)
    w = state["fps_window"]; w.append(now); del w[:-60]
    if capture_ts:
        lw = state["lat_window"]; lw.append((now - capture_ts) * 1000); del lw[:-60]
    # fan out: keep only the newest frame per viewer
    for q in list(viewers):
        if q.full():
            try: q.get_nowait()
            except asyncio.QueueEmpty: pass
        q.put_nowait(data)


def _stats():
    w = state["fps_window"]
    fps = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else 0.0
    lw = state["lat_window"]
    lat = sum(lw) / len(lw) if lw else None
    age = time.time() - state["latest_ts"] if state["latest"] else None
    return {"frames": state["frames"], "fps": round(fps, 1),
            "phone_to_mac_ms": None if lat is None else round(lat),
            "last_frame_age_s": None if age is None else round(age, 2),
            "frame_kb": round(len(state["latest"]) / 1024, 1) if state["latest"] else 0,
            "viewers": len(viewers), "phones": len(phones), "model": "fake" if FAKE else MODEL,
            "inspections": len(state["report"]), "narration": narration["enabled"], "interval": narration["interval"],
            "reactive": identify.state["enabled"], "reactive_status": identify.state["status"], "last_id": identify.state["last_id"],
            "catalog_parts": len(identify.catalog), "identify_model": identify.MODEL,
            "tts": {"provider": "elevenlabs" if tts.enabled() else "apple", "chars": speech["chars"],
                    "errors": speech["errors"], "last_error": speech["last_error"],
                    "cache_hits": speech.get("cache_hits", 0), "speed": tts.SPEED}}


async def _send_all(sockets: set[WebSocket], msg: dict):
    text = json.dumps(msg)
    for ws in list(sockets):
        try:
            await ws.send_text(text)
        except Exception:
            sockets.discard(ws)


async def _caption(text: str, final: bool):
    caption["text"], caption["final"] = text, final
    await _send_all(viewer_sockets, {"type": "caption", "text": text, "final": final})


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


async def _say(text: str):
    """Send one sentence to the phone. With ElevenLabs configured the phone shows the caption and
    waits for PCM audio; otherwise it speaks the text with Apple's voice as before."""
    if not tts.enabled():
        await _send_all(phones, {"type": "speak", "text": text})
        return
    speech["next_id"] += 1
    uid = speech["next_id"]
    await _send_all(phones, {"type": "speak", "text": text, "id": uid, "audio": True})
    if speech["queue"] is None:
        speech["queue"] = asyncio.Queue()
    speech["queue"].put_nowait((uid, text))
    t = speech.get("task")
    if t is None or t.done():
        speech["task"] = asyncio.create_task(_tts_worker())


def _drop_pending_speech():
    """Forget sentences not yet rendered, so a stop does not get followed by stale audio."""
    q = speech.get("queue")
    if q is None:
        return
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:
            break


async def _tts_worker():
    """Drain the sentence queue one at a time so audio reaches the phone in order."""
    q = speech["queue"]
    while not q.empty():
        uid, text = await q.get()
        sent = 0
        t0 = time.time()
        try:
            pre = tts.cached(text)
            if pre is not None:
                # pre-rendered (catalog line or previously spoken): push it all, no API round trip
                for chunk in tts.chunks(pre):
                    await _send_all(phones, {"type": "audio", "id": uid, "rate": tts.SAMPLE_RATE,
                                             "pcm": base64.b64encode(chunk).decode()})
                    sent += 1
                speech["cache_hits"] = speech.get("cache_hits", 0) + 1
                await _send_all(phones, {"type": "audio_end", "id": uid, "ms": int((time.time() - t0) * 1000), "cached": True})
                q.task_done()
                continue
            buf = b""
            async for chunk in tts.stream_pcm(text):
                if sent == 0:
                    speech["chars"] += len(text)
                buf += chunk
                await _send_all(phones, {"type": "audio", "id": uid, "rate": tts.SAMPLE_RATE,
                                         "pcm": base64.b64encode(chunk).decode()})
                sent += 1
            tts.store(text, buf)
            await _send_all(phones, {"type": "audio_end", "id": uid, "ms": int((time.time() - t0) * 1000)})
        except Exception as e:
            speech["errors"] += 1
            speech["last_error"] = str(e)[:200]
            if sent == 0:
                # nothing played yet: let the phone read it with the built in voice
                await _send_all(phones, {"type": "speak_fallback", "id": uid, "text": text})
            else:
                await _send_all(phones, {"type": "audio_end", "id": uid, "ms": int((time.time() - t0) * 1000)})
        q.task_done()


async def _fake_stream(question: str):
    for chunk in ["Looking at a laptop and a phone on a table. ", "The phone screen shows a camera preview. ",
                  "No hazards visible. ", "Next: hold the frame steady for a closer read."]:
        for word in chunk.split(" "):
            yield word + " "
            await asyncio.sleep(0.08)


async def _claude_stream(question: str, system: str, b64: str):
    import anthropic
    aclient = anthropic.AsyncAnthropic()
    async with aclient.beta.messages.stream(
        model=MODEL, max_tokens=400, system=system,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": question}]}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            yield " The model declined to analyze this frame."


async def analyze(question: str, *, narrate: bool = False, speak: bool = True) -> dict | None:
    """Run Claude on the latest frame, streaming sentences to the phone (spoken) and viewers (caption)."""
    data = state["latest"]
    if not data:
        return None
    if not FAKE and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set on the server; restart with the key exported (or INSPECT_FAKE=1).")
    ts = time.time()
    frame_path = FRAMES_DIR / f"{int(ts)}.jpg"
    frame_path.write_bytes(data)
    b64 = base64.standard_b64encode(data).decode()
    system = NARRATION_PROMPT if narrate else SYSTEM_PROMPT
    q = question
    if narrate and narration["last"]:
        q = f"Previous narration: {narration['last']}\n\n{question}"
    await _caption("", final=False)
    full, pending = "", ""
    gen = _fake_stream(q) if FAKE else _claude_stream(q, system, b64)
    async for piece in gen:
        full += piece
        pending += piece
        await _caption(full, final=False)
        # flush complete sentences to the phone as they land so speech starts early
        parts = _SENTENCE_END.split(pending)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence and speak and sentence.lower() != "no change":
                    await _say(sentence)
            pending = parts[-1]
    tail = pending.strip()
    if tail and speak and tail.lower() != "no change":
        await _say(tail)
    await _send_all(phones, {"type": "speak_end"})
    text = full.strip()
    await _caption(text, final=True)
    if narrate:
        if text.lower() != "no change":
            narration["last"] = text
        if text.lower() == "no change":
            return None
    entry = {"ts": ts, "question": question, "result": text, "frame": frame_path.name,
             "model": "fake" if FAKE else MODEL, "narration": narrate}
    state["report"].append(entry)
    with REPORT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    if SPEAK and text.lower() != "no change":
        subprocess.Popen(["say", text])
    return entry


async def _narration_loop():
    while narration["enabled"]:
        fresh = state["latest"] and (time.time() - state["latest_ts"]) < 3
        if fresh and not narration["busy"]:
            narration["busy"] = True
            try:
                await analyze("Narrate what the wearer is looking at now.", narrate=True)
            except Exception as e:
                await _caption(f"narration error: {e}", final=True)
            finally:
                narration["busy"] = False
        await asyncio.sleep(narration["interval"])


def set_narration(enabled: bool, interval: float | None = None):
    if interval:
        narration["interval"] = max(3.0, float(interval))
    narration["enabled"] = bool(enabled)
    t = narration.get("task")
    if enabled and (t is None or t.done()):
        narration["last"] = ""
        narration["task"] = asyncio.create_task(_narration_loop())


async def handle_phone_command(msg: dict):
    kind = msg.get("type")
    if kind == "inspect":
        if not narration["busy"]:
            narration["busy"] = True
            try:
                await analyze(msg.get("question") or "Inspect this frame.")
            except Exception as e:
                await _say(f"Inspection failed: {e}")
                await _send_all(phones, {"type": "speak_end"})
            finally:
                narration["busy"] = False
    elif kind == "narrate":
        set_narration(msg.get("enabled", False), msg.get("interval"))
        await _send_all(phones, {"type": "narration", "enabled": narration["enabled"], "interval": narration["interval"]})
    elif kind == "reactive":
        set_reactive(msg.get("enabled", False))
        await _send_all(phones, {"type": "reactive", **_reactive_public()})


@app.websocket("/")
async def ws_ingest_root(ws: WebSocket):
    await ws_ingest(ws)


@app.websocket("/ws/ingest")
async def ws_ingest(ws: WebSocket):
    await ws.accept()
    phones.add(ws)
    await ws.send_text(json.dumps({"type": "narration", "enabled": narration["enabled"], "interval": narration["interval"]}))
    await ws.send_text(json.dumps({"type": "reactive", **_reactive_public()}))
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                if len(data) > 8 and data[:2] != b"\xff\xd8":
                    ts_ms = struct.unpack(">Q", data[:8])[0]
                    _ingest(data[8:], ts_ms / 1000.0)
                else:
                    _ingest(data, None)
            elif (text := message.get("text")) is not None:
                try:
                    asyncio.create_task(handle_phone_command(json.loads(text)))
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        phones.discard(ws)


@app.websocket("/ws/view")
async def ws_view(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    viewers.add(q)
    viewer_sockets.add(ws)
    if state["latest"]:
        q.put_nowait(state["latest"])
    await ws.send_text(json.dumps({"type": "caption", "text": caption["text"], "final": caption["final"]}))
    last_stats = 0.0
    try:
        while True:
            try:
                frame = await asyncio.wait_for(q.get(), timeout=1.0)
                await ws.send_bytes(frame)
            except asyncio.TimeoutError:
                pass
            if time.time() - last_stats > 0.5:
                last_stats = time.time()
                await ws.send_text(json.dumps({"type": "stats", **_stats()}))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        viewers.discard(q)
        viewer_sockets.discard(ws)


@app.post("/frame")
async def post_frame(request: Request):
    data = await request.body()
    ts = request.headers.get("x-capture-ts")
    _ingest(data, float(ts) / 1000.0 if ts else None)
    return {"ok": True, "frames": state["frames"]}


@app.get("/latest.jpg")
def latest_jpg():
    data = state["latest"]
    if not data:
        return Response(status_code=204)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return _stats()


@app.post("/inspect")
async def inspect(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    question = (body or {}).get("question") or "Inspect this frame."
    if not state["latest"]:
        return JSONResponse({"error": "no frame yet"}, status_code=409)
    if narration["busy"]:
        return JSONResponse({"error": "analysis already running"}, status_code=429)
    narration["busy"] = True
    try:
        entry = await analyze(question)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    finally:
        narration["busy"] = False
    return entry or {"result": caption["text"]}


@app.post("/narrate")
async def narrate(request: Request):
    body = await request.json()
    set_narration(body.get("enabled", False), body.get("interval"))
    await _send_all(phones, {"type": "narration", "enabled": narration["enabled"], "interval": narration["interval"]})
    return {"enabled": narration["enabled"], "interval": narration["interval"]}


def _reactive_public():
    st = identify.state
    return {"enabled": st["enabled"], "status": st["status"], "last_id": st["last_id"],
            "last_result": st["last_result"], "calls": st["calls"], "catalog": identify.catalog_source,
            "parts": len(identify.catalog), "min_confidence": st["min_confidence"]}


def set_reactive(enabled: bool, **kw):
    identify.set_enabled(enabled, **kw)
    t = identify.state.get("task")
    if enabled and (t is None or t.done()):
        async def on_result(obj):
            # keep the frame so identifications can be reviewed for accuracy afterwards
            frame_name = f"identify-{int(obj['ts'])}.jpg"
            analyzed = obj.pop("_frame", None) or state["latest"]
            if analyzed:
                (FRAMES_DIR / frame_name).write_bytes(analyzed)
            await _send_all(viewer_sockets, {"type": "identified", **{k: v for k, v in obj.items() if k != "record"},
                                             "record": obj.get("record"), "frame": frame_name})
            entry = {"ts": obj["ts"], "question": "reactive identify", "result": obj.get("spoken") or obj.get("name"),
                     "id": obj.get("id"), "confidence": obj.get("confidence"), "evidence": obj.get("evidence"),
                     "frame": frame_name, "model": "fake" if FAKE else MODEL, "narration": True}
            state["report"].append(entry)
            with REPORT_PATH.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        async def speak(text):
            await _say(text)
            await _send_all(phones, {"type": "speak_end"})
        async def speak_stop():
            _drop_pending_speech()
            await _send_all(phones, {"type": "speak_stop"})
        identify.state["task"] = asyncio.create_task(identify.reactive_loop(
            lambda: (state["latest"], state["latest_ts"]), on_result, speak, speak_stop, _caption))


@app.post("/reactive")
async def reactive(request: Request):
    body = await request.json()
    set_reactive(body.get("enabled", False), **{k: body.get(k) for k in ("min_confidence", "settle_seconds", "cooldown_seconds", "short_spoken")})
    await _send_all(phones, {"type": "reactive", **_reactive_public()})
    return _reactive_public()


@app.get("/reactive")
def reactive_status():
    return _reactive_public()


@app.post("/identify")
async def identify_once():
    """One-shot identification of the latest frame (no speech)."""
    if not state["latest"]:
        return JSONResponse({"error": "no frame yet"}, status_code=409)
    try:
        obj = await identify.identify_frame(state["latest"])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    await _send_all(viewer_sockets, {"type": "identified", **{k: v for k, v in obj.items() if k != "record"}, "record": obj.get("record")})
    return obj


@app.post("/catalog/reload")
def catalog_reload():
    return {"loaded": identify.load_catalog()}


@app.get("/narrate")
def narrate_status():
    return {"enabled": narration["enabled"], "interval": narration["interval"], "last": narration["last"]}


@app.get("/report")
def report():
    return list(reversed(state["report"]))


@app.get("/frames/{name}")
def frame_file(name: str):
    p = FRAMES_DIR / Path(name).name
    if not p.exists():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg")


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html><head><meta charset=utf-8><title>Glasses Inspector</title>
<style>
:root{color-scheme:dark}
body{font-family:system-ui;background:#111;color:#eee;margin:0;height:100vh;display:grid;grid-template-columns:minmax(0,1fr) 380px;grid-template-rows:100vh}
#stage{position:relative;min-width:0;min-height:0;background:#000;display:flex;align-items:center;justify-content:center}
#cv{max-width:100%;max-height:100%;display:block}
#hud{position:absolute;left:12px;top:10px;font:12px/1.5 ui-monospace,monospace;color:#9f9;background:rgba(0,0,0,.55);padding:6px 10px;border-radius:8px;white-space:pre}
#tools{position:absolute;right:12px;top:10px;display:flex;gap:6px}
#tools button{background:rgba(255,255,255,.12);color:#eee;border:0;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer}
aside{padding:16px;overflow:auto;border-left:1px solid #333;min-height:0}
button.primary{font-size:16px;padding:10px 16px;background:#2b6;color:#000;border:0;border-radius:8px;cursor:pointer;width:100%}
input{width:100%;box-sizing:border-box;padding:8px;margin:8px 0;background:#222;color:#eee;border:1px solid #444;border-radius:6px}
.e{border-bottom:1px solid #333;padding:10px 0;font-size:14px}.e small{color:#888}.e img{width:100%;border-radius:6px;margin-top:6px}
#out{margin-top:10px;white-space:pre-wrap;font-size:14px}
</style></head><body>
<div id=stage><canvas id=cv></canvas><div id=hud>connecting…</div>
<div id=tools><button onclick="rot=(rot+90)%360;draw()">Rotate</button><button onclick="fit=!fit;draw()">Fit/Fill</button></div></div>
<aside>
<input id=q placeholder="Question (optional)">
<button class=primary onclick="inspect()">Inspect this frame</button>
<div style="display:flex;gap:8px;align-items:center;margin-top:10px">
  <label style="font-size:14px"><input type=checkbox id=react onchange="reactive()"> <b>Reactive identify</b></label>
  <span id=rstat style="font:12px ui-monospace,monospace;color:#9f9"></span>
</div>
<div id=card style="display:none;margin-top:10px;padding:10px;background:#1c1c1c;border:1px solid #333;border-radius:8px;font-size:13px"></div>
<div style="display:flex;gap:8px;align-items:center;margin-top:10px">
  <label style="font-size:14px"><input type=checkbox id=narr onchange="narrate()"> Narrate continuously</label>
  <input id=interval type=number min=3 step=1 value=8 style="width:70px;margin:0" onchange="narrate()"> s
</div>
<div id=out></div>
<h3>Report</h3><div id=rep></div></aside>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),hud=document.getElementById('hud'),stage=document.getElementById('stage');
let bmp=null,rot=0,fit=true,stats={},shown=0,lastShown=performance.now(),dispFps=0;
function draw(){
  if(!bmp)return;
  const rotated=rot%180!==0, iw=rotated?bmp.height:bmp.width, ih=rotated?bmp.width:bmp.height;
  const W=stage.clientWidth,H=stage.clientHeight;
  const s=fit?Math.min(W/iw,H/ih):Math.max(W/iw,H/ih);
  const w=Math.round(iw*s),h=Math.round(ih*s);
  cv.width=Math.min(w,W);cv.height=Math.min(h,H);
  ctx.save();ctx.translate(cv.width/2,cv.height/2);ctx.rotate(rot*Math.PI/180);
  ctx.drawImage(bmp,-bmp.width*s/2,-bmp.height*s/2,bmp.width*s,bmp.height*s);ctx.restore();
}
function connect(){
  const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/view');
  ws.binaryType='blob';
  ws.onmessage=async e=>{
    if(typeof e.data==='string'){const m=JSON.parse(e.data);
      if(m.type==='identified'){const c=document.getElementById('card');c.style.display='block';
        c.innerHTML=`<b>${m.name||'?'}</b> <span style="color:#888">id=${m.id} · ${Math.round((m.confidence||0)*100)}% · id in ${m.timing?m.timing.id_at_ms:'?'} ms, done ${m.timing?m.timing.total_ms:'?'} ms</span><div>${m.spoken||''}</div><div style="color:#888;font-size:12px">${m.evidence||''}</div>`+
        (m.record?`<pre style="white-space:pre-wrap;font-size:11px;color:#bbb;margin:6px 0 0">${JSON.stringify(m.record,null,1).slice(0,1200)}</pre>`:'');return;}
      if(m.type==='caption'){const out=document.getElementById('out');out.textContent=m.text||(m.final?'':'thinking…');out.style.opacity=m.final?1:0.7;if(m.final&&m.text)loadReport();return;}
      stats=m;renderHud();document.getElementById('narr').checked=!!m.narration;
      document.getElementById('react').checked=!!m.reactive;document.getElementById('rstat').textContent=m.reactive?`${m.reactive_status} · last ${m.last_id??'-'} · catalog ${m.catalog_parts} parts`:`catalog ${m.catalog_parts} parts`;return;}
    try{const b=await createImageBitmap(e.data);if(bmp)bmp.close();bmp=b;draw();
      shown++;const now=performance.now();if(now-lastShown>1000){dispFps=shown*1000/(now-lastShown);shown=0;lastShown=now;}}catch(err){}
  };
  ws.onclose=()=>{hud.textContent='disconnected, retrying…';setTimeout(connect,1000)};
}
function renderHud(){
  hud.textContent=`${bmp?bmp.width+'×'+bmp.height:'no frame'} · relay ${stats.fps??'-'} fps · shown ${dispFps.toFixed(1)} fps\\nphone→mac ${stats.phone_to_mac_ms??'-'} ms · ${stats.frame_kb??'-'} KB/frame · frames ${stats.frames??0} · phones ${stats.phones??0} · ${stats.model??''}`;
}
window.addEventListener('resize',draw);
async function inspect(){const out=document.getElementById('out');out.textContent='thinking…';
 const r=await fetch('/inspect',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value||undefined})});
 const j=await r.json();if(j.error)out.textContent=j.error;loadReport();}
async function reactive(){await fetch('/reactive',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({enabled:document.getElementById('react').checked})});}
async function narrate(){await fetch('/narrate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({enabled:document.getElementById('narr').checked,interval:+document.getElementById('interval').value})});}
async function loadReport(){const rep=await (await fetch('/report')).json();
 document.getElementById('rep').innerHTML=rep.map(e=>`<div class=e><small>${new Date(e.ts*1000).toLocaleTimeString()} · ${e.model}</small><div>${e.result}</div><img src="/frames/${e.frame}"></div>`).join('');}
loadReport();connect();
</script></body></html>"""
