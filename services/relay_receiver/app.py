"""Glasses Inspector — Mac-side receiver for the Ray-Ban Meta camera stream, reactive edition.

The iOS app pushes JPEG frames over a WebSocket to /ws/ingest (8-byte big-endian capture
timestamp in ms, then JPEG bytes) and the wearer's spoken questions as {"type":"ask"} text.
Nothing here analyzes frames on its own: the relay keeps a short ring buffer of recent frames,
and when a question arrives it picks the sharpest frames from the moment the wearer was speaking,
sends them to Claude with the whole part catalog in a cached system prompt, and streams the
answer back to the phone sentence by sentence to be spoken in the glasses.

Browsers subscribe to /ws/view for the live picture, the detector's boxes (a visual only), and
captions. POST /frame is kept as an ingest fallback (laptop webcam demo).
"""
import asyncio
import base64
import io
import json
import os
import re
import struct
import subprocess
import time
from collections import deque
from pathlib import Path

import catalog
import detect
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

import tts_elevenlabs as tts

MODEL = os.environ.get("MODEL", os.environ.get("ASK_MODEL", "claude-sonnet-5"))   # one model for everything
ADVERTISE = os.environ.get("ADVERTISE", "1") == "1"   # ADVERTISE=0: no Bonjour, for a second test instance on another port
SPEAK = os.environ.get("SPEAK", "0") == "1"           # also `say` answers on the Mac speaker
FAKE = os.environ.get("INSPECT_FAKE", "0") == "1"      # canned answers (no API key needed) to test the audio path
FRAMES_PER_ASK = int(os.environ.get("FRAMES_PER_ASK", "3"))   # sharpest frames sent with each question
STALE_SECONDS = float(os.environ.get("STALE_SECONDS", "8"))    # newest frame older than this: say so instead of answering
FRAME_MAX_SIDE = int(os.environ.get("FRAME_MAX_SIDE", "1280"))   # the High stream is 720x1280; keep it whole
CROP_CONF = float(os.environ.get("CROP_CONF", "0.25"))   # detector threshold for the close-up crop (a wrong crop only adds an image)
REPORT_PATH = Path(__file__).with_name("report.jsonl")
FRAMES_DIR = Path(__file__).with_name("frames")
FRAMES_DIR.mkdir(exist_ok=True)

TEACH_PROMPT = """You are Inspector, a voice assistant speaking into the smart glasses of an apprentice technician on their first days in a workshop. They ask questions out loud about the electronic parts in front of them, and you answer as a patient senior colleague glancing over their shoulder.

What you receive: a few camera frames from the glasses taken while they were speaking (the last one is the most recent), their question, and the workshop's part catalog below. The camera sees what the wearer sees, so "on my left" means the left of the image, "top right" the top right of the image, and so on.

Rules:
- First, silently match what you see against the quick index (shape, color, markings, connector, wires), then answer from that part's full record. The parts on this bench come from the kit in the catalog, so a kit part is the likely answer even when a detail differs (a connector or label color, a variant marking).
- The catalog is the ground truth. Take pin counts, pinouts, voltages, ratings, wiring and safety facts from the matching record, never from memory. If nothing in the catalog matches, say so in a few words and answer only what the image shows.
- There may be several parts in view. Work out which one the question refers to from the position words or the description; when it is genuinely ambiguous, say what you see and ask which one they mean.
- Answer in one or two short sentences. Use up to four only when the question truly needs it (a full pinout, a wiring sequence, a comparison of several parts).
- Plain spoken language: no lists, no markdown, no dashes, no headings. Say units and numbers in words a person would say (five volts, three pins). Do not repeat the question.
- If an earlier exchange is included, it is the same session a moment ago: resolve "it" and "that one" from it, and do not restate a part's name you already gave unless the part in view has changed.
- Read markings in the image when they matter (a printed part number, a value). If the frames are too blurry or too far away to answer, say what they should do (bring it closer, hold still, turn it over).
- Identify a part from its overall look; once identified, take details you cannot verify in the picture (wire count, pin count, ratings) from its catalog record rather than from what you think you see at this distance.
- Never invent a specification. If the catalog record marks a field uncertain, say it is unconfirmed."""

app = FastAPI()
print("Catalog:", catalog.load(), flush=True)


def _lan_ips() -> list[str]:
    """All non-loopback IPv4 addresses (Wi-Fi, USB link to the phone, ...)."""
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    return [ip for ip in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out) if not ip.startswith("127.")]


_zc = None
_refresh_task = None
_warm_task = None


def _system_blocks() -> list[dict]:
    """The cached prefix: teaching rules plus the whole catalog. Identical bytes every call, so
    Anthropic's prompt cache serves it after the first request."""
    text = TEACH_PROMPT
    if catalog.prompt_text:
        text += ("\n\nQuick index of the catalog (match what you see here first):\n" + catalog.index_text
                 + "\n\nFull catalog records (ground truth):\n\n" + catalog.prompt_text)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


# Sonnet 5 thinks by default. THINKING=off trades reasoning for the first word (a thinking block
# also ate the whole output budget before any text once, which is what emptied two answers);
# THINKING=adaptive with EFFORT=low keeps a little reasoning for the visual match.
THINKING_MODE = os.environ.get("THINKING", "off")
EFFORT = os.environ.get("EFFORT", "low")
THINKING = {"type": "disabled"} if THINKING_MODE == "off" else {"type": "adaptive"}
OUTPUT_CONFIG = None if THINKING_MODE == "off" else {"effort": EFFORT}
MAX_TOKENS = 400 if THINKING_MODE == "off" else 4000     # room for the thinking block plus the answer


async def _warm_once() -> str:
    import anthropic
    client = anthropic.AsyncAnthropic()
    kw = {"output_config": OUTPUT_CONFIG} if OUTPUT_CONFIG else {}
    r = await client.messages.create(model=MODEL, max_tokens=1, system=_system_blocks(), thinking=THINKING,
                                     messages=[{"role": "user", "content": "ping"}], **kw)
    u = r.usage
    return f"write={getattr(u, 'cache_creation_input_tokens', 0)} read={getattr(u, 'cache_read_input_tokens', 0)}"


async def _warm():
    """Keep the catalog prompt in Anthropic's cache: one tiny call at startup and one every four
    minutes (the cache lives five), so the first question after a lull is not the slow one."""
    if FAKE or not os.environ.get("ANTHROPIC_API_KEY"):
        print("Prompt cache: skipped", flush=True)
        return
    while True:
        try:
            print("Prompt cache:", await _warm_once(), flush=True)
        except Exception as e:
            print("Prompt cache warm failed:", e, flush=True)
        await asyncio.sleep(240)


@app.on_event("startup")
async def _start_detector():
    print("component detector:", detect.load(), flush=True)
    if detect.state["available"]:
        detector["task"] = asyncio.create_task(_detect_loop())


@app.on_event("startup")
async def _advertise():
    global _warm_task
    _warm_task = asyncio.create_task(_warm())
    """Advertise this receiver over Bonjour so the phone app can find it without typing an IP."""
    global _zc
    if not ADVERTISE:
        print("Bonjour: disabled (ADVERTISE=0)", flush=True)
        return
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
    "recent": deque(maxlen=200),   # (receive_ts, jpeg) ring buffer, the source of frames for a question
    "frames": 0,
    "bytes": 0,
    "fps_window": [],
    "lat_window": [],      # phone->mac latency samples (ms)
    "report": [],
    "busy": False,         # one answer at a time
    "last_qa": None,       # (ts, question, answer): short-term memory for follow-ups ("does it need a driver?")
    "gaps": [],            # recent frame gaps > 2 s: {"at", "gap_s", "while_speaking"}; freeze telemetry
    "speaking_until": 0.0, # when the last spoken sentence is expected to finish playing
}
MEMORY_SECONDS = 120
viewers: set[asyncio.Queue] = set()
viewer_sockets: set[WebSocket] = set()   # for text (stats/caption) pushes
phones: set[WebSocket] = set()           # ingest sockets: frames in, speech text out
caption = {"text": "", "final": True}
detector = {"task": None, "last_ts": 0.0, "enabled": os.environ.get("DETECT_DEFAULT", "0") == "1"}   # boxes are a dashboard visual only
speech = {"queue": None, "task": None, "next_id": 0, "chars": 0, "errors": 0, "last_error": "",
          "stop_gen": 0}   # bumped by hush(): cuts the answer being streamed and the sentence being rendered
voice_in = {"asks": 0, "last": None, "intents": {"hush": 0, "answer": 0, "no_frame": 0}}
voice = {"provider": "apple"}   # phone or dashboard can switch to elevenlabs
report_page = {"enabled": False}   # judges' view; switched from the phone's gear menu, default off

if REPORT_PATH.exists():
    state["report"] = [json.loads(l) for l in REPORT_PATH.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- frames in

def _ingest(data: bytes, capture_ts: float | None):
    now = time.time()
    # Freeze telemetry: a gap in frames from the phone, and whether the glasses were talking.
    if state["latest_ts"] and now - state["latest_ts"] > 2.0:
        gap = {"at": round(state["latest_ts"], 1), "gap_s": round(now - state["latest_ts"], 1),
               "while_speaking": state["latest_ts"] < state["speaking_until"]}
        state["gaps"] = (state["gaps"] + [gap])[-20:]
        print(f"frame gap: {gap['gap_s']} s ending now" + (" (glasses were speaking)" if gap["while_speaking"] else ""), flush=True)
    state["latest"] = data
    state["latest_ts"] = now
    state["capture_ts"] = capture_ts or 0.0
    state["frames"] += 1
    state["bytes"] += len(data)
    state["recent"].append((now, data))
    w = state["fps_window"]; w.append(now); del w[:-60]
    if capture_ts:
        lw = state["lat_window"]; lw.append((now - capture_ts) * 1000); del lw[:-60]
    # fan out: keep only the newest frame per viewer
    for q in list(viewers):
        if q.full():
            try: q.get_nowait()
            except asyncio.QueueEmpty: pass
        q.put_nowait(data)


def _sharpness(data: bytes) -> float:
    """Variance of a Laplacian on a 96 px gray thumbnail; motion blur drives it toward zero."""
    g = np.asarray(Image.open(io.BytesIO(data)).convert("L").resize((96, 96)), dtype=np.float32) / 255.0
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def select_frames(since: float, k: int = FRAMES_PER_ASK) -> list[tuple[float, bytes]]:
    """The k sharpest frames spread over the window [since, now], oldest first. The window is the
    time the wearer was speaking, when they were looking at the thing they asked about. Falls
    back to the newest frame when the buffer is short."""
    now = time.time()
    since = min(max(since, now - 10.0), now - 0.5)   # phone clock skew or a stale timestamp must not empty the window
    recent = [(ts, d) for ts, d in state["recent"] if ts >= since - 0.3]
    if len(recent) < k:
        recent = list(state["recent"])[-k:]   # thin window (stream just started, or a late timestamp): newest k
    if len(recent) <= k:
        return recent
    # Score at most ~30 candidates evenly spaced over the window, then the sharpest per slice.
    step = max(1, len(recent) // 30)
    cands = recent[::step]
    if cands[-1] is not recent[-1]:
        cands.append(recent[-1])
    scored = [(ts, d, _sharpness(d)) for ts, d in cands]
    t0, t1 = scored[0][0], scored[-1][0]
    span = max(t1 - t0, 1e-3)
    picks = []
    for i in range(k):
        lo, hi = t0 + span * i / k, t0 + span * (i + 1) / k
        in_slice = [s for s in scored if lo <= s[0] <= hi] or [min(scored, key=lambda s: abs(s[0] - (lo + hi) / 2))]
        best = max(in_slice, key=lambda s: s[2])
        if not picks or best[0] != picks[-1][0]:
            picks.append(best)
    return [(ts, d) for ts, d, _ in picks]


def _downscale(data: bytes, max_side: int = FRAME_MAX_SIDE) -> bytes:
    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_side:
        return data
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=85)
    return out.getvalue()


# ---------------------------------------------------------------- status

def _frame_size():
    if not state["latest"]:
        return None
    try:
        return list(Image.open(io.BytesIO(state["latest"])).size)
    except Exception:
        return None


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
            "frame_px": _frame_size(),
            "recent": len(state["recent"]), "gaps": state["gaps"][-10:],
            "viewers": len(viewers), "phones": len(phones), "model": "fake" if FAKE else MODEL,
            "thinking": THINKING_MODE, "effort": EFFORT if OUTPUT_CONFIG else None,
            "frames_per_ask": FRAMES_PER_ASK, "busy": state["busy"],
            "inspections": len(state["report"]),
            "catalog_parts": len(catalog.records), "catalog_kb": len(catalog.prompt_text) // 1024,
            "voice_in": voice_in, "report_page": report_page["enabled"],
            "detector": {"available": detect.state["available"], "enabled": detector["enabled"], "model": detect.state["model"],
                         "reason": detect.state["reason"], "ms": detect.state["ms"], "boxes": len(detect.state["latest"]),
                         "frames": detect.state["frames"]},
            "tts": {"provider": _voice_active(), "available": tts.enabled(), "chars": speech["chars"],
                    "errors": speech["errors"], "last_error": speech["last_error"],
                    "cache_hits": speech.get("cache_hits", 0), "speed": tts.SPEED}}


def _status_msg() -> dict:
    """What the phone needs to know; sent on connect (its hello) and after every change."""
    return {"type": "status", "prompt_version": 7, "thinking": THINKING_MODE, "effort": EFFORT if OUTPUT_CONFIG else None,
            "voice": _voice_active(), "detector_enabled": detector["enabled"],
            "detector": detect.state["available"], "report_page": report_page["enabled"],
            "model": "fake" if FAKE else MODEL, "catalog_parts": len(catalog.records), "busy": state["busy"]}


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


# ---------------------------------------------------------------- speech out

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def _voice_active() -> str:
    return "elevenlabs" if (tts.enabled() and voice["provider"] == "elevenlabs") else "apple"


def speech_seconds(text: str) -> float:
    """How long this sentence will take to play: exact for cached ElevenLabs audio, else estimated."""
    if _voice_active() == "elevenlabs":
        pre = tts.cached(text)
        if pre is not None:
            return len(pre) / (tts.SAMPLE_RATE * 2)
        return len(text) / 12.0
    return len(text) / 14.0   # Apple voice at default rate


async def _say(text: str) -> float:
    """Send one sentence to the phone and return its approximate playback length in seconds.
    With ElevenLabs active the phone shows the caption and waits for PCM audio; otherwise it
    speaks the text with Apple's voice."""
    secs = speech_seconds(text)
    state["speaking_until"] = max(state["speaking_until"], time.time()) + secs
    if _voice_active() != "elevenlabs":
        await _send_all(phones, {"type": "speak", "text": text})
        return secs
    speech["next_id"] += 1
    uid = speech["next_id"]
    await _send_all(phones, {"type": "speak", "text": text, "id": uid, "audio": True})
    if speech["queue"] is None:
        speech["queue"] = asyncio.Queue()
    speech["queue"].put_nowait((uid, text, speech["stop_gen"]))
    t = speech.get("task")
    if t is None or t.done():
        speech["task"] = asyncio.create_task(_tts_worker())
    return secs


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
    """Drain the sentence queue one at a time so audio reaches the phone in order. A hush bumps
    speech["stop_gen"]: sentences queued before it are skipped and the one being streamed is cut."""
    q = speech["queue"]
    while not q.empty():
        uid, text, gen = await q.get()
        if gen != speech["stop_gen"]:
            q.task_done()
            continue
        sent = 0
        t0 = time.time()
        try:
            pre = tts.cached(text)
            if pre is not None:
                for chunk in tts.chunks(pre):
                    if gen != speech["stop_gen"]:
                        break
                    await _send_all(phones, {"type": "audio", "id": uid, "rate": tts.SAMPLE_RATE,
                                             "pcm": base64.b64encode(chunk).decode()})
                    sent += 1
                speech["cache_hits"] = speech.get("cache_hits", 0) + 1
                await _send_all(phones, {"type": "audio_end", "id": uid, "ms": int((time.time() - t0) * 1000), "cached": True})
                q.task_done()
                continue
            buf = b""
            complete = True
            async for chunk in tts.stream_pcm(text):
                if gen != speech["stop_gen"]:
                    complete = False
                    break
                if sent == 0:
                    speech["chars"] += len(text)
                buf += chunk
                await _send_all(phones, {"type": "audio", "id": uid, "rate": tts.SAMPLE_RATE,
                                         "pcm": base64.b64encode(chunk).decode()})
                sent += 1
            if complete:
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


async def hush():
    """Stop what is being said: cut the answer being streamed and the sentence being rendered,
    drop queued sentences, stop playback on the phone."""
    speech["stop_gen"] += 1
    _drop_pending_speech()
    await _send_all(phones, {"type": "speak_stop"})


# ---------------------------------------------------------------- the question

async def _fake_stream(question: str, n_frames: int):
    for chunk in [f"I can see {n_frames} frames of a table with a couple of parts. ",
                  "The one on the left is an ultrasonic sensor with four pins. ",
                  "Power it from five volts and keep the wiring short."]:
        for word in chunk.split(" "):
            yield word + " "
            await asyncio.sleep(0.08)


def _closeup(frames: list[bytes]) -> tuple[bytes | None, dict | None]:
    """A padded crop of the part the local detector finds in the sharpest frame. Small parts held at
    arm's length are only a hundred pixels wide in the full frame; the crop gives Claude a magnified
    view of markings and wires. Runs on demand (about 100 ms), whatever the dashboard toggle says."""
    if not detect.state["available"] or not frames:
        return None, None
    # detect.detect() also records its result as the "latest" boxes for the dashboard; these
    # low-threshold boxes on older frames are not that, so put the dashboard state back after.
    saved = {k: detect.state[k] for k in ("latest", "latest_frame", "latest_ts", "almost")}
    try:
        # Sharpest frame first, but fall through: the detector is at about 40% recall, so the box
        # is often in one of the other frames.
        for frame in sorted(frames, key=_sharpness, reverse=True):
            try:
                top = detect.primary(detect.detect(frame, conf=CROP_CONF))
            except Exception:
                return None, None
            if top:
                return detect.crop(frame, top["box"], pad=0.35, min_side=384), top
        return None, None
    finally:
        detect.state.update(saved)


async def _claude_stream(question: str, frames: list[bytes], closeup: bytes | None = None, det: dict | None = None):
    import anthropic
    client = anthropic.AsyncAnthropic()
    content = []
    for data in frames:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(_downscale(data)).decode()}})
    when = "one frame" if len(frames) == 1 else f"{len(frames)} frames taken while they spoke, oldest first"
    note = ""
    if closeup:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(closeup).decode()}})
        note = (f" The final image is a close-up crop of the part a local detector found in the sharpest frame"
                f" ({det['conf']:.0%} confidence); use it to read shape, markings and wires.")
    content.append({"type": "text", "text": f"({when}.{note})\nThe wearer asked: {question}"})
    messages = []
    last = state.get("last_qa")
    if last and time.time() - last[0] < MEMORY_SECONDS and last[2]:
        # The previous exchange, text only (no frames), so "it" resolves and names are not repeated.
        messages += [{"role": "user", "content": f"(Earlier, {int(time.time() - last[0])} seconds ago.)\nThe wearer asked: {last[1]}"},
                     {"role": "assistant", "content": last[2]}]
    messages.append({"role": "user", "content": content})
    kw = {"output_config": OUTPUT_CONFIG} if OUTPUT_CONFIG else {}
    async with client.messages.stream(model=MODEL, max_tokens=MAX_TOKENS, system=_system_blocks(), thinking=THINKING,
                                      messages=messages, **kw) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            yield " I cannot answer that one."


async def answer(question: str, *, since: float | None = None, source: str = "phone") -> dict:
    """Answer one spoken question from the frames the wearer was looking at, speaking each
    sentence as it lands. Returns the report entry."""
    if not FAKE and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set on the server; restart with the key exported (or INSPECT_FAKE=1).")
    ts = time.time()
    picked = select_frames(since if since else ts - 2.0)
    frames = [d for _, d in picked]
    names = []
    for i, (fts, d) in enumerate(picked):
        name = f"ask-{int(ts)}-{i}.jpg"
        (FRAMES_DIR / name).write_bytes(d)
        names.append(name)
    await _caption(f"You: {question}", final=False)
    closeup, det = (None, None) if FAKE else await asyncio.to_thread(_closeup, frames)
    if closeup:
        (FRAMES_DIR / f"ask-{int(ts)}-crop.jpg").write_bytes(closeup)
        names.append(f"ask-{int(ts)}-crop.jpg")
    full, pending, secs = "", "", 0.0
    stop_gen = speech["stop_gen"]      # a hush() while streaming ends this answer early
    stopped = False
    stream = _fake_stream(question, len(frames)) if FAKE else _claude_stream(question, frames, closeup, det)
    t_first = None
    async for piece in stream:
        if speech["stop_gen"] != stop_gen:
            stopped = True
            break
        if t_first is None:
            t_first = time.time()
        full += piece
        pending += piece
        await _caption(full, final=False)
        parts = _SENTENCE_END.split(pending)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    secs += await _say(sentence)
            pending = parts[-1]
    tail = pending.strip()
    if tail and not stopped:
        secs += await _say(tail)
    await _send_all(phones, {"type": "speak_end"})
    text = full.strip()
    if text and not stopped:
        state["last_qa"] = (time.time(), question, text)
    await _caption(text + (" [stopped]" if stopped else ""), final=True)
    entry = {"ts": ts, "kind": "ask", "question": question, "result": text, "frame": names[0] if names else None,
             "frames": names, "frame_span_s": round(picked[-1][0] - picked[0][0], 2) if len(picked) > 1 else 0.0,
             "closeup": bool(closeup), "det_conf": det["conf"] if det else None,
             "model": "fake" if FAKE else MODEL, "source": source, "speech_s": round(secs, 1),
             "first_token_ms": int((t_first - ts) * 1000) if t_first else None,
             "total_ms": int((time.time() - ts) * 1000)}
    if stopped:
        entry["stopped"] = True
    state["report"].append(entry)
    with REPORT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    if SPEAK and text:
        subprocess.Popen(["say", text])
    return entry


_STOP_WORDS = ("stop", "hush", "quiet", "shut up", "enough", "silence", "cancel")


async def handle_ask(text: str, source: str = "phone", heard_at: float | None = None) -> dict | None:
    """A sentence the wearer said (recognized on the phone). Stop words silence the glasses;
    everything else is a question answered from the frames of the moment they spoke.
    `heard_at` is the epoch second the phone first heard them; it anchors the frame window."""
    text = (text or "").strip()
    if not text:
        return None
    voice_in["asks"] += 1
    voice_in["last"] = {"text": text, "ts": time.time(), "source": source}
    await _send_all(viewer_sockets, {"type": "heard", "text": text, "ts": time.time()})
    low = re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()
    if len(low.split()) <= 3 and any(w in low for w in _STOP_WORDS):
        voice_in["intents"]["hush"] += 1
        await hush()
        await _caption("", final=True)
        return {"intent": "hush", "text": text}
    if not state["latest"] or time.time() - state["latest_ts"] > STALE_SECONDS:
        voice_in["intents"]["no_frame"] += 1
        await _say("I have no camera picture yet." if not state["latest"] else "I have no recent picture; is the camera streaming?")
        await _send_all(phones, {"type": "speak_end"})
        return {"intent": "no_frame", "text": text}
    # A new question outranks the answer still playing: cut it, then wait briefly for it to unwind.
    await hush()
    for _ in range(80):
        if not state["busy"]:
            break
        await asyncio.sleep(0.1)
    if state["busy"]:
        await _say("One moment, still answering the last one.")
        await _send_all(phones, {"type": "speak_end"})
        return {"intent": "busy", "text": text}
    state["busy"] = True
    await _send_all(phones, _status_msg())
    try:
        voice_in["intents"]["answer"] += 1
        entry = await answer(text, since=heard_at, source=source)
    except Exception as e:
        await _say(f"Sorry, that failed: {e}")
        await _send_all(phones, {"type": "speak_end"})
        return {"intent": "error", "text": text, "error": str(e)}
    finally:
        state["busy"] = False
        await _send_all(phones, _status_msg())
    return {"intent": "answer", **entry}


# ---------------------------------------------------------------- detector (dashboard visual)

def set_detector(enabled: bool):
    """Runtime switch for the local YOLO detector. It only draws boxes on the dashboard now; it
    never triggers Claude. Off by default because it costs about 100 ms of CPU per frame."""
    detector["enabled"] = bool(enabled) and detect.state["available"]
    if not detector["enabled"]:
        detect.state["latest"] = []
        detect.state["latest_frame"] = None


async def _detect_loop():
    """Run the local detector on every new frame (latest wins) and push boxes to the dashboards."""
    while True:
        await asyncio.sleep(0.02)
        if not detector["enabled"]:
            if detect.state["latest"] or detect.state["latest_frame"] is not None:
                detect.state.update(latest=[], latest_frame=None)
                await _send_all(viewer_sockets, {"type": "detections", "ts": time.time(), "ms": 0, "boxes": [], "off": True})
            continue
        ts = state["latest_ts"]
        data = state["latest"]
        if not data or ts == detector["last_ts"]:
            continue
        detector["last_ts"] = ts
        try:
            dets = await asyncio.to_thread(detect.detect, data)
        except Exception as e:
            detect.state.update(available=False, reason=f"inference failed: {e}")
            await _send_all(viewer_sockets, {"type": "detections", "ts": ts, "boxes": [], "error": str(e)})
            return
        detect.state["latest_ts"] = ts
        if viewer_sockets:
            await _send_all(viewer_sockets, {"type": "detections", "ts": ts, "ms": detect.state["ms"], "boxes": dets,
                                             "almost": detect.state.get("almost")})


# ---------------------------------------------------------------- phone socket

async def handle_phone_command(msg: dict):
    kind = msg.get("type")
    if kind == "ask":
        heard = msg.get("heard_at")
        await handle_ask(str(msg.get("text") or ""), source="phone",
                         heard_at=float(heard) / 1000.0 if heard else None)
    elif kind == "hush":
        voice_in["intents"]["hush"] += 1
        await hush()
        await _caption("", final=True)
    elif kind == "inspect":
        # The Describe button: the same path as a spoken question, with a default question.
        await handle_ask(msg.get("question") or "What am I looking at? Name the parts you can see.", source="phone")
    elif kind == "detector":
        set_detector(msg.get("enabled", False))
        await _send_all(phones, _status_msg())
        await _send_all(viewer_sockets, {"type": "detections", "ts": time.time(), "ms": 0, "boxes": [], "off": not detector["enabled"]})
    elif kind == "report_page":
        report_page["enabled"] = bool(msg.get("enabled", False))
        await _send_all(phones, _status_msg())
        await _send_all(viewer_sockets, {"type": "report_page", "enabled": report_page["enabled"]})
    elif kind == "voice":
        if msg.get("provider") in ("apple", "elevenlabs"):
            voice["provider"] = msg["provider"]
        await _send_all(phones, _status_msg())


@app.websocket("/")
async def ws_ingest_root(ws: WebSocket):
    await ws_ingest(ws)


@app.websocket("/ws/ingest")
async def ws_ingest(ws: WebSocket):
    await ws.accept()
    # A phone that reconnects (cable retry, app relaunch) can leave its previous socket open until
    # the ping timeout; speech would then go to both and play twice. Keep one socket per client host.
    host = ws.client.host if ws.client else None
    for old in [p for p in phones if p is not ws and p.client and p.client.host == host]:
        phones.discard(old)
        try:
            await old.close(code=1000)
        except Exception:
            pass
    phones.add(ws)
    await ws.send_text(json.dumps(_status_msg()))   # the hello; the phone answers with its saved preferences
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


# ---------------------------------------------------------------- HTTP

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


async def _ask_http(text: str, heard_at: float | None, source: str):
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    if not state["latest"] or time.time() - state["latest_ts"] > STALE_SECONDS:
        return JSONResponse({"error": "no recent frame; is the camera streaming?"}, status_code=409)
    result = await handle_ask(text, source=source, heard_at=heard_at)
    if result and result.get("intent") == "busy":
        return JSONResponse({"error": "analysis already running"}, status_code=429)
    if result and result.get("intent") == "error":
        return JSONResponse(result, status_code=503)
    return result or {}


@app.post("/ask")
async def ask(request: Request):
    """What the phone sends when the wearer speaks; also handy from curl or the dashboard.
    Body: {"text": ..., "heard_at"?: epoch ms when the wearer started speaking}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    heard = (body or {}).get("heard_at")
    return await _ask_http(str((body or {}).get("text") or "").strip(),
                           float(heard) / 1000.0 if heard else None, source="http")


@app.post("/inspect")
async def inspect(request: Request):
    """Kept for the dashboard button and old scripts: a question with a default text."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    q = (body or {}).get("question") or "What am I looking at? Name the parts you can see."
    return await _ask_http(q, None, source="http")


@app.post("/detector")
async def post_detector(request: Request):
    """Runtime on/off for the local detector (dashboard boxes only)."""
    body = await request.json()
    set_detector(body.get("enabled", False))
    await _send_all(phones, _status_msg())
    await _send_all(viewer_sockets, {"type": "detections", "ts": time.time(), "ms": 0, "boxes": [], "off": not detector["enabled"]})
    return {"enabled": detector["enabled"], "available": detect.state["available"]}


@app.post("/voice")
async def set_voice(request: Request):
    body = await request.json()
    if body.get("provider") in ("apple", "elevenlabs"):
        voice["provider"] = body["provider"]
    await _send_all(phones, _status_msg())
    return {"voice": _voice_active(), "elevenlabs_available": tts.enabled()}


@app.get("/status")
def status():
    return _status_msg()


@app.get("/debug/tasks")
def debug_tasks():
    """Health of the background loops (detector, speech, prompt warmer)."""
    def info(t):
        if t is None:
            return None
        if not t.done():
            return "running"
        return f"done exc={t.exception()!r}" if not t.cancelled() else "cancelled"
    return {"detector": info(detector["task"]), "speech": info(speech.get("task")), "warm": info(_warm_task)}


@app.get("/detections")
def detections():
    """Latest local detector boxes (normalized x1,y1,x2,y2) plus detector status."""
    return {"ts": detect.state["latest_ts"], "ms": detect.state["ms"], "boxes": detect.state["latest"],
            "almost": detect.state.get("almost"), "available": detect.state["available"], "enabled": detector["enabled"],
            "reason": detect.state["reason"], "model": detect.state["model"], "classes": detect.state["classes"]}


@app.post("/catalog/reload")
async def catalog_reload():
    loaded = catalog.load()
    await _send_all(phones, _status_msg())
    return {"loaded": loaded, "prompt_kb": len(catalog.prompt_text) // 1024}


@app.get("/catalog/prompt")
def catalog_prompt():
    """The exact text Claude gets as ground truth, for checking what the catalog says."""
    return Response(content=_system_blocks()[0]["text"], media_type="text/plain; charset=utf-8")


@app.get("/report")
def report():
    return list(reversed(state["report"]))


@app.post("/report_page")
async def set_report_page(request: Request):
    body = await request.json()
    report_page["enabled"] = bool(body.get("enabled", False))
    await _send_all(phones, _status_msg())
    await _send_all(viewer_sockets, {"type": "report_page", "enabled": report_page["enabled"]})
    return report_page


@app.get("/report.html", response_class=HTMLResponse)
def report_page_view():
    """Judges' view: every question the apprentice asked, what the glasses saw, and the answer."""
    if not report_page["enabled"]:
        return HTMLResponse("<!doctype html><html><body style='font-family:system-ui;background:#0f0f0f;color:#eee;padding:40px'>"
                            "<h2>Session report is off</h2><p>Turn on “Session report page” in the Glasses Inspector gear menu on the phone.</p></body></html>", status_code=404)
    rows = list(reversed(state["report"]))
    def esc(x):
        return (str(x or "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    items = []
    for r in rows:
        when = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
        frames = r.get("frames") or ([r["frame"]] if r.get("frame") else [])
        pics = "".join(f'<img src="/frames/{esc(f)}" loading="lazy">' for f in frames) or '<div class=nof>no frame</div>'
        timing = f' · first word {r["first_token_ms"]} ms' if r.get("first_token_ms") else ""
        items.append(f'''<article><div class=pics>{pics}</div><div class=body>
  <div class=meta><span class=badge>{esc(r.get("kind") or "ask")}</span> {when} · {esc(r.get("model"))}{timing}{" · stopped" if r.get("stopped") else ""}</div>
  {"<div class=q>“" + esc(r.get("question")) + "”</div>" if r.get("question") else ""}
  <div class=text>{esc(r.get("result"))}</div>
</div></article>''')
    return f'''<!doctype html><html><head><meta charset=utf-8><title>Glasses Inspector report</title>
<style>
body{{font-family:system-ui;background:#0f0f0f;color:#eee;margin:0;padding:24px;max-width:1100px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#999;margin-bottom:18px;font-size:14px}}
a{{color:#7cf}} .grid{{display:flex;flex-direction:column;gap:12px}}
article{{display:grid;grid-template-columns:auto 1fr;gap:14px;background:#181818;border:1px solid #2a2a2a;border-radius:10px;padding:12px}}
.pics{{display:flex;gap:4px}} .pics img{{width:110px;height:146px;object-fit:cover;border-radius:6px;background:#000}}
.nof{{width:110px;height:146px;background:#222;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#666;font-size:12px}}
.meta{{color:#999;font-size:13px;margin-bottom:6px}} .badge{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;background:#6cf;color:#000;margin-right:4px}}
.q{{color:#fc6;font-size:15px;margin-bottom:4px}} .text{{font-size:16px;line-height:1.4}}
@media print{{body{{background:#fff;color:#000}} article{{background:#fff;border-color:#ccc}} .meta{{color:#555}}}}
</style></head><body>
<h1>Glasses Inspector — session report</h1>
<div class=sub>{len(rows)} questions · generated {time.strftime("%Y-%m-%d %H:%M")} · <a href="/report">JSON</a> · <a href="/">live dashboard</a></div>
<div class=grid>{"".join(items) or "<p>No questions yet.</p>"}</div>
</body></html>'''


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
body{font-family:system-ui;background:#111;color:#eee;margin:0;height:100vh;display:grid;grid-template-columns:minmax(0,1fr) 400px;grid-template-rows:100vh}
#stage{position:relative;min-width:0;min-height:0;background:#000;display:flex;align-items:center;justify-content:center}
#cv{max-width:100%;max-height:100%;display:block}
#hud{position:absolute;left:12px;top:10px;font:12px/1.5 ui-monospace,monospace;color:#9f9;background:rgba(0,0,0,.55);padding:6px 10px;border-radius:8px;white-space:pre}
#tools{position:absolute;right:12px;top:10px;display:flex;gap:6px}
#tools button{background:rgba(255,255,255,.12);color:#eee;border:0;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer}
#detbar{position:absolute;left:12px;bottom:10px;font:12px/1.4 ui-monospace,monospace;color:#fc6;background:rgba(0,0,0,.55);padding:5px 10px;border-radius:8px}
#detbar b{color:#6cf}
aside{padding:16px;overflow:auto;border-left:1px solid #333;min-height:0}
button.primary{font-size:16px;padding:10px 16px;background:#2b6;color:#000;border:0;border-radius:8px;cursor:pointer;width:100%}
button.secondary{font-size:14px;padding:10px 12px;background:#333;color:#eee;border:0;border-radius:8px;cursor:pointer}
input{width:100%;box-sizing:border-box;padding:8px;margin:8px 0;background:#222;color:#eee;border:1px solid #444;border-radius:6px}
.e{border-bottom:1px solid #333;padding:10px 0;font-size:14px}.e small{color:#888}.e .q{color:#fc6}.e .pics{display:flex;gap:4px;margin-top:6px}.e .pics img{width:32%;border-radius:6px}
#out{margin-top:10px;white-space:pre-wrap;font-size:15px;line-height:1.4}
#heard{margin-top:8px;color:#fc6;font-size:14px}
</style></head><body>
<div id=stage><canvas id=cv></canvas><div id=hud>connecting…</div>
<div id=tools><button onclick="rot=(rot+90)%360;draw()">Rotate</button><button onclick="fit=!fit;draw()">Fit/Fill</button><button id=boxbtn onclick="showBoxes=!showBoxes;this.style.opacity=showBoxes?1:.5;draw()">Boxes</button></div>
<div id=detbar>detector: loading…</div></div>
<aside>
<input id=q placeholder="Ask as if speaking into the glasses: how many pins does the part on the left have?">
<div style="display:flex;gap:8px"><button class=primary onclick="ask()" style="flex:1">Ask</button>
<button class=secondary onclick="describe()" title="What am I looking at? Name the parts you can see.">What's here?</button>
<button class=secondary onclick="hushNow()" title="Stop speaking">Hush</button></div>
<div style="display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap">
  <b style="font-size:14px">Voice</b>
  <select id=voice onchange="setVoice()" style="background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:4px">
    <option value="apple">Apple (on phone)</option><option value="elevenlabs">ElevenLabs</option></select>
  <label style="font-size:13px;color:#ccc" title="Local YOLO detector: boxes drawn on the video. It never triggers Claude."><input type=checkbox id=det onchange="setDetector()"> detector boxes</label>
  <span id=rstat style="font:12px ui-monospace,monospace;color:#9f9"></span>
</div>
<div id=heard></div>
<div id=out></div>
<h3>Questions <a id=replink href="/report.html" target=_blank style="font-size:13px;font-weight:normal;display:none">open judges' view</a> <a href="/catalog/prompt" target=_blank style="font-size:13px;font-weight:normal">catalog text</a></h3><div id=rep></div></aside>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),hud=document.getElementById('hud'),stage=document.getElementById('stage');
let bmp=null,rot=0,fit=true,stats={},shown=0,lastShown=performance.now(),dispFps=0;
let dets=[],detMs=0,showBoxes=true,almost=null;   // local detector boxes (normalized)
function draw(){
  if(!bmp)return;
  const rotated=rot%180!==0, iw=rotated?bmp.height:bmp.width, ih=rotated?bmp.width:bmp.height;
  const W=stage.clientWidth,H=stage.clientHeight;
  const s=fit?Math.min(W/iw,H/ih):Math.max(W/iw,H/ih);
  const w=Math.round(iw*s),h=Math.round(ih*s);
  cv.width=Math.min(w,W);cv.height=Math.min(h,H);
  ctx.save();ctx.translate(cv.width/2,cv.height/2);ctx.rotate(rot*Math.PI/180);
  ctx.drawImage(bmp,-bmp.width*s/2,-bmp.height*s/2,bmp.width*s,bmp.height*s);ctx.restore();
  if(showBoxes)drawBoxes(s);
}
function drawBoxes(s){
  // boxes are normalized to the image; map through the same rotate/fit transform as drawImage
  const bw=bmp.width*s,bh=bmp.height*s,cx=cv.width/2,cy=cv.height/2,a=rot*Math.PI/180,ca=Math.cos(a),sa=Math.sin(a);
  const tp=(nx,ny)=>{const x=nx*bw-bw/2,y=ny*bh-bh/2;return [cx+x*ca-y*sa,cy+x*sa+y*ca]};
  for(const d of dets){
    const [x1,y1,x2,y2]=d.box;const pts=[tp(x1,y1),tp(x2,y1),tp(x2,y2),tp(x1,y2)];
    ctx.save();ctx.lineWidth=2;ctx.strokeStyle='#fc6';ctx.beginPath();ctx.moveTo(...pts[0]);for(const p of pts.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.stroke();
    const top=pts.reduce((m,p)=>p[1]<m[1]?p:m);const label=`${d.display||d.name} ${Math.round(d.conf*100)}%`;
    ctx.font='12px system-ui';const tw=ctx.measureText(label).width+10;
    ctx.fillStyle='#fc6';ctx.fillRect(top[0],top[1]-20,tw,20);ctx.fillStyle='#000';ctx.fillText(label,top[0]+5,top[1]-6);
    ctx.restore();
  }
}
function renderDetbar(){const el=document.getElementById('detbar');const d=stats.detector||{};
  if(!d.available){el.innerHTML=`detector: <b>off</b> ${d.reason||''}`;return}
  if(d.enabled===false){el.innerHTML=`detector boxes: <b>off</b>`;return}
  el.innerHTML=`detector <b>${d.model}</b> · ${detMs||d.ms} ms · ${dets.length} box${dets.length===1?'':'es'}${dets.length?' · '+dets.map(x=>(x.display||x.name)+' '+Math.round(x.conf*100)+'%').join(', '):''}${!dets.length&&almost?` · <span style="color:#888">almost: ${almost.display||almost.label} ${Math.round(almost.conf*100)}%</span>`:''}`}
function connect(){
  const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/view');
  ws.binaryType='blob';
  ws.onmessage=async e=>{
    if(typeof e.data==='string'){const m=JSON.parse(e.data);
      if(m.type==='detections'){dets=m.boxes||[];detMs=m.ms||0;almost=m.almost||null;draw();renderDetbar();return;}
      if(m.type==='report_page'){document.getElementById('replink').style.display=m.enabled?'inline':'none';return;}
      if(m.type==='heard'){document.getElementById('heard').textContent='🎤 '+m.text;document.getElementById('out').textContent='';return;}
      if(m.type==='caption'){const out=document.getElementById('out');const isQ=(m.text||'').startsWith('You: ');out.textContent=m.text&&!isQ?m.text:(m.final?'':'thinking…');out.style.opacity=m.final?1:0.7;if(m.final&&m.text)loadReport();return;}
      stats=m;renderHud();renderDetbar();
      if(document.activeElement.id!=='voice')document.getElementById('voice').value=(m.tts&&m.tts.provider)||'apple';
      if(m.report_page!==undefined)document.getElementById('replink').style.display=m.report_page?'inline':'none';
      if(m.detector&&m.detector.enabled!==undefined&&document.activeElement.id!=='det')document.getElementById('det').checked=!!m.detector.enabled;
      document.getElementById('rstat').textContent=`${m.busy?'answering…':'ready'} · catalog ${m.catalog_parts} parts (${m.catalog_kb} KB) · ${m.frames_per_ask} frames per question`;return;}
    try{const b=await createImageBitmap(e.data);if(bmp)bmp.close();bmp=b;draw();
      shown++;const now=performance.now();if(now-lastShown>1000){dispFps=shown*1000/(now-lastShown);shown=0;lastShown=now;}}catch(err){}
  };
  ws.onclose=()=>{hud.textContent='disconnected, retrying…';setTimeout(connect,1000)};
}
function renderHud(){
  hud.textContent=`${bmp?bmp.width+'×'+bmp.height:'no frame'} · relay ${stats.fps??'-'} fps · shown ${dispFps.toFixed(1)} fps\\nphone→mac ${stats.phone_to_mac_ms??'-'} ms · ${stats.frame_kb??'-'} KB/frame · frames ${stats.frames??0} · buffer ${stats.recent??0} · phones ${stats.phones??0}\\nmodel ${stats.model??'?'} · voice ${stats.tts?stats.tts.provider:'?'} · questions ${stats.voice_in?stats.voice_in.asks:0}`;
}
window.addEventListener('resize',draw);
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});return r.json();}
async function ask(){const out=document.getElementById('out');const t=document.getElementById('q').value.trim();if(!t){out.textContent='type a question first';return;}out.textContent='listening…';
 const j=await post('/ask',{text:t});if(j.error)out.textContent=j.error;loadReport();}
async function describe(){const out=document.getElementById('out');out.textContent='looking…';const j=await post('/inspect',{});if(j.error)out.textContent=j.error;loadReport();}
async function hushNow(){await post('/ask',{text:'stop'});}
async function setDetector(){await post('/detector',{enabled:document.getElementById('det').checked});}
async function setVoice(){await post('/voice',{provider:document.getElementById('voice').value});}
async function loadReport(){const rep=await (await fetch('/report')).json();
 document.getElementById('rep').innerHTML=rep.slice(0,40).map(e=>`<div class=e><small>${new Date(e.ts*1000).toLocaleTimeString()} · ${e.model}${e.first_token_ms?` · first word ${e.first_token_ms} ms`:''}${e.stopped?' · stopped':''}</small>${e.question?`<div class=q>“${e.question}”</div>`:''}<div>${e.result}</div><div class=pics>${(e.frames||(e.frame?[e.frame]:[])).map(f=>`<img src="/frames/${f}">`).join('')}</div></div>`).join('');}
loadReport();connect();
</script></body></html>"""
