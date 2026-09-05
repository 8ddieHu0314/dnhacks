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
import struct
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

MODEL = os.environ.get("INSPECT_MODEL", "claude-opus-5")
SPEAK = os.environ.get("SPEAK", "0") == "1"
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

app = FastAPI()


def _lan_ips() -> list[str]:
    """All non-loopback IPv4 addresses (Wi-Fi, USB link to the phone, ...)."""
    import subprocess, re
    out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    return [ip for ip in re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out) if not ip.startswith("127.")]


_zc = None
_refresh_task = None


@app.on_event("startup")
async def _advertise():
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
        _zc.register_service(info)
        print(f"Bonjour: advertising _glassesrelay._tcp with {urls}")

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
        print("Bonjour advertise failed:", e)


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
            "viewers": len(viewers), "model": MODEL, "inspections": len(state["report"])}


@app.websocket("/")
async def ws_ingest_root(ws: WebSocket):
    await ws_ingest(ws)


@app.websocket("/ws/ingest")
async def ws_ingest(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_bytes()
            if len(msg) > 8 and msg[:2] != b"\xff\xd8":
                ts_ms = struct.unpack(">Q", msg[:8])[0]
                _ingest(msg[8:], ts_ms / 1000.0)
            else:
                _ingest(msg, None)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/view")
async def ws_view(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    viewers.add(q)
    if state["latest"]:
        q.put_nowait(state["latest"])
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
                await ws.send_text(json.dumps(_stats()))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        viewers.discard(q)


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
    import anthropic
    try:
        body = await request.json()
    except Exception:
        body = {}
    question = (body or {}).get("question") or "Inspect this frame."
    data = state["latest"]
    if not data:
        return JSONResponse({"error": "no frame yet"}, status_code=409)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JSONResponse({"error": "ANTHROPIC_API_KEY not set on the server; restart with the key exported."}, status_code=503)

    ts = time.time()
    frame_path = FRAMES_DIR / f"{int(ts)}.jpg"
    frame_path.write_bytes(data)
    b64 = base64.standard_b64encode(data).decode()
    client = anthropic.Anthropic()
    response = await asyncio.to_thread(lambda: client.beta.messages.create(
        model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": question}]}]))
    if response.stop_reason == "refusal":
        text = "The model declined to analyze this frame."
    else:
        text = "".join(b.text for b in response.content if b.type == "text").strip()
    entry = {"ts": ts, "question": question, "result": text, "frame": frame_path.name, "model": response.model}
    state["report"].append(entry)
    with REPORT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    if SPEAK:
        subprocess.Popen(["say", text])
    return entry


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
    if(typeof e.data==='string'){stats=JSON.parse(e.data);renderHud();return;}
    try{const b=await createImageBitmap(e.data);if(bmp)bmp.close();bmp=b;draw();
      shown++;const now=performance.now();if(now-lastShown>1000){dispFps=shown*1000/(now-lastShown);shown=0;lastShown=now;}}catch(err){}
  };
  ws.onclose=()=>{hud.textContent='disconnected, retrying…';setTimeout(connect,1000)};
}
function renderHud(){
  hud.textContent=`${bmp?bmp.width+'×'+bmp.height:'no frame'} · relay ${stats.fps??'-'} fps · shown ${dispFps.toFixed(1)} fps\\nphone→mac ${stats.phone_to_mac_ms??'-'} ms · ${stats.frame_kb??'-'} KB/frame · frames ${stats.frames??0}`;
}
window.addEventListener('resize',draw);
async function inspect(){const out=document.getElementById('out');out.textContent='thinking…';
 const r=await fetch('/inspect',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value||undefined})});
 const j=await r.json();out.textContent=j.result||j.error;loadReport();}
async function loadReport(){const rep=await (await fetch('/report')).json();
 document.getElementById('rep').innerHTML=rep.map(e=>`<div class=e><small>${new Date(e.ts*1000).toLocaleTimeString()} · ${e.model}</small><div>${e.result}</div><img src="/frames/${e.frame}"></div>`).join('');}
loadReport();connect();
</script></body></html>"""
