"""Gradio app for the PPE compliance playground.

Three tabs:
- Image: single image -> annotated output, detections JSON, compliance panel,
  and the sentence the glasses would speak.
- Video: uploaded clip -> offline pass with real debounce, a HUD overlay, and
  an alert events table.
- Live: browser webcam -> streamed frames through the real Spine (real
  debounce, tools model behind CachedPlugin), same HUD and events table, live.
"""

import os

import cv2
import gradio as gr
from dotenv import load_dotenv

from core.detectors import SH17Detector
from core.live import EventTracker, LiveSession, draw_hud, events_table_html
from core.spine import Spine, annotate, roboflow_tools_plugin
from core.stream import CachedPlugin

load_dotenv()

OUTPUT_DIR = os.path.abspath("outputs")
_SPINE_CACHE = {}


def build_spine(model_name: str, conf: float, roboflow_on: bool, debounce_s: float, cached_tools: bool) -> Spine:
    """The only place the playground wires detectors into a Spine.

    cached_tools=True mirrors server.py: the tools model runs behind
    CachedPlugin(every_n_frames=5, ttl_seconds=1.0). cached_tools=False runs
    it inline every frame (offline passes where latency does not matter).
    """
    weights_path = f"weights/{model_name}.pt"
    sh17 = SH17Detector(weights_path, device="mps", conf=conf)

    plugins = []
    if roboflow_on:
        plugin = roboflow_tools_plugin(conf=conf)
        if cached_tools:
            plugin = CachedPlugin(plugin, every_n_frames=5, ttl_seconds=1.0)
        plugins.append((plugin, 1))

    return Spine(sh17=sh17, plugins=plugins, debounce_seconds=float(debounce_s))


def get_spine(model_name: str, conf: float, roboflow_on: bool) -> Spine:
    key = (model_name, round(float(conf), 3), bool(roboflow_on))
    if key in _SPINE_CACHE:
        return _SPINE_CACHE[key]

    # debounce_seconds=0: this app only ever feeds one frame at a time (a
    # single image), so violations should fire immediately rather than
    # waiting for sustained activity. cached_tools=False: no benefit to a
    # background cadence when there is only ever one frame to judge.
    spine = build_spine(model_name, conf, roboflow_on, debounce_s=0.0, cached_tools=False)
    _SPINE_CACHE[key] = spine
    return spine


def _compliance_html(result) -> str:
    fired_rules = {f["rule"] for f in result.fired}
    rows = []
    for r in result.rule_results:
        if r.rule in fired_rules:
            status, bg, fg = "VIOLATION", "#ffdddd", "#b30000"
        elif not r.applicable:
            status, bg, fg = "UNKNOWN", "#eeeeee", "#555555"
        else:
            status, bg, fg = "PASS", "#ddffdd", "#0a7a0a"
        rows.append(
            f"<tr style='background:{bg};color:#111'>"
            f"<td style='padding:4px 8px;color:#111'>{r.rule}</td>"
            f"<td style='padding:4px 8px;color:{fg};font-weight:bold'>{status}</td>"
            f"<td style='padding:4px 8px;color:#111'>{r.detail}</td>"
            "</tr>"
        )

    for err in result.errors:
        rows.append(
            "<tr style='background:#fff3cd;color:#111'>"
            "<td style='padding:4px 8px;color:#111'>DETECTOR</td>"
            "<td style='padding:4px 8px;color:#8a6d00;font-weight:bold'>ERROR</td>"
            f"<td style='padding:4px 8px;color:#111'>{err}</td>"
            "</tr>"
        )

    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th style='text-align:left;padding:4px 8px'>Rule</th>"
        "<th style='text-align:left;padding:4px 8px'>Status</th>"
        "<th style='text-align:left;padding:4px 8px'>Detail</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def process_image(model_name, conf, roboflow_on, image):
    if image is None:
        return None, [], "<p>Upload an image first.</p>", "(silent)"

    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    spine = get_spine(model_name, conf, roboflow_on)
    spine.reset()
    result = spine.process(frame_bgr, dt=1.0)

    annotated_bgr = annotate(frame_bgr, result.detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections_json = [
        {
            "class": d.class_name,
            "conf": round(d.conf, 3),
            "box": [round(v, 1) for v in d.box],
            "source": d.source,
        }
        for d in result.detections
    ]

    html = _compliance_html(result)
    sentence = result.sentence or "(silent)"
    return annotated_rgb, detections_json, html, sentence


def process_video(model_name, conf, roboflow_on, debounce_s, sample_fps, video_path):
    """Offline pass over an uploaded video. Every frame is available, so
    there is no mailbox here: sample the video at `sample_fps`, feed the
    spine with the true dt between sampled frames, draw boxes and a HUD,
    and collect the alert events. Returns the annotated video path, an
    event table, and a stats block."""
    import statistics
    import time as _time

    if not video_path:
        return None, "<p>Upload a video first.</p>", ""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "<p>Could not open that video.</p>", ""
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / float(sample_fps))))
    dt = step / src_fps

    # No CachedPlugin here: every sampled frame is judged synchronously in
    # this offline pass, so there is no benefit to a background cadence.
    spine = build_spine(model_name, conf, roboflow_on, debounce_s, cached_tools=False)

    # Gradio only serves files from declared paths, so write under ./outputs
    # (declared via allowed_paths in launch) rather than the system temp dir.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"annotated_{int(_time.time())}.mp4")
    writer = None
    tracker = EventTracker()
    latencies, spoken = [], []
    idx = processed = 0
    t_video = 0.0
    last_sentence = None
    t_start = _time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        t_video = idx / src_fps
        result = spine.process(frame, dt=dt)
        latencies.append(result.latency_ms)
        processed += 1

        fired_now = {f["rule"] for f in result.fired}
        tracker.update(fired_now, t_video)
        if result.sentence and result.sentence != last_sentence:
            spoken.append((t_video, result.sentence))
        last_sentence = result.sentence

        out = annotate(frame, result.detections)
        draw_hud(out, result, fired_now, f"t={t_video:5.1f}s")

        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), float(sample_fps), (w, h))
            if not writer.isOpened():
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(sample_fps), (w, h))
        writer.write(out)
        idx += 1

    tracker.close_all(t_video)
    cap.release()
    if writer is not None:
        writer.release()
    wall = _time.perf_counter() - t_start

    said = "".join(f"<li style='color:#111'>t={t:.1f} s: {s}</li>" for t, s in spoken) or "<li style='color:#111'>(silent)</li>"
    table = events_table_html(tracker.rows(t_video)) + (
        "<div style='background:#fff;color:#111;padding:8px'>"
        f"<b>Glasses would say</b><ul>{said}</ul></div>"
    )
    med = statistics.median(latencies) if latencies else 0.0
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    stats = (
        f"source: {src_fps:.1f} fps, {total} frames | sampled every {step} frame(s) at {float(sample_fps):.1f} fps -> {processed} judged\n"
        f"judge latency: median {med:.1f} ms, p95 {p95:.1f} ms | wall time {wall:.1f} s | debounce {float(debounce_s):.1f} s\n"
        f"alerts: {len(tracker.events)} | sentences spoken: {len(spoken)}"
    )
    return out_path, table, stats


def live_step(model_name, conf, roboflow_on, debounce_s, frame_rgb, session, request: gr.Request):
    """Handler for the Live tab's streamed webcam frames. `request` is built
    once per Start..Stop event and reused for every chunk in it, so its
    identity marks the session boundary: a new id means the user clicked
    Start again and needs a fresh Spine and events table.
    """
    if frame_rgb is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    # Hold the request object itself, not id(request): CPython reuses a freed
    # object's address, so the next run's request could get the same id.
    params = (model_name, round(float(conf), 3), bool(roboflow_on), float(debounce_s))
    if session is None or session.request is not request or session.key != params:
        spine = build_spine(model_name, conf, roboflow_on, debounce_s, cached_tools=True)
        session = LiveSession(spine, params)
        session.request = request

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, rows, stats = session.step(frame_bgr)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, events_table_html(rows), stats, session


with gr.Blocks(title="PPE Compliance Check") as demo:
    gr.Markdown("# PPE Compliance Check")

    with gr.Row():
        model_dd = gr.Dropdown(["yolo8s", "yolo8s-gloves", "yolo8s-gloves-v2", "yolo8s-gloves-v4", "yolo8m"], value="yolo8s-gloves-v4", label="Model")
        conf_slider = gr.Slider(0.1, 0.9, value=0.40, step=0.05, label="Confidence")
        roboflow_cb = gr.Checkbox(value=True, label="Use Roboflow tools model")

    with gr.Tab("Image"):
        img_in = gr.Image(label="Upload image", type="numpy")
        img_btn = gr.Button("Run")
        img_out = gr.Image(label="Annotated")
        json_out = gr.JSON(label="Detections")
        html_out = gr.HTML(label="Compliance")
        sentence_out = gr.Textbox(label="Glasses would say")

        img_btn.click(
            process_image,
            inputs=[model_dd, conf_slider, roboflow_cb, img_in],
            outputs=[img_out, json_out, html_out, sentence_out],
        )

    with gr.Tab("Video"):
        gr.Markdown("Upload a clip. Frames are sampled at the chosen rate, run through the spine with the real time between frames, and alerts must hold for the debounce window before they fire, same as live.")
        with gr.Row():
            debounce_in = gr.Slider(0.0, 5.0, value=2.0, step=0.5, label="Debounce seconds")
            sample_fps_in = gr.Slider(1, 30, value=10, step=1, label="Judge fps")
        video_in = gr.Video(label="Upload video")
        video_btn = gr.Button("Run video")
        video_out = gr.Video(label="Annotated video with HUD")
        events_out = gr.HTML(label="Alert events")
        stats_out = gr.Textbox(label="Stats", lines=3)

        video_btn.click(
            process_video,
            inputs=[model_dd, conf_slider, roboflow_cb, debounce_in, sample_fps_in, video_in],
            outputs=[video_out, events_out, stats_out],
        )

    with gr.Tab("Live", elem_id="live-tab"):
        gr.Markdown("Point a webcam (phone or laptop) at your hands and press the record button. Frames go through the real spine: real debounce and the tools model on its background cadence, exactly like server.py.")
        live_debounce = gr.Slider(0.0, 5.0, value=2.0, step=0.5, label="Debounce seconds", elem_id="live-debounce")
        with gr.Row():
            live_in = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam", elem_id="live-webcam")
            live_out = gr.Image(label="Annotated with HUD", streaming=True, elem_id="live-annotated")
        live_events = gr.HTML(label="Alert events", value=events_table_html([]), elem_id="live-events")
        live_stats = gr.Textbox(label="Live stats", lines=1, interactive=False, elem_id="live-stats")
        live_state = gr.State(None)
        live_in.stream(
            live_step,
            inputs=[model_dd, conf_slider, roboflow_cb, live_debounce, live_in, live_state],
            outputs=[live_out, live_events, live_stats, live_state],
            stream_every=0.1,
            concurrency_limit=1,
            show_progress="hidden",
        )


if __name__ == "__main__":
    url = "http://127.0.0.1:7860"
    print(f"Launching Gradio app at {url}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, allowed_paths=[OUTPUT_DIR])
