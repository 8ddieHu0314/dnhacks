"""Gradio app for Step 0 PPE compliance checking.

Two tabs:
- Image: single image -> annotated output, detections JSON, compliance panel,
  and the sentence the glasses would speak.
- Batch: multiple images -> gallery of annotated images + results table.
"""

import os

import cv2
import gradio as gr
from dotenv import load_dotenv

from core.detectors import SH17Detector
from core.spine import Spine, annotate, roboflow_tools_plugin

load_dotenv()

_SPINE_CACHE = {}


def get_spine(model_name: str, conf: float, roboflow_on: bool) -> Spine:
    key = (model_name, round(float(conf), 3), bool(roboflow_on))
    if key in _SPINE_CACHE:
        return _SPINE_CACHE[key]

    weights_path = f"weights/{model_name}.pt"
    sh17 = SH17Detector(weights_path, device="mps", conf=conf)

    plugins = []
    if roboflow_on:
        plugin = roboflow_tools_plugin(conf=conf)
        plugins.append((plugin, 1))

    # debounce_seconds=0: this app only ever feeds one frame at a time
    # (single image, or one image per batch row), so violations should
    # fire immediately rather than waiting for sustained activity.
    spine = Spine(sh17=sh17, plugins=plugins, debounce_seconds=0.0)
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


def process_batch(model_name, conf, roboflow_on, files):
    if not files:
        return [], "<p>Upload one or more images first.</p>"

    spine = get_spine(model_name, conf, roboflow_on)

    gallery = []
    rows = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        filename = os.path.basename(path)
        frame_bgr = cv2.imread(path)
        if frame_bgr is None:
            rows.append(
                f"<tr><td style='padding:4px 8px'>{filename}</td>"
                "<td style='padding:4px 8px' colspan='2'>could not read image</td></tr>"
            )
            continue

        spine.reset()
        result = spine.process(frame_bgr, dt=1.0)

        annotated_bgr = annotate(frame_bgr, result.detections)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        gallery.append((annotated_rgb, filename))

        fired_names = ", ".join(f["rule"] for f in result.fired) or "(none)"
        sentence = result.sentence or "(silent)"
        rows.append(
            f"<tr><td style='padding:4px 8px'>{filename}</td>"
            f"<td style='padding:4px 8px'>{fired_names}</td>"
            f"<td style='padding:4px 8px'>{sentence}</td></tr>"
        )

    html = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th style='text-align:left;padding:4px 8px'>Filename</th>"
        "<th style='text-align:left;padding:4px 8px'>Fired rules</th>"
        "<th style='text-align:left;padding:4px 8px'>Sentence</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return gallery, html


def process_video(model_name, conf, roboflow_on, debounce_s, sample_fps, video_path):
    """Offline pass over an uploaded video. Every frame is available, so
    there is no mailbox here: sample the video at `sample_fps`, feed the
    spine with the true dt between sampled frames, draw boxes and a HUD,
    and collect the alert events. Returns the annotated video path, an
    event table, and a stats block."""
    import statistics
    import tempfile
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

    sh17 = SH17Detector(f"weights/{model_name}.pt", device="mps", conf=conf)
    plugins = [(roboflow_tools_plugin(conf=conf), 1)] if roboflow_on else []
    spine = Spine(sh17=sh17, plugins=plugins, debounce_seconds=float(debounce_s))

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = None
    events, open_events = [], {}
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
        for rule in fired_now:
            if rule not in open_events:
                open_events[rule] = {"rule": rule, "start": t_video}
        for rule in list(open_events):
            if rule not in fired_now:
                ev = open_events.pop(rule)
                ev["end"] = t_video
                events.append(ev)
        if result.sentence and result.sentence != last_sentence:
            spoken.append((t_video, result.sentence))
        last_sentence = result.sentence

        out = annotate(frame, result.detections)
        hud = [f"t={t_video:5.1f}s  judge {result.latency_ms:.0f} ms"]
        for r in result.rule_results:
            timer = result.timers.get(r.rule, 0.0)
            state = "FIRED" if r.rule in fired_now else ("active" if r.active else ("pass" if r.applicable else "unknown"))
            hud.append(f"{r.rule}: {state} {timer:.1f}s")
        if result.sentence:
            hud.append(f'SAY: "{result.sentence}"')
        y = 22
        for i, line in enumerate(hud):
            color = (0, 0, 255) if (i == len(hud) - 1 and result.sentence) else (255, 255, 255)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 22

        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), float(sample_fps), (w, h))
            if not writer.isOpened():
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(sample_fps), (w, h))
        writer.write(out)
        idx += 1

    for rule, ev in open_events.items():
        ev["end"] = t_video
        events.append(ev)
    cap.release()
    if writer is not None:
        writer.release()
    wall = _time.perf_counter() - t_start

    rows = "".join(
        f"<tr><td style='padding:4px 8px;color:#111'>{e['rule']}</td>"
        f"<td style='padding:4px 8px;color:#111'>{e['start']:.1f} s</td>"
        f"<td style='padding:4px 8px;color:#111'>{e['end']:.1f} s</td>"
        f"<td style='padding:4px 8px;color:#111'>{e['end'] - e['start']:.1f} s</td></tr>"
        for e in sorted(events, key=lambda e: e["start"])
    ) or "<tr><td colspan='4' style='padding:4px 8px;color:#111'>no alerts fired</td></tr>"
    said = "".join(f"<li>t={t:.1f} s: {s}</li>" for t, s in spoken) or "<li>(silent)</li>"
    table = (
        "<table style='border-collapse:collapse;background:#fff'>"
        "<tr><th style='text-align:left;padding:4px 8px;color:#111'>Rule</th><th style='color:#111'>Start</th>"
        "<th style='color:#111'>End</th><th style='color:#111'>Duration</th></tr>"
        f"{rows}</table><p style='color:#111;background:#fff;padding:6px'><b>Glasses would say</b><ul>{said}</ul></p>"
    )
    med = statistics.median(latencies) if latencies else 0.0
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    stats = (
        f"source: {src_fps:.1f} fps, {total} frames | sampled every {step} frame(s) at {float(sample_fps):.1f} fps -> {processed} judged\n"
        f"judge latency: median {med:.1f} ms, p95 {p95:.1f} ms | wall time {wall:.1f} s | debounce {float(debounce_s):.1f} s\n"
        f"alerts: {len(events)} | sentences spoken: {len(spoken)}"
    )
    return out_path, table, stats


with gr.Blocks(title="PPE Compliance Check") as demo:
    gr.Markdown("# PPE Compliance Check")

    with gr.Row():
        model_dd = gr.Dropdown(["yolo8s", "yolo8m"], value="yolo8s", label="Model")
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

    with gr.Tab("Batch"):
        files_in = gr.File(label="Upload images", file_count="multiple")
        batch_btn = gr.Button("Run batch")
        gallery_out = gr.Gallery(label="Annotated images")
        table_out = gr.HTML(label="Results")

        batch_btn.click(
            process_batch,
            inputs=[model_dd, conf_slider, roboflow_cb, files_in],
            outputs=[gallery_out, table_out],
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


if __name__ == "__main__":
    url = "http://127.0.0.1:7860"
    print(f"Launching Gradio app at {url}")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
