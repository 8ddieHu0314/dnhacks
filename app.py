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


with gr.Blocks(title="PPE Compliance Check") as demo:
    gr.Markdown("# PPE Compliance Check (Step 0)")

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


if __name__ == "__main__":
    url = "http://127.0.0.1:7860"
    print(f"Launching Gradio app at {url}")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
