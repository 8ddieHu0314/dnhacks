"""Auto-label webcam capture frames with Grounding DINO for a single-class "component" detector.

Input: one or more session dirs with frames/*.jpg (from the capture page on :8000).
Output: a YOLO dataset (images/, labels/, data.yaml) with class 0 = component, a raw_detections.json
so the decision pass can be re-run without the model, and preview/ contact sheets for auditing.

Rules (learned from the gloves labeler):
- Grounding DINO is a witness, not an oracle. Boxes above `--box-thr` count, boxes over `--max-area`
  of the frame are dropped (a torso or the whole desk), boxes under `--min-area` are dropped (noise).
- Frames with no box are kept as negatives (empty label file): they teach hands, phone, shirt, shelf.
- Time-chunked val split so near-duplicate neighbours do not leak across train/val.

    .venv-inf/bin/python tools/components/gdino_label.py --session sessions/capture_X --every 3 --out datasets/components_v1
    .venv-inf/bin/python tools/components/gdino_label.py --from-raw datasets/components_v1/raw_detections.json --out datasets/components_v1 --box-thr 0.4
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

PROMPT = "an electronic circuit board. a sensor module. a breadboard. an electronic component. a servo motor. a jumper wire."
NEGATIVE_WORDS = ("hand", "person", "face", "shirt", "watch", "glasses", "phone")


def load_gdino(model_id, device):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


def gdino(processor, model, device, image: Image.Image, box_thr, text_thr):
    import torch
    w, h = image.size
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[(h, w)])[0]
    labels = res.get("text_labels", None)
    if labels is None:
        labels = res["labels"]
    items = []
    for box, score, label in zip(res["boxes"], res["scores"], labels):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        items.append({"label": str(label).lower(), "score": round(float(score), 3),
                      "box": [max(0.0, x1) / w, max(0.0, y1) / h, min(w, x2) / w, min(h, y2) / h]})
    return items


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def decide(items, box_thr, min_area, max_area, nms_iou=0.5, max_aspect=2.5, edge=0.01):
    """Raw witness boxes -> kept component boxes.
    Audit lessons (components_v1): the hexdump print on a t-shirt scores as a 'circuit board' in
    wide thin boxes (aspect filter); the parts organizer tray scores as a component (area cap);
    slivers on the frame border are the monitor edge (edge filter)."""
    keep = []
    for it in sorted(items, key=lambda x: -x["score"]):
        if it["score"] < box_thr:
            continue
        if any(wd in it["label"] for wd in NEGATIVE_WORDS):
            continue
        x1, y1, x2, y2 = it["box"]
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        if bw / max(bh, 1e-6) > max_aspect or bh / max(bw, 1e-6) > max_aspect:
            continue
        if (x1 <= edge or y1 <= edge or x2 >= 1 - edge or y2 >= 1 - edge) and area < 0.02:
            continue
        if any(iou(it["box"], k["box"]) > nms_iou for k in keep):
            continue
        keep.append(it)
    return keep


def contact_sheet(paths_boxes, out, cols=6, tw=320):
    tiles = []
    for p, boxes, tag in paths_boxes:
        im = Image.open(p).convert("RGB"); w, h = im.size
        d = ImageDraw.Draw(im)
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(40, 230, 90), width=6)
            d.text((x1 * w + 8, y1 * h + 8), f"{b['label'][:18]} {b['score']:.2f}", fill=(255, 255, 0))
        d.text((10, 10), tag, fill=(255, 80, 80))
        th = int(h * tw / w)
        tiles.append(im.resize((tw, th)))
    if not tiles:
        return
    th = tiles[0].height; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(out, quality=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="*", default=[])
    ap.add_argument("--every", type=int, default=3, help="use every Nth extracted frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--raw-thr", type=float, default=0.20, help="record witnesses down to this score for re-runs")
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--max-area", type=float, default=0.15)
    ap.add_argument("--max-aspect", type=float, default=2.5)
    ap.add_argument("--skip-uncertain", action="store_true",
                    help="drop frames where a witness scored above --uncertain-thr but nothing was kept (a part may be there)")
    ap.add_argument("--uncertain-thr", type=float, default=0.20)
    ap.add_argument("--val-chunks", type=int, default=5, help="1 of every N time chunks goes to val")
    ap.add_argument("--n-chunks", type=int, default=20)
    ap.add_argument("--from-raw", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    out = a.out
    if a.from_raw:
        raw = json.load(open(a.from_raw))
        frames = raw["frames"]
    else:
        paths = []
        for s in a.session:
            fs = sorted(os.listdir(os.path.join(s, "frames")))
            fs = [os.path.join(s, "frames", f) for f in fs if f.endswith(".jpg")][::a.every]
            paths += fs
        if a.limit:
            paths = paths[:a.limit]
        print(f"labeling {len(paths)} frames with {a.gdino} on {a.device}", flush=True)
        processor, model = load_gdino(a.gdino, a.device)
        frames = []
        t0 = time.time()
        for i, p in enumerate(paths):
            items = gdino(processor, model, a.device, Image.open(p).convert("RGB"), a.raw_thr, a.text_thr)
            frames.append({"path": p, "items": items})
            if i % 50 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(paths)}  {el:.0f}s  eta {el / max(1, i) * (len(paths) - i):.0f}s", flush=True)
        os.makedirs(out, exist_ok=True)
        json.dump({"prompt": PROMPT, "frames": frames}, open(os.path.join(out, "raw_detections.json"), "w"))

    # decision pass
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    n = len(frames); chunk = max(1, n // a.n_chunks)
    stats = {"train": 0, "val": 0, "boxes": 0, "negatives": 0, "skipped_uncertain": 0}
    kept_all = []
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        for f in os.listdir(os.path.join(out, sub)):
            os.remove(os.path.join(out, sub, f))
    for i, fr in enumerate(frames):
        split = "val" if (i // chunk) % a.val_chunks == a.val_chunks - 1 else "train"
        kept = decide(fr["items"], a.box_thr, a.min_area, a.max_area, max_aspect=a.max_aspect)
        if not kept and a.skip_uncertain and any(it["score"] >= a.uncertain_thr and not any(w in it["label"] for w in NEGATIVE_WORDS) for it in fr["items"]):
            stats["skipped_uncertain"] += 1
            continue
        stem = f"{i:05d}"
        shutil.copy(fr["path"], os.path.join(out, "images", split, stem + ".jpg"))
        with open(os.path.join(out, "labels", split, stem + ".txt"), "w") as f:
            for b in kept:
                x1, y1, x2, y2 = b["box"]
                f.write(f"0 {(x1 + x2) / 2:.5f} {(y1 + y2) / 2:.5f} {x2 - x1:.5f} {y2 - y1:.5f}\n")
        stats[split] += 1; stats["boxes"] += len(kept); stats["negatives"] += (not kept)
        kept_all.append((fr["path"], kept, f"{stem} {split}" + ("" if kept else " NEG")))
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['component']\n")
    # audit sheets: every 12th frame overall, plus every frame with 2+ boxes, plus a sample of negatives
    random.seed(0)
    sample = kept_all[::12]
    multi = [k for k in kept_all if len(k[1]) >= 2]
    negs = random.sample([k for k in kept_all if not k[1]], min(24, stats["negatives"]))
    for name, group in (("sample", sample), ("multi_box", multi), ("negatives", negs)):
        for j in range(0, len(group), 36):
            contact_sheet(group[j:j + 36], os.path.join(out, "preview", f"{name}_{j // 36:02d}.jpg"))
    print(json.dumps({"frames": n, **stats, "multi_box_frames": len(multi),
                      "box_thr": a.box_thr, "out": out}, indent=1))


if __name__ == "__main__":
    main()
