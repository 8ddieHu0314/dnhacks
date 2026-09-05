"""Auto-label captured frames for a gloves fine-tune of the SH17 model.

Two label sources, merged per frame:
  1. The current SH17 model (weights/yolo8s.pt) at a low conf gives pseudo-labels
     for every class EXCEPT gloves and hands. Keeping those boxes matters: if the
     new frames carried only glove labels, every unlabeled face and person would
     count as background and the fine-tune would unlearn them.
  2. Grounding DINO, prompted with "a glove. a hand.", gives the gloves and hands
     boxes. It is an open-vocabulary detector, so it does not share the SH17
     model's blind spot for this particular glove.

Output is an Ultralytics dataset (images/, labels/, data.yaml) with the 17 SH17
class names in their original order, split into train/val by time so the val
frames are not near-duplicates of train frames, plus preview/ with annotated
samples and a contact sheet for a quick human skim.

Offline tool. Nothing here runs on the live frame path.

Usage:
  .venv/bin/python label/gdino_label.py --session sessions/capture_XXX --every 15 --out datasets/gloves_v1
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SH17_NAMES = [
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask", "foot", "tool", "glasses",
    "gloves", "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest",
]
GLOVES_ID = SH17_NAMES.index("gloves")
HANDS_ID = SH17_NAMES.index("hands")
# A specific prompt scores higher and separates the classes better than
# "a glove. a hand." (probed on 12 frames: glove scores 0.47-0.75 vs 0.25-0.59).
PROMPT = "a yellow work glove. a bare hand."
# Grounding DINO still confuses the two on some frames, but the glove is the
# only bright yellow thing in the scene, so the fraction of yellow pixels in a
# box is a near-perfect tiebreaker: real gloves measured 0.30-0.67, bare hands
# 0.00-0.07. Boxes in between are ambiguous (a bare hand overlapping a glove)
# and fall back to the text label when it is unambiguous, else are dropped.
YELLOW_GLOVE_MIN = 0.30
YELLOW_HAND_MAX = 0.10


def yellow_fraction(frame_bgr, box) -> float:
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 90)
    return float(mask.mean())


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes: list, thr: float) -> list:
    """boxes: list of (cls, score, x1, y1, x2, y2). Greedy NMS within class."""
    keep = []
    for b in sorted(boxes, key=lambda b: -b[1]):
        if all(not (k[0] == b[0] and iou(k[2:], b[2:]) > thr) for k in keep):
            keep.append(b)
    return keep


def load_gdino(model_id: str, device: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


def run_gdino(processor, model, device, frame_bgr, box_thr: float, text_thr: float) -> list:
    """Returns list of (cls_id, score, x1, y1, x2, y2) in pixel coords for gloves/hands."""
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    h, w = frame_bgr.shape[:2]
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[(h, w)]
    )[0]
    labels = results.get("text_labels", None)
    if labels is None:
        labels = results["labels"]
    out = []
    counters = {"color_gloves": 0, "color_hands": 0, "text_fallback": 0, "ambiguous_dropped": 0}
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        label = str(label).lower()
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        clipped = (max(0.0, x1), max(0.0, y1), min(w, x2), min(h, y2))
        yf = yellow_fraction(frame_bgr, clipped)
        has_glove, has_hand = "glove" in label, "hand" in label
        if yf >= YELLOW_GLOVE_MIN:
            cls = GLOVES_ID
            counters["color_gloves"] += 1
        elif yf <= YELLOW_HAND_MAX:
            cls = HANDS_ID
            counters["color_hands"] += 1
        elif has_glove != has_hand:
            cls = GLOVES_ID if has_glove else HANDS_ID
            counters["text_fallback"] += 1
        else:
            counters["ambiguous_dropped"] += 1
            continue
        out.append((cls, float(score), *clipped))
    run_gdino.counters = counters
    return out


def resolve_hand_glove_conflicts(boxes: list, thr: float = 0.5) -> tuple:
    """SH17 convention: a gloved hand is a gloves box with NO hands box under
    it. When Grounding DINO says both for the same region, keep the higher
    score. Returns (kept, n_dropped)."""
    gloves = [b for b in boxes if b[0] == GLOVES_ID]
    hands = [b for b in boxes if b[0] == HANDS_ID]
    dropped = 0
    kept = []
    for g in gloves:
        rivals = [h for h in hands if iou(g[2:], h[2:]) > thr]
        if any(h[1] > g[1] for h in rivals):
            dropped += 1
            continue
        kept.append(g)
    for h in hands:
        rivals = [g for g in gloves if iou(g[2:], h[2:]) > thr]
        if any(g[1] >= h[1] for g in rivals):
            dropped += 1
            continue
        kept.append(h)
    return kept, dropped


def to_yolo_line(box, w, h) -> str:
    cls, _score, x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def draw(frame_bgr, boxes):
    out = frame_bgr.copy()
    for cls, score, x1, y1, x2, y2 in boxes:
        if cls == GLOVES_ID:
            color = (0, 200, 255)  # orange-ish for gloves
        elif cls == HANDS_ID:
            color = (0, 0, 255)  # red for bare hands
        else:
            color = (0, 255, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{SH17_NAMES[cls]} {score:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def contact_sheet(images: list, cols: int = 6, thumb_w: int = 320) -> np.ndarray:
    thumbs = []
    for im in images:
        h, w = im.shape[:2]
        thumbs.append(cv2.resize(im, (thumb_w, int(h * thumb_w / w))))
    if not thumbs:
        return np.zeros((10, 10, 3), np.uint8)
    th = max(t.shape[0] for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * thumb_w, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th:r * th + t.shape[0], c * thumb_w:c * thumb_w + t.shape[1]] = t
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session dir with frames/")
    ap.add_argument("--every", type=int, default=15, help="keep every Nth frame")
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept frames (smoke test)")
    ap.add_argument("--out", required=True, help="dataset output dir")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--sh17-weights", default="weights/yolo8s.pt")
    ap.add_argument("--sh17-conf", type=float, default=0.25)
    ap.add_argument("--val-chunks", type=int, default=5, help="1 of every N time chunks goes to val")
    ap.add_argument("--n-chunks", type=int, default=10)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    frames_dir = os.path.join(args.session, "frames")
    names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg"))
    names = names[:: max(1, args.every)]
    if args.limit:
        names = names[: args.limit]
    if not names:
        sys.exit(f"no frames in {frames_dir}")
    print(f"{len(names)} frames selected from {frames_dir} (every {args.every})")

    for sub in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    from ultralytics import YOLO

    sh17 = YOLO(args.sh17_weights)
    t = time.time()
    processor, gdino = load_gdino(args.gdino, args.device)
    print(f"grounding dino loaded on {args.device} in {time.time() - t:.1f}s")

    n_chunks = max(1, args.n_chunks)
    chunk_size = max(1, (len(names) + n_chunks - 1) // n_chunks)

    stats = {
        "frames": 0, "train": 0, "val": 0, "conflicts_dropped": 0,
        "per_class": {n: 0 for n in SH17_NAMES}, "frames_no_hand_or_glove": 0,
        "gate": {"color_gloves": 0, "color_hands": 0, "text_fallback": 0, "ambiguous_dropped": 0},
        "prompt": PROMPT, "yellow_glove_min": YELLOW_GLOVE_MIN, "yellow_hand_max": YELLOW_HAND_MAX,
        "gdino_seconds": 0.0, "session": args.session, "every": args.every,
        "gdino_model": args.gdino, "box_thr": args.box_thr, "text_thr": args.text_thr,
    }
    previews = []
    for i, name in enumerate(names):
        path = os.path.join(frames_dir, name)
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # 1. pseudo-labels for everything except gloves/hands
        res = sh17.predict(frame, conf=args.sh17_conf, device=args.device, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            cls = int(b.cls[0])
            if cls in (GLOVES_ID, HANDS_ID):
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            boxes.append((cls, float(b.conf[0]), x1, y1, x2, y2))

        # 2. gloves and hands from grounding dino
        t = time.time()
        gd = run_gdino(processor, gdino, args.device, frame, args.box_thr, args.text_thr)
        stats["gdino_seconds"] += time.time() - t
        for k, v in run_gdino.counters.items():
            stats["gate"][k] += v
        gd = nms(gd, 0.7)
        gd, dropped = resolve_hand_glove_conflicts(gd)
        stats["conflicts_dropped"] += dropped
        if not gd:
            stats["frames_no_hand_or_glove"] += 1
        boxes.extend(gd)

        for b in boxes:
            stats["per_class"][SH17_NAMES[b[0]]] += 1

        split = "val" if (i // chunk_size) % args.val_chunks == args.val_chunks - 1 else "train"
        stats[split] += 1
        stats["frames"] += 1
        stem = os.path.splitext(name)[0]
        shutil.copyfile(path, os.path.join(args.out, "images", split, f"{stem}.jpg"))
        with open(os.path.join(args.out, "labels", split, f"{stem}.txt"), "w") as f:
            f.write("\n".join(to_yolo_line(b, w, h) for b in boxes) + ("\n" if boxes else ""))

        if i % max(1, len(names) // 48) == 0:
            annotated = draw(frame, boxes)
            cv2.imwrite(os.path.join(args.out, "preview", f"{split}_{stem}.jpg"), annotated)
            previews.append(annotated)
        if (i + 1) % 25 == 0:
            g, hd = stats["per_class"]["gloves"], stats["per_class"]["hands"]
            print(f"  {i + 1}/{len(names)} frames, gloves={g} hands={hd}, "
                  f"gdino {stats['gdino_seconds'] / (i + 1):.2f}s/frame", flush=True)

    cv2.imwrite(os.path.join(args.out, "preview", "contact_sheet.jpg"), contact_sheet(previews))

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(SH17_NAMES)))
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps({k: v for k, v in stats.items() if k != "per_class"}, indent=2))
    print("per_class:", {k: v for k, v in stats["per_class"].items() if v})
    print(f"dataset: {yaml_path}\npreview: {os.path.join(args.out, 'preview', 'contact_sheet.jpg')}")


if __name__ == "__main__":
    main()
