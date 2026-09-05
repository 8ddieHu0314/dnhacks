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
HELMET_ID = SH17_NAMES.index("helmet")
# A specific prompt scores higher and separates the classes better than
# "a glove. a hand." (probed on 12 frames: glove scores 0.47-0.75 vs 0.25-0.59).
PROMPT = "a yellow work glove. a bare hand."
# Helmets get their own pass. The v1 model learned "big yellow blob = gloves"
# and called a yellow hard hat gloves 0.88, so helmets are labeled as hard
# negatives, masked out of the yellow measurement, and any glove/hand box
# mostly inside one is dropped. A separate prompt is needed: with "a hard hat"
# appended to PROMPT, the glove phrase wins the label on the helmet box
# ("a yellow work glove" 0.48) while alone it is "a hard hat" 0.88.
HELMET_PROMPT = "a hard hat."
# Grounding DINO still confuses the two on some frames, but the glove is the
# only bright yellow thing in the scene, so the fraction of yellow pixels in a
# box is a near-perfect tiebreaker: real gloves measured 0.30-0.67, bare hands
# 0.00-0.07. Boxes in between are ambiguous (a bare hand overlapping a glove)
# and fall back to the text label when it is unambiguous, else are dropped.
YELLOW_GLOVE_MIN = 0.30
YELLOW_HAND_MAX = 0.10
# Probed: "a hard hat." alone scores 0.30-0.56 on glove-only frames (gloves
# look hat-like to it) and 0.58-0.86 on real helmets. SH17's own helmet
# pseudo-labels are the second source, so a high bar here is safe.
HELMET_MIN_SCORE = 0.60
HELMET_WEAK_SCORE = 0.35   # weak hat hits only count together with the yellow-plastic check
YELLOW_PLASTIC_MIN = 0.70  # above this a "glove" box is smooth plastic, i.e. the helmet


def yellow_fraction(frame_bgr, box, exclude=()) -> float:
    """Fraction of bright-yellow pixels in the box. Pixels inside any `exclude`
    box (helmets) are ignored, so a bare hand holding a yellow hard hat is not
    counted as yellow."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame_bgr[y1:max(y1, y2), x1:max(x1, x2)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 90)
    valid = np.ones(mask.shape, dtype=bool)
    for ex in exclude:
        ex1, ey1, ex2, ey2 = [int(v) for v in ex]
        valid[max(0, ey1 - y1):max(0, ey2 - y1), max(0, ex1 - x1):max(0, ex2 - x1)] = False
    if valid.sum() < 0.2 * valid.size:
        return None  # box is mostly helmet, color cannot judge it
    return float(mask[valid].mean())


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


def gdino_raw(processor, model, device, frame_bgr, box_thr: float, text_thr: float) -> dict:
    """The two Grounding DINO passes for one frame, no decisions yet:
    hats: [(score, box)] from the helmet prompt (weak hits included, tiny
          boxes removed: a background object scored 0.6 once)
    items: [(label, score, box)] from the glove/hand prompt"""
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w = frame_bgr.shape[:2]

    def detect(prompt, thr, tthr):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=thr, text_threshold=tthr, target_sizes=[(h, w)]
        )[0]
        labels = results.get("text_labels", None)
        if labels is None:
            labels = results["labels"]
        items = []
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            items.append((str(label).lower(), float(score), (max(0.0, x1), max(0.0, y1), min(w, x2), min(h, y2))))
        return items

    min_area = 0.01 * w * h
    hats = [(score, clipped) for _label, score, clipped in detect(HELMET_PROMPT, HELMET_WEAK_SCORE, text_thr)
            if (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) >= min_area]
    return {"hats": hats, "items": detect(PROMPT, box_thr, text_thr)}


def strong_helmets(raw: dict) -> list:
    """Helmet boxes confident enough to stand on their own."""
    return [box for score, box in raw["hats"] if score >= HELMET_MIN_SCORE]


def classify(frame_bgr, raw: dict, known_helmets=(), neighbor_helmets=()) -> tuple:
    """Turn one frame's raw detections into (cls, score, x1, y1, x2, y2) boxes
    for gloves, hands and helmets. Returns (boxes, counters).

    known_helmets: helmet boxes from the SH17 pseudo-labels for this frame.
    neighbor_helmets: confident helmet boxes from frames half a second either
    side. An object cannot be a helmet in one frame and a glove in the next,
    so a weak hat hit here plus a confident helmet nearby means helmet.
    Helmets veto glove boxes, are masked out of the yellow measurement, and
    the SH17 ones are not returned (the caller already has them)."""
    out = []
    counters = {"color_gloves": 0, "color_hands": 0, "text_fallback": 0, "ambiguous_dropped": 0,
                "helmets": 0, "helmet_veto": 0, "plastic_to_helmet": 0, "temporal_to_helmet": 0}
    hat_candidates = raw["hats"]
    helmets = [(HELMET_ID, score, *box) for score, box in hat_candidates if score >= HELMET_MIN_SCORE]
    counters["helmets"] = len(helmets)
    # A cut-off or oddly held helmet can win the glove label outright
    # ("yellow work glove" 0.74 vs "hard hat" 0.42 on one frame). Two rescues,
    # each needing a weak hat hit on the same box as a first witness:
    #   plastic:  very yellow (helmet boxes measured 0.72-0.93, real gloves
    #             never exceeded 0.67 across the first clip)
    #   temporal: a confident helmet in a neighboring frame at the same place
    still = []
    for label, score, clipped in raw["items"]:
        weak_hat = any(iou(clipped, hb) > 0.5 for _s, hb in hat_candidates)
        if weak_hat:
            yf_raw = yellow_fraction(frame_bgr, clipped)
            if yf_raw is not None and yf_raw > YELLOW_PLASTIC_MIN:
                helmets.append((HELMET_ID, score, *clipped))
                counters["plastic_to_helmet"] += 1
                continue
            if any(iou(clipped, nb) > 0.3 for nb in neighbor_helmets):
                helmets.append((HELMET_ID, score, *clipped))
                counters["temporal_to_helmet"] += 1
                continue
        still.append((label, score, clipped))
    items = still
    # Every helmet we know about, from either source, is masked out of the
    # yellow measurement so a bare hand holding a yellow hard hat stays a hand.
    helmet_boxes = [hm[2:] for hm in helmets] + [tuple(b) for b in known_helmets]
    for label, score, clipped in items:
        yf = yellow_fraction(frame_bgr, clipped, exclude=helmet_boxes)
        has_glove, has_hand = "glove" in label, "hand" in label
        if yf is not None and yf >= YELLOW_GLOVE_MIN:
            cls = GLOVES_ID
            counters["color_gloves"] += 1
        elif yf is not None and yf <= YELLOW_HAND_MAX:
            cls = HANDS_ID
            counters["color_hands"] += 1
        elif has_glove != has_hand:
            cls = GLOVES_ID if has_glove else HANDS_ID
            counters["text_fallback"] += 1
        else:
            counters["ambiguous_dropped"] += 1
            continue
        out.append((cls, float(score), *clipped))
    # Veto: a glove or hand box that mostly sits inside a helmet box is the
    # helmet, not a glove (measured as intersection over the smaller box).
    kept = []
    for b in out:
        # A glove box mostly inside a helmet box is the helmet. A hand box is
        # only dropped when it is almost entirely helmet: hands holding the
        # helmet must stay labeled or the model learns them as background.
        limit = 0.6 if b[0] == GLOVES_ID else 0.9
        if any(_ios(b[2:], hb) > limit for hb in helmet_boxes):
            counters["helmet_veto"] += 1
            continue
        kept.append(b)
    return kept + helmets, counters


def _ios(a, b) -> float:
    """Intersection over the smaller box's area."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (smaller + 1e-9) if smaller > 0 else 0.0


def resolve_hand_glove_conflicts(boxes: list, thr: float = 0.5) -> tuple:
    """SH17 convention: a gloved hand is a gloves box with NO hands box under
    it. When Grounding DINO says both for the same region, keep the higher
    score. Returns (kept, n_dropped)."""
    gloves = [b for b in boxes if b[0] == GLOVES_ID]
    hands = [b for b in boxes if b[0] == HANDS_ID]
    dropped = 0
    kept = [b for b in boxes if b[0] not in (GLOVES_ID, HANDS_ID)]  # helmets pass through
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
        elif cls == HELMET_ID:
            color = (255, 0, 0)  # blue for helmets
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
    ap.add_argument("--session", required=True, nargs="+", help="one or more session dirs with frames/")
    ap.add_argument("--every", type=int, default=[15], nargs="+",
                    help="keep every Nth frame; one value, or one per --session")
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
    # entries: (path, stem, split). Each session is chunked by time on its
    # own, so every clip contributes to both train and val, and stems carry a
    # session index so frame names from different clips cannot collide.
    entries = []
    n_chunks = max(1, args.n_chunks)
    for k, session in enumerate(args.session):
        every = args.every[k] if len(args.every) > 1 else args.every[0]
        frames_dir = os.path.join(session, "frames")
        names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg"))
        names = names[:: max(1, every)]
        if args.limit:
            names = names[: args.limit]
        if not names:
            sys.exit(f"no frames in {frames_dir}")
        chunk_size = max(1, (len(names) + n_chunks - 1) // n_chunks)
        for i, name in enumerate(names):
            split = "val" if (i // chunk_size) % args.val_chunks == args.val_chunks - 1 else "train"
            entries.append((os.path.join(frames_dir, name), f"s{k}_{os.path.splitext(name)[0]}", split))
        print(f"{len(names)} frames selected from {frames_dir} (every {every})")
    names = entries

    for sub in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    from ultralytics import YOLO

    sh17 = YOLO(args.sh17_weights)
    t = time.time()
    processor, gdino = load_gdino(args.gdino, args.device)
    print(f"grounding dino loaded on {args.device} in {time.time() - t:.1f}s")

    stats = {
        "frames": 0, "train": 0, "val": 0, "conflicts_dropped": 0,
        "per_class": {n: 0 for n in SH17_NAMES}, "frames_no_hand_or_glove": 0,
        "gate": {"color_gloves": 0, "color_hands": 0, "text_fallback": 0, "ambiguous_dropped": 0,
                 "helmets": 0, "helmet_veto": 0, "plastic_to_helmet": 0, "temporal_to_helmet": 0},
        "prompt": PROMPT, "yellow_glove_min": YELLOW_GLOVE_MIN, "yellow_hand_max": YELLOW_HAND_MAX,
        "gdino_seconds": 0.0, "session": args.session, "every": args.every,
        "gdino_model": args.gdino, "box_thr": args.box_thr, "text_thr": args.text_thr,
    }
    # Pass A: run both detectors on every frame, decide nothing yet.
    cache = []  # per entry: (sh17_boxes, raw) or None if unreadable
    for i, (path, stem, split) in enumerate(names):
        frame = cv2.imread(path)
        if frame is None:
            cache.append(None)
            continue
        # 1. pseudo-labels for everything except gloves/hands
        res = sh17.predict(frame, conf=args.sh17_conf, device=args.device, verbose=False)[0]
        sh17_boxes = []
        for b in res.boxes:
            cls = int(b.cls[0])
            if cls in (GLOVES_ID, HANDS_ID):
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            sh17_boxes.append((cls, float(b.conf[0]), x1, y1, x2, y2))
        # 2. gloves, hands and helmet candidates from grounding dino
        t = time.time()
        raw = gdino_raw(processor, gdino, args.device, frame, args.box_thr, args.text_thr)
        stats["gdino_seconds"] += time.time() - t
        cache.append((sh17_boxes, raw))
        if (i + 1) % 25 == 0:
            print(f"  detect {i + 1}/{len(names)} frames, gdino {stats['gdino_seconds'] / (i + 1):.2f}s/frame", flush=True)

    # Pass B: classify each frame with its neighbors' confident helmets.
    def session_of(stem):
        return stem.split("_", 1)[0]

    previews = []
    for i, (path, stem, split) in enumerate(names):
        if cache[i] is None:
            continue
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        sh17_boxes, raw = cache[i]
        boxes = list(sh17_boxes)
        sh17_helmets = [b[2:] for b in boxes if b[0] == HELMET_ID]
        neighbors = []
        for j in range(max(0, i - 2), min(len(names), i + 3)):
            if j == i or cache[j] is None or session_of(names[j][1]) != session_of(stem):
                continue
            neighbors.extend(strong_helmets(cache[j][1]))
            neighbors.extend(b[2:] for b in cache[j][0] if b[0] == HELMET_ID)
        gd, counters = classify(frame, raw, known_helmets=sh17_helmets, neighbor_helmets=neighbors)
        for k, v in counters.items():
            stats["gate"][k] += v
        gd = nms(gd, 0.7)
        gd, dropped = resolve_hand_glove_conflicts(gd)
        stats["conflicts_dropped"] += dropped
        if not any(b[0] in (GLOVES_ID, HANDS_ID) for b in gd):
            stats["frames_no_hand_or_glove"] += 1
        # Merge helmet boxes that both sources found (partial vs full box of
        # the same helmet overlap far less than NMS's IoU bar, so use
        # intersection over the smaller box), then one NMS pass.
        boxes.extend(gd)
        helmets_sorted = sorted((b for b in boxes if b[0] == HELMET_ID), key=lambda b: -b[1])
        merged = []
        for hm in helmets_sorted:
            if not any(_ios(hm[2:], k[2:]) > 0.7 for k in merged):
                merged.append(hm)
        boxes = [b for b in boxes if b[0] != HELMET_ID] + merged
        boxes = nms(boxes, 0.6)

        for b in boxes:
            stats["per_class"][SH17_NAMES[b[0]]] += 1

        stats[split] += 1
        stats["frames"] += 1
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
