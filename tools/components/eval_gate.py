"""Eval gate for a component detector ONNX, run through the relay's own detect.py so the numbers
reflect exactly what the relay will do (same preprocessing, NMS and thresholds).

Measures on the held-out val split of our own webcam frames:
- recall: labeled parts found (IoU >= 0.4) at the relay's trigger threshold
- false positives per negative frame (frames with no part), the failure the Universe model had
- edge-touching box rate among detections (the other Universe failure)
Exits 0 when recall >= --min-recall and FP per negative frame <= --max-fp.

    services/relay_receiver/.venv/bin/python tools/components/eval_gate.py --onnx weights/components_v1.onnx --data datasets/components_v1
    ... --onnx weights/components_yolov8.onnx   # the old Universe model, for comparison (any class counts as a box)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "relay_receiver"))
os.environ.setdefault("DETECT_CONF", "0.25")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data", required=True, help="dataset dir with images/val and labels/val")
    ap.add_argument("--conf", type=float, default=0.6, help="trigger threshold to evaluate at (relay det_min_conf)")
    ap.add_argument("--min-recall", type=float, default=0.8)
    ap.add_argument("--max-fp", type=float, default=0.05)
    a = ap.parse_args()
    os.environ["DETECT_ONNX"] = str(Path(a.onnx).resolve())
    import detect
    print(detect.load())
    imgs = sorted(Path(a.data, "images", "val").glob("*.jpg"))
    hit = tot = 0; fp_neg = 0; negs = 0; edge = 0; n_det = 0; pos_frames_found = 0; pos_frames = 0
    for img in imgs:
        lab = Path(a.data, "labels", "val", img.stem + ".txt")
        gts = []
        if lab.exists():
            for l in lab.read_text().splitlines():
                c, cx, cy, w, h = map(float, l.split()[:5])
                gts.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        dets = [d for d in detect.detect(img.read_bytes(), conf=0.1) if d["conf"] >= a.conf]
        for d in dets:
            n_det += 1
            x1, y1, x2, y2 = d["box"]
            if x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99:
                edge += 1
        if not gts:
            negs += 1; fp_neg += len(dets)
            continue
        pos_frames += 1
        found = 0
        for g in gts:
            tot += 1
            if any(iou(g, d["box"]) >= 0.4 for d in dets):
                hit += 1; found += 1
        pos_frames_found += found > 0
    recall = hit / tot if tot else 0
    fp = fp_neg / negs if negs else 0
    print(f"val frames {len(imgs)}: {pos_frames} with parts, {negs} negatives")
    print(f"recall@conf{a.conf}: {hit}/{tot} = {recall:.0%} boxes, {pos_frames_found}/{pos_frames} frames with at least one hit")
    print(f"false positives on negative frames: {fp_neg} over {negs} frames = {fp:.2f} per frame")
    print(f"edge-touching detections: {edge}/{n_det}")
    ok = recall >= a.min_recall and fp <= a.max_fp
    print("GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
