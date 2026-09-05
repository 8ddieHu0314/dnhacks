"""Compare SH17 weights on the gloves dataset's held-out val split, at the
production confidence, on the metrics the demo actually depends on:

  gloves_recall      fraction of labeled gloves boxes the model finds as gloves
  hands_on_glove     predicted `hands` boxes sitting on a labeled gloves box.
                     Each one is a false BARE_HANDS alarm on a gloved wearer.
  bare_hands_agree   frame-level agreement of the BARE_HANDS rule run on the
                     model's boxes vs run on the labels (the demo-level number)

Plus a regression check: rule verdicts on test_images/ and outputs/*.jpg must
not change between the old and new weights unless the frame contains the
yellow glove.

Exit code 0 when the new weights meet the gate, 1 otherwise, so a training
loop can be left to run against it.

Usage:
  .venv/bin/python label/eval_gloves.py --data datasets/gloves_v1/data.yaml \
      --old weights/yolo8s.pt --new runs/gloves/v1/weights/best.pt
"""

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import Detection  # noqa: E402
from core.rules import bare_hands, iou  # noqa: E402
from label.gdino_label import GLOVES_ID, HANDS_ID, HELMET_ID, SH17_NAMES  # noqa: E402


def read_labels(path, w, h):
    boxes = []
    if not os.path.isfile(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            boxes.append((cls, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return boxes


def to_detections(preds, w=None, h=None):
    return [Detection(class_name=SH17_NAMES[c], conf=1.0, box=b, source="sh17") for c, b in preds]


def predict(model, frame, conf, device):
    res = model.predict(frame, conf=conf, device=device, verbose=False)[0]
    return [(int(b.cls[0]), tuple(float(v) for v in b.xyxy[0].tolist())) for b in res.boxes]


def eval_val(model, data_dir, conf, device):
    img_dir = os.path.join(data_dir, "images", "val")
    lbl_dir = os.path.join(data_dir, "labels", "val")
    names = sorted(os.listdir(img_dir))
    gt_gloves = found_gloves = hands_on_glove = gloves_on_helmet = gt_helmets = agree = frames = 0
    for name in names:
        frame = cv2.imread(os.path.join(img_dir, name))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = read_labels(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), w, h)
        pred = predict(model, frame, conf, device)
        gt_g = [b for c, b in gt if c == GLOVES_ID]
        gt_hm = [b for c, b in gt if c == HELMET_ID]
        pr_g = [b for c, b in pred if c == GLOVES_ID]
        pr_h = [b for c, b in pred if c == HANDS_ID]
        gt_gloves += len(gt_g)
        gt_helmets += len(gt_hm)
        found_gloves += sum(1 for g in gt_g if any(iou(g, p) > 0.5 for p in pr_g))
        hands_on_glove += sum(1 for p in pr_h if any(iou(g, p) > 0.5 for g in gt_g))
        # The v1 failure: a yellow hard hat called gloves. Count predicted
        # gloves boxes that sit on a labeled helmet.
        gloves_on_helmet += sum(1 for p in pr_g if any(iou(hm, p) > 0.5 for hm in gt_hm))
        rule_gt = bare_hands(to_detections(gt)).active
        rule_pr = bare_hands(to_detections(pred)).active
        agree += int(rule_gt == rule_pr)
        frames += 1
    return {
        "frames": frames,
        "gt_gloves": gt_gloves,
        "gloves_recall": round(found_gloves / gt_gloves, 3) if gt_gloves else None,
        "hands_on_glove": hands_on_glove,
        "gt_helmets": gt_helmets,
        "gloves_on_helmet": gloves_on_helmet,
        "bare_hands_agree": round(agree / frames, 3) if frames else None,
    }


def regression(model, paths, conf, device):
    out = {}
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        pred = predict(model, frame, conf, device)
        r = bare_hands(to_detections(pred))
        out[os.path.basename(p)] = {"active": r.active, "detail": r.detail,
                                    "classes": sorted({SH17_NAMES[c] for c, _ in pred})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset data.yaml or dir")
    ap.add_argument("--old", default="weights/yolo8s.pt")
    ap.add_argument("--new", required=True)
    ap.add_argument("--conf", type=float, default=0.4, help="production confidence")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--max-hands-on-glove", type=int, default=0)
    ap.add_argument("--report", default="", help="write JSON report here")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_dir = args.data if os.path.isdir(args.data) else os.path.dirname(args.data)
    reg_paths = sorted(glob.glob("test_images/*.jpg") + glob.glob("test_images/*.png") + glob.glob("outputs/*.jpg"))

    report = {"conf": args.conf, "val": {}, "regression": {}}
    for tag, path in (("old", args.old), ("new", args.new)):
        model = YOLO(path)
        report["val"][tag] = {"weights": path, **eval_val(model, data_dir, args.conf, args.device)}
        report["regression"][tag] = regression(model, reg_paths, args.conf, args.device)

    changed = [k for k in report["regression"]["old"]
               if report["regression"]["old"][k]["active"] != report["regression"]["new"].get(k, {}).get("active")]
    report["regression_changed_verdicts"] = changed

    new = report["val"]["new"]
    gate = {
        "gloves_recall_ok": (new["gloves_recall"] or 0) >= args.min_recall,
        "hands_on_glove_ok": new["hands_on_glove"] <= args.max_hands_on_glove,
        "gloves_on_helmet_ok": new["gloves_on_helmet"] == 0,
        "regression_ok": len(changed) == 0,
    }
    report["gate"] = gate
    report["pass"] = all(gate.values())

    print(json.dumps(report["val"], indent=2))
    print("regression verdicts changed:", changed or "none")
    print("gate:", gate, "->", "PASS" if report["pass"] else "FAIL")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
