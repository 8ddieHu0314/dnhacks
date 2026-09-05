"""Side-by-side old vs new weights on a set of images, at the production
confidence, with the BARE_HANDS verdict printed on each half. Writes one PNG
per input image so a human can see the difference, not just read a number.

Usage:
  .venv/bin/python label/compare_weights.py --old weights/yolo8s.pt --new weights/yolo8s-gloves.pt \
      --out compare/ image1.jpg image2.png ...
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors import Detection  # noqa: E402
from core.rules import bare_hands  # noqa: E402
from label.gdino_label import GLOVES_ID, HANDS_ID, SH17_NAMES  # noqa: E402


def annotate(model, frame, conf, device, title):
    res = model.predict(frame, conf=conf, device=device, verbose=False)[0]
    out = frame.copy()
    dets = []
    for b in res.boxes:
        cls = int(b.cls[0])
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        score = float(b.conf[0])
        dets.append(Detection(class_name=SH17_NAMES[cls], conf=score, box=(x1, y1, x2, y2), source="sh17"))
        if cls == GLOVES_ID:
            color, thick = (0, 200, 255), 3
        elif cls == HANDS_ID:
            color, thick = (0, 0, 255), 3
        else:
            color, thick = (0, 255, 0), 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        if cls in (GLOVES_ID, HANDS_ID):
            cv2.putText(out, f"{SH17_NAMES[cls]} {score:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    r = bare_hands(dets)
    verdict = f"BARE_HANDS {'FIRES' if r.active else 'quiet'}: {r.detail}"
    vcolor = (0, 0, 255) if r.active else (0, 200, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 58), (20, 20, 20), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, verdict, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vcolor, 2, cv2.LINE_AA)
    return out, r.active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="weights/yolo8s.pt")
    ap.add_argument("--new", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    from ultralytics import YOLO

    old, new = YOLO(args.old), YOLO(args.new)
    os.makedirs(args.out, exist_ok=True)
    for p in args.images:
        frame = cv2.imread(p)
        if frame is None:
            print(f"skip {p}: unreadable")
            continue
        a, fa = annotate(old, frame, args.conf, args.device, f"OLD {os.path.basename(args.old)}")
        b, fb = annotate(new, frame, args.conf, args.device, f"NEW {os.path.basename(args.new)}")
        side = cv2.hconcat([a, b])
        dst = os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + "_old_vs_new.jpg")
        cv2.imwrite(dst, side, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{os.path.basename(p)}: old BARE_HANDS={'FIRES' if fa else 'quiet'}  new BARE_HANDS={'FIRES' if fb else 'quiet'}  -> {dst}")


if __name__ == "__main__":
    main()
