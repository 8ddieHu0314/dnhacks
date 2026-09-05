"""Smoke test: run one frame through the Spine and print the result.

Usage: .venv/bin/python smoke.py [image_path]
Defaults to the first jpg in test_images/.
Exit 0 if there is at least one detection, else exit 1.
"""

import glob
import sys

import cv2
from dotenv import load_dotenv

from core.detectors import SH17Detector
from core.spine import Spine, roboflow_tools_plugin

load_dotenv()


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        candidates = sorted(glob.glob("test_images/*.jpg"))
        if not candidates:
            print("No jpg found in test_images/ and no path given.")
            sys.exit(1)
        image_path = candidates[0]

    print(f"Image: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    sh17 = SH17Detector("weights/yolo8s.pt", device="mps", conf=0.4)
    plugin = roboflow_tools_plugin()
    spine = Spine(sh17=sh17, plugins=[(plugin, 1)], debounce_seconds=0)
    spine.reset()

    result = spine.process(frame, dt=1.0)

    print("\n--- Detections ---")
    for d in result.detections:
        print(f"  {d.source:9s} {d.class_name:20s} conf={d.conf:.2f} box={tuple(round(v, 1) for v in d.box)}")

    print("\n--- Rule results ---")
    for r in result.rule_results:
        status = "ACTIVE" if r.active else ("PASS" if r.applicable else "UNKNOWN")
        print(f"  {r.rule:20s} {status:8s} severity={r.severity:8s} detail={r.detail}")

    print("\n--- Fired ---")
    if result.fired:
        for f in result.fired:
            print(f"  {f}")
    else:
        print("  (none)")

    print(f"\nSentence: {result.sentence}")
    print(f"Errors: {result.errors}")

    if len(result.detections) > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
