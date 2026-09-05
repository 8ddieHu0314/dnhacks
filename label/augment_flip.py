"""Offline augmentation: add upside-down copies of the helmet frames.

The v2 model's weak spot is a helmet seen from an angle the clip never
shows (interior up, brim down). Ultralytics' flipud would flip every frame;
this only touches frames whose labels contain a helmet, and only in the
train split, so val stays an honest held-out set.

For each train frame with a helmet box it writes:
  <stem>_flipud.jpg   vertical flip        y -> 1 - y
  <stem>_rot180.jpg   180 degree rotation  x -> 1 - x, y -> 1 - y
Box width and height are unchanged under both, so the YOLO lines survive
with just the center moved.

Usage:
  .venv/bin/python label/augment_flip.py --src datasets/gloves_v2b --out datasets/gloves_v3
"""

import argparse
import glob
import os
import shutil

import cv2

HELMET_ID = 10  # SH17 class index, see label/gdino_label.py


def flip_lines(lines: list, mode: str) -> list:
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts[0], *map(float, parts[1:])
        y = 1.0 - y
        if mode == "rot180":
            x = 1.0 - x
        out.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset dir with images/ labels/ data.yaml")
    ap.add_argument("--out", required=True, help="new dataset dir (copied, then augmented)")
    ap.add_argument("--class-id", type=int, default=HELMET_ID)
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists, refusing to overwrite")
    shutil.copytree(args.src, args.out, ignore=shutil.ignore_patterns("*.cache", "preview"))

    # data.yaml carries an absolute path; point it at the new dir.
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path) as f:
        text = f.read().replace(os.path.abspath(args.src), os.path.abspath(args.out))
    with open(yaml_path, "w") as f:
        f.write(text)

    img_dir = os.path.join(args.out, "images", "train")
    lbl_dir = os.path.join(args.out, "labels", "train")
    written = 0
    sources = 0
    for lbl_path in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
        with open(lbl_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not any(ln.split()[0] == str(args.class_id) for ln in lines):
            continue
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = os.path.join(img_dir, stem + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"skip {stem}: no image")
            continue
        sources += 1
        for mode, flipped in (("flipud", cv2.flip(img, 0)), ("rot180", cv2.rotate(img, cv2.ROTATE_180))):
            cv2.imwrite(os.path.join(img_dir, f"{stem}_{mode}.jpg"), flipped, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(lbl_dir, f"{stem}_{mode}.txt"), "w") as f:
                f.write("\n".join(flip_lines(lines, mode)) + "\n")
            written += 1
    print(f"{sources} helmet frames in train, {written} augmented copies written to {args.out}")


if __name__ == "__main__":
    main()
