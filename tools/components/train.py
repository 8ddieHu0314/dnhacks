"""Fine-tune a single-class YOLOv8 "component" detector on Universe images plus our own webcam frames.

    .venv/bin/python tools/components/train.py --own datasets/components_v1 --universe datasets/universe_1class \
        --name components_v1 --epochs 25

Writes a merged dataset under datasets/<name>_merged/ (symlinks), trains from weights/yolov8n.pt on MPS,
exports best.pt to ONNX (opset 12, static 640, the same shape detect.py already handles) and copies it
to weights/<name>.onnx with a one-line classes file. Run tools/components/eval_gate.py before swapping it in.
"""
import argparse
import os
import shutil
from pathlib import Path


def link_split(src: Path, dst: Path, split: str, prefix: str):
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted((src / "images" / split).glob("*.jpg")):
        lab = src / "labels" / split / (img.stem + ".txt")
        os.symlink(img.resolve(), dst / "images" / split / f"{prefix}_{img.name}")
        if lab.exists():
            os.symlink(lab.resolve(), dst / "labels" / split / f"{prefix}_{img.stem}.txt")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", required=True)
    ap.add_argument("--universe", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="weights/yolov8n.pt")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    merged = Path("datasets") / f"{a.name}_merged"
    if merged.exists():
        shutil.rmtree(merged)
    counts = {}
    for split in ("train", "val"):
        counts[f"own_{split}"] = link_split(Path(a.own), merged, split, "own")
        if a.universe:
            counts[f"universe_{split}"] = link_split(Path(a.universe), merged, split, "uni")
    (merged / "data.yaml").write_text(f"path: {merged.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['component']\n")
    print("merged dataset:", counts, flush=True)

    from ultralytics import YOLO
    model = YOLO(a.base)
    model.train(data=str(merged / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project="runs/components", name=a.name, exist_ok=True, patience=a.patience,
                # webcam footage: parts are small, hands move, colour is not the cue
                mosaic=1.0, scale=0.6, degrees=15, fliplr=0.5, flipud=0.2, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5,
                lr0=0.005, workers=4, plots=True, verbose=True)
    best = next(Path("runs").rglob(f"components/{a.name}/weights/best.pt"))  # ultralytics nests project under runs/detect/
    print("best:", best, flush=True)
    onnx = YOLO(str(best)).export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True, dynamic=False)
    out = Path("weights") / f"{a.name}.onnx"
    shutil.copy(onnx, out)
    (Path("weights") / f"{a.name}.classes.txt").write_text("component\n")
    print("exported:", out, flush=True)


if __name__ == "__main__":
    main()
