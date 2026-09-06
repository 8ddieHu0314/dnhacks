"""Claude verifies Grounding DINO's candidate boxes, frame by frame.

Grounding DINO proposes boxes (recall), Claude keeps only the ones that are electronic components
(precision) and flags frames where a part is visible but unboxed (those are dropped, not used as
negatives). Input is the raw_detections.json written by gdino_label.py.

    services/relay_receiver/.venv/bin/python tools/components/claude_verify.py \
        --raw datasets/components_v1/raw_detections.json --out datasets/components_v2 --box-thr 0.2

Needs ANTHROPIC_API_KEY (read from services/relay_receiver/.env if not set).
"""
import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdino_label import contact_sheet, decide  # noqa: E402

MODEL = os.environ.get("VERIFY_MODEL", "claude-sonnet-5")
PROMPT = ("This is a webcam frame of a person showing Arduino kit parts. Numbered green boxes are candidate detections.\n"
          "Which boxes contain an electronic component or module: circuit board, sensor module, breadboard, servo, "
          "display, LED matrix, keypad, wire or jumper bundle, bag of resistors or LEDs? "
          "Exclude: printed graphics on clothing, furniture, shelf compartments, the plastic organizer tray, watch, phone, "
          "glasses, hands with nothing in them.\n"
          "Answer in exactly two lines:\n"
          "keep: <comma-separated box numbers, or none>\n"
          "missed: <yes if an electronic component is clearly visible with no box on it, else no>")


def _env_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = Path(__file__).resolve().parents[2] / "services" / "relay_receiver" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()


def render(path, boxes, max_w=896):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h * max_w / w)))
        w, h = im.size
    d = ImageDraw.Draw(im)
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = b["box"]
        d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(40, 255, 90), width=4)
        d.rectangle([x1 * w, y1 * h - 22, x1 * w + 28, y1 * h], fill=(40, 255, 90))
        d.text((x1 * w + 6, y1 * h - 20), str(i), fill=(0, 0, 0))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=85)
    return base64.standard_b64encode(out.getvalue()).decode()


async def ask(client, sem, b64):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.messages.create(
                    model=MODEL, max_tokens=40,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": PROMPT}]}])
                return "".join(b.text for b in r.content if b.type == "text")
            except Exception as e:
                if attempt == 3:
                    return f"error: {e}"
                await asyncio.sleep(2 * (attempt + 1))


def parse(text, n):
    keep = set()
    m = re.search(r"keep:\s*(.*)", text, re.I)
    if m and "none" not in m.group(1).lower():
        keep = {int(x) for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= n}
    missed = bool(re.search(r"missed:\s*yes", text, re.I))
    return keep, missed


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-thr", type=float, default=0.20)
    ap.add_argument("--max-area", type=float, default=0.20)
    ap.add_argument("--max-aspect", type=float, default=3.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-chunks", type=int, default=5)
    ap.add_argument("--n-chunks", type=int, default=20)
    a = ap.parse_args()
    _env_key()
    import anthropic
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(a.concurrency)

    frames = json.load(open(a.raw))["frames"]
    if a.limit:
        frames = frames[:a.limit]
    cands = [decide(fr["items"], a.box_thr, 0.002, a.max_area, max_aspect=a.max_aspect) for fr in frames]
    print(f"{len(frames)} frames, {sum(len(c) for c in cands)} candidate boxes -> {MODEL} x{a.concurrency}", flush=True)
    t0 = time.time()
    # frames with no candidates still get asked (missed?) so clean negatives are verified too
    b64s = [render(fr["path"], c) for fr, c in zip(frames, cands)]
    answers = await asyncio.gather(*(ask(client, sem, b) for b in b64s))
    print(f"verified in {time.time() - t0:.0f}s", flush=True)

    out = a.out
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        Path(out, sub).mkdir(parents=True, exist_ok=True)
        for f in Path(out, sub).glob("*"):
            f.unlink()
    n = len(frames); chunk = max(1, n // a.n_chunks)
    stats = {"train": 0, "val": 0, "boxes": 0, "negatives": 0, "dropped_missed": 0, "errors": 0,
             "candidates": sum(len(c) for c in cands), "rejected_by_claude": 0}
    kept_all, rejected_all, log = [], [], []
    for i, (fr, c, ans) in enumerate(zip(frames, cands, answers)):
        if ans.startswith("error"):
            stats["errors"] += 1
            continue
        keep_idx, missed = parse(ans, len(c))
        kept = [b for j, b in enumerate(c, 1) if j in keep_idx]
        rejected = [b for j, b in enumerate(c, 1) if j not in keep_idx]
        stats["rejected_by_claude"] += len(rejected)
        log.append({"path": fr["path"], "answer": ans.strip(), "kept": len(kept), "missed": missed})
        if rejected:
            rejected_all.append((fr["path"], rejected, f"{i:05d} REJECTED"))
        if missed and not kept:
            stats["dropped_missed"] += 1
            continue
        if missed:
            # a part is unboxed: keep the frame's positives but it must not teach "no part here"
            stats["dropped_missed"] += 1
            continue
        split = "val" if (i // chunk) % a.val_chunks == a.val_chunks - 1 else "train"
        stem = f"{i:05d}"
        shutil.copy(fr["path"], Path(out, "images", split, stem + ".jpg"))
        with open(Path(out, "labels", split, stem + ".txt"), "w") as f:
            for b in kept:
                x1, y1, x2, y2 = b["box"]
                f.write(f"0 {(x1 + x2) / 2:.5f} {(y1 + y2) / 2:.5f} {x2 - x1:.5f} {y2 - y1:.5f}\n")
        stats[split] += 1; stats["boxes"] += len(kept); stats["negatives"] += (not kept)
        kept_all.append((fr["path"], kept, f"{stem} {split}" + ("" if kept else " NEG")))
    Path(out, "data.yaml").write_text(f"path: {Path(out).resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['component']\n")
    json.dump(log, open(Path(out, "claude_verify.json"), "w"), indent=1)
    random.seed(0)
    groups = {"sample": kept_all[::10], "negatives": random.sample([k for k in kept_all if not k[1]], min(36, stats["negatives"])),
              "rejected": rejected_all[::max(1, len(rejected_all) // 36)][:36]}
    for name, group in groups.items():
        for j in range(0, len(group), 36):
            contact_sheet(group[j:j + 36], str(Path(out, "preview", f"{name}_{j // 36:02d}.jpg")))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
