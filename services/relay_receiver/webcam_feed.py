"""Feed the Mac webcam into the relay as if it were the phone.

Lets you test the dashboard, the component detector and the Claude path on localhost with a
part held up to the laptop camera, no glasses needed. Frames go to POST /frame with the same
capture timestamp header the phone sends.

    python3 webcam_feed.py                      # camera 0, 8 fps, 960 px wide, to localhost:8787
    python3 webcam_feed.py --fps 5 --width 640 --camera 1 --url http://localhost:8787
    python3 webcam_feed.py --rotate 90          # mimic the glasses' portrait frames
    python3 webcam_feed.py --save sessions/components_a   # also record every frame (training data capture)

Needs OpenCV; the repo-root .venv or .venv-inf have it (the relay venv does not, on purpose).
macOS will ask for camera permission for your terminal the first time.
"""
import argparse
import os
import sys
import time

import cv2
import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8787")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="rotate frames clockwise before sending")
    ap.add_argument("--reactive", default="parts", choices=["off", "parts", "scene"], help="hands-free mode to enable on the relay")
    ap.add_argument("--save", default="", help="also write every frame as JPEG under <dir>/frames/ (training data capture)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"could not open camera {a.camera} (check macOS camera permission for your terminal)")
    client = httpx.Client(base_url=a.url, timeout=5)
    if a.reactive != "off":
        try:
            client.post("/reactive", json={"enabled": True, "mode": a.reactive})
        except Exception as e:
            print("relay not reachable yet:", e)
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(a.rotate)
    period = 1.0 / a.fps
    sent, failed, t_report = 0, 0, time.time()
    save_dir = None
    if a.save:
        save_dir = os.path.join(a.save, "frames")
        os.makedirs(save_dir, exist_ok=True)
        print(f"recording every frame to {save_dir}")
    print(f"webcam {a.camera} -> {a.url}/frame at {a.fps} fps, {a.width}px wide, rotate {a.rotate}. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            h, w = frame.shape[:2]
            if w > a.width:
                frame = cv2.resize(frame, (a.width, int(h * a.width / w)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            if save_dir:
                with open(os.path.join(save_dir, f"{sent:06d}.jpg"), "wb") as f:
                    f.write(jpg.tobytes())
            try:
                client.post("/frame", content=jpg.tobytes(), headers={"x-capture-ts": str(int(t0 * 1000))})
                sent += 1
            except Exception:
                failed += 1
            if time.time() - t_report > 5:
                try:
                    d = client.get("/detections").json()
                    boxes = ", ".join(f"{b.get('display', b['label'])} {b['conf']:.2f}" for b in d["boxes"]) or "none"
                    print(f"sent {sent} (failed {failed}) | detector {d['ms']} ms | boxes: {boxes}")
                except Exception as e:
                    print(f"sent {sent} (failed {failed}) | relay unreachable: {e}")
                t_report = time.time()
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
