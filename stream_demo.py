"""Step 1 demo: prove the spine is safe for a live 10 fps feed, with no
phone or glasses needed.

Default source is a synthetic "slideshow" built from test_images/ that
walks bare hands -> gloves -> bare hands, so the 2 second debounce should
fire about 2 s into segment 1, clear during segment 2, and fire again about
2 s into segment 3.

A producer thread paces frames into a Mailbox at --fps using wall-clock
timestamps. A StreamRunner thread drains the Mailbox through the Spine.
The Roboflow plugin (250 ms to 2 s per hosted call) is wrapped in a
CachedPlugin so it never sits on the hot path: the mailbox and the cache
are what keep per-frame spine latency low regardless of source pace.

With --show, a "detect slow, draw fast" render loop runs in the main
thread: it redraws every produced frame at the feed rate, overlaying the
most recent judged boxes (via LatestResult) and a small HUD, so the window
looks live even though the spine may only judge a fraction of those
frames. Press q to quit the window early. If no cv2 GUI window can open in
this environment, it falls back to writing annotated frames to
/tmp/overlay_frames/ every 15th displayed frame.

Usage:
  .venv/bin/python stream_demo.py [--source slideshow|<path.mp4>|<webcam index>]
                                   [--fps 10] [--seconds 12]
                                   [--roboflow/--no-roboflow] [--speak/--no-speak]
                                   [--debounce 2.0] [--model yolo8s] [--conf 0.4]
                                   [--show]
"""

import argparse
import os
import statistics
import sys
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from core.detectors import SH17Detector
from core.spine import Spine, annotate, roboflow_tools_plugin
from core.stream import CachedPlugin, LatestResult, Mailbox, Speaker, StreamRunner

load_dotenv()

SLIDESHOW_IMAGES = [
    "test_images/bare_hands_wrench_1.jpg",
    "test_images/gloves_drill_1.jpg",
    "test_images/bare_hands_wrench_1.jpg",
]
SLIDESHOW_SEGMENT_SECONDS = 4.0
LONG_SIDE = 640


def resize_long_side(frame, target=LONG_SIDE):
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side == target:
        return frame
    scale = target / float(long_side)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(frame, new_size)


class SlideshowSource:
    """Synthetic stream: a still image held for SLIDESHOW_SEGMENT_SECONDS,
    then swapped, to prove the debounce fires / clears / fires again.
    """

    def __init__(self):
        self.frames = []
        for path in SLIDESHOW_IMAGES:
            frame = cv2.imread(path)
            if frame is None:
                raise SystemExit(f"could not read slideshow image: {path}")
            self.frames.append(resize_long_side(frame))

    def frame_at(self, elapsed: float):
        idx = min(int(elapsed // SLIDESHOW_SEGMENT_SECONDS), len(self.frames) - 1)
        return self.frames[idx]

    def release(self) -> None:
        pass


class VideoSource:
    """Wraps cv2.VideoCapture for a video file or a webcam index. Video
    files loop back to the start if they run out before --seconds elapses;
    a webcam just keeps reading.
    """

    def __init__(self, path_or_index):
        self.cap = cv2.VideoCapture(path_or_index)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open video source: {path_or_index}")
        self._last_frame = None

    def frame_at(self, elapsed: float):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return self._last_frame
        frame = resize_long_side(frame)
        self._last_frame = frame
        return frame

    def release(self) -> None:
        self.cap.release()


def make_source(source_arg):
    if source_arg == "slideshow":
        return SlideshowSource()
    try:
        index = int(source_arg)
        return VideoSource(index)
    except ValueError:
        return VideoSource(source_arg)


class _FrameBox:
    """Thread-safe holder for the most recently produced raw frame.

    Demo-only plumbing for --show: separate from the Mailbox (which the
    StreamRunner drains for judging), this just lets the main-thread render
    loop see the newest produced frame to draw the HUD on top of.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


def producer_loop(
    mailbox: Mailbox,
    source,
    fps: float,
    seconds: float,
    stats: dict,
    frame_box: "_FrameBox" = None,
    stop_event: threading.Event = None,
) -> None:
    period = 1.0 / fps
    start = time.time()
    next_t = start
    count = 0
    while True:
        now = time.time()
        elapsed = now - start
        if elapsed >= seconds or (stop_event is not None and stop_event.is_set()):
            break
        frame = source.frame_at(elapsed)
        if frame is not None:
            mailbox.put(frame, time.time())
            if frame_box is not None:
                frame_box.set(frame)
            count += 1
        next_t += period
        sleep_for = next_t - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
    stats["frames_produced"] = count


def render_loop(
    frame_box: "_FrameBox",
    latest_result: LatestResult,
    mailbox: Mailbox,
    fps: float,
    seconds: float,
    stats: dict,
    stop_event: threading.Event,
) -> None:
    """Runs in the MAIN thread (cv2 GUI calls must happen there).

    Redraws every produced frame at ~feed rate, overlaying the last known
    judged boxes and a HUD (feed fps, judged fps, dropped count, active
    rule timers, last sentence). This decouples display rate from judge
    rate: the spine may only judge a fraction of the displayed frames, but
    the window reuses the last known boxes so it still looks live.
    """
    window_name = "PPE Compliance (Step 1 demo)"
    headless = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
    except Exception as exc:
        headless = True
        print(
            f"[show] cv2 GUI window could not open here ({exc}); "
            f"falling back to headless mode: writing frames to /tmp/overlay_frames/"
        )

    out_dir = "/tmp/overlay_frames"
    if headless:
        os.makedirs(out_dir, exist_ok=True)

    period = 1.0 / fps
    start = time.time()
    next_t = start
    displayed = 0

    while True:
        now = time.time()
        elapsed = now - start
        if elapsed >= seconds or stop_event.is_set():
            break

        frame = frame_box.get()
        if frame is None:
            time.sleep(0.01)
            continue

        result, _ = latest_result.get()
        detections = result.detections if result is not None else []
        canvas = annotate(frame, detections)

        feed_fps = displayed / max(elapsed, 1e-6)
        judged_fps = stats["frames_processed"] / max(elapsed, 1e-6)
        active = []
        sentence_text = "-"
        if result is not None:
            active = [
                f"{r.rule}={result.timers.get(r.rule, 0.0):.1f}s"
                for r in result.rule_results
                if r.active
            ]
            sentence_text = result.sentence or "-"

        hud_lines = [
            f"feed {feed_fps:4.1f}fps  judged {judged_fps:4.1f}fps  dropped {mailbox.dropped}",
            f"active: {', '.join(active) if active else '-'}",
            f"say: {sentence_text}",
        ]
        y = 20
        for line in hud_lines:
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1, cv2.LINE_AA)
            y += 20

        displayed += 1

        if headless:
            if displayed % 15 == 0:
                cv2.imwrite(os.path.join(out_dir, f"frame_{displayed:05d}.jpg"), canvas)
        else:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break

        next_t += period
        sleep_for = next_t - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)

    if not headless:
        cv2.destroyAllWindows()

    stats["frames_displayed"] = displayed
    stats["show_headless"] = headless


def make_on_result(stats: dict, speaker: Speaker, t0: float):
    def on_result(result, ts, latency_ms, dropped):
        stats["frames_processed"] += 1
        stats["latencies_ms"].append(latency_ms)
        t_rel = ts - t0

        active = [
            f"{r.rule}={result.timers.get(r.rule, 0.0):.2f}s"
            for r in result.rule_results
            if r.active
        ]
        fired_names = [f["rule"] for f in result.fired]

        line = (
            f"t=+{t_rel:6.2f}s frame#{stats['frames_processed']:04d} "
            f"latency={latency_ms:7.2f}ms dropped={dropped} "
            f"active=[{', '.join(active) if active else '-'}]"
        )
        if fired_names:
            line += f" FIRED={fired_names}"
            stats["fired_events"].append((t_rel, list(fired_names)))
        if result.sentence:
            line += f' sentence="{result.sentence}"'
        print(line)

        if result.sentence and speaker.say(result.sentence):
            stats["spoken_events"].append((t_rel, result.sentence))

    return on_result


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1 live-feed spine demo")
    parser.add_argument(
        "--source", default="slideshow", help='"slideshow", a video path, or a webcam index'
    )
    parser.add_argument("--fps", type=float, default=10.0, help="feed rate in frames/sec")
    parser.add_argument("--seconds", type=float, default=12.0, help="how long to run")
    parser.add_argument("--roboflow", dest="roboflow", action="store_true", default=True)
    parser.add_argument("--no-roboflow", dest="roboflow", action="store_false")
    parser.add_argument("--speak", dest="speak", action="store_true", default=True)
    parser.add_argument("--no-speak", dest="speak", action="store_false")
    parser.add_argument("--debounce", type=float, default=2.0)
    parser.add_argument("--model", default="yolo8s")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="open a live overlay window (falls back to /tmp/overlay_frames/ if no GUI is available)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    source = make_source(args.source)

    sh17 = SH17Detector(f"weights/{args.model}.pt", device="mps", conf=args.conf)

    plugins = []
    cached_roboflow = None
    if args.roboflow:
        base_plugin = roboflow_tools_plugin(conf=args.conf)
        cached_roboflow = CachedPlugin(base_plugin, every_n_frames=5, ttl_seconds=1.0)
        plugins.append((cached_roboflow, 1))  # CachedPlugin gates its own cadence

    spine = Spine(sh17=sh17, plugins=plugins, debounce_seconds=args.debounce)

    mailbox = Mailbox()
    speaker = Speaker(enabled=args.speak)
    latest_result = LatestResult()

    stats = {
        "frames_produced": 0,
        "frames_processed": 0,
        "latencies_ms": [],
        "fired_events": [],
        "spoken_events": [],
        "frames_displayed": 0,
        "show_headless": False,
    }

    t0 = time.time()
    on_result = make_on_result(stats, speaker, t0)
    runner = StreamRunner(spine, mailbox, on_result, latest_result=latest_result)
    runner_thread = threading.Thread(target=runner.run, daemon=True)
    runner_thread.start()

    frame_box = _FrameBox() if args.show else None
    stop_event = threading.Event()

    producer_thread = threading.Thread(
        target=producer_loop,
        args=(mailbox, source, args.fps, args.seconds, stats),
        kwargs={"frame_box": frame_box, "stop_event": stop_event},
    )
    producer_thread.start()

    if args.show:
        # Runs in this (the main) thread: cv2 GUI calls must happen here.
        render_loop(frame_box, latest_result, mailbox, args.fps, args.seconds, stats, stop_event)
        stop_event.set()

    producer_thread.join()

    # Give the runner a moment to drain the last frame(s) before stopping it.
    time.sleep(0.5)
    runner.stop()
    runner_thread.join(timeout=2.0)
    source.release()

    print("\n--- Summary ---")
    print(f"frames produced:  {stats['frames_produced']}")
    print(f"frames processed: {stats['frames_processed']}")
    print(f"frames dropped:   {mailbox.dropped}")

    latencies = stats["latencies_ms"]
    if latencies:
        median_lat = statistics.median(latencies)
        p95_lat = float(np.percentile(latencies, 95))
    else:
        median_lat = float("inf")
        p95_lat = float("inf")
    print(f"spine latency ms: median={median_lat:.2f} p95={p95_lat:.2f}")

    if cached_roboflow is not None:
        last_lat = cached_roboflow.last_latency_ms
        last_lat_str = f"{last_lat:.1f}ms" if last_lat is not None else "n/a"
        print(f"roboflow plugin:  runs={cached_roboflow.runs} last_latency={last_lat_str}")
    else:
        print("roboflow plugin:  disabled")

    print("spoken events:")
    if stats["spoken_events"]:
        for t_rel, sentence in stats["spoken_events"]:
            print(f'  t=+{t_rel:.2f}s "{sentence}"')
    else:
        print("  (none)")

    if args.show:
        displayed = stats["frames_displayed"]
        judged = stats["frames_processed"]
        ratio = (judged / displayed) if displayed else 0.0
        mode = "headless (/tmp/overlay_frames)" if stats["show_headless"] else "gui window"
        print(f"show mode:         {mode}")
        print(f"displayed frames:  {displayed}")
        print(f"judged frames:     {judged}")
        print(f"judged/displayed:  {ratio:.2f}")

    ok = len(stats["spoken_events"]) >= 1 and median_lat < 150.0
    if ok:
        print("\nRESULT: PASS")
        sys.exit(0)
    else:
        reasons = []
        if len(stats["spoken_events"]) < 1:
            reasons.append("no sentence was spoken")
        if median_lat >= 150.0:
            reasons.append(f"median spine latency {median_lat:.2f}ms >= 150ms")
        print(f"\nRESULT: FAIL ({'; '.join(reasons)})")
        sys.exit(1)


if __name__ == "__main__":
    main()
