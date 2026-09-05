"""Live-feed plumbing for Step 1: a latest-frame-wins mailbox, a background
runner that drains it through a Spine, a cache wrapper that keeps a slow
plugin (hosted Roboflow calls: 250 ms to 2 s) off the hot path, and a
rate-limited speaker.

Nothing here is model-specific. This module only manages threads, time, and
queues so a 10 fps live feed never backs up behind a slow plugin call.
"""

import subprocess
import threading
import time

from core.spine import FrameState


class Mailbox:
    """A one-slot, latest-frame-wins queue.

    A producer calling put() faster than the consumer drains it never
    builds a backlog: each put() replaces whatever frame is waiting, so the
    consumer always sees the newest frame, never a stale one from a queue.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._ts = None
        self._has_frame = False
        self.dropped = 0

    def put(self, frame, ts) -> None:
        with self._cond:
            if self._has_frame:
                self.dropped += 1
            self._frame = frame
            self._ts = ts
            self._has_frame = True
            self._cond.notify()

    def get(self, timeout=None):
        """Block until a frame is available and return (frame, ts).

        If timeout elapses with nothing available, returns (None, None) so
        a caller (e.g. StreamRunner) can periodically check whether it
        should keep waiting.
        """
        with self._cond:
            got = self._cond.wait_for(lambda: self._has_frame, timeout=timeout)
            if not got:
                return None, None
            frame, ts = self._frame, self._ts
            self._frame = None
            self._ts = None
            self._has_frame = False
            return frame, ts


class CachedPlugin:
    """Wraps a plugin callable (frame_bgr, state) -> None so a slow plugin
    (a hosted Roboflow call) never blocks the live frame loop.

    The plugin runs on a background thread at most every `every_n_frames`
    calls, or when the worker goes idle again, whichever is later. Every
    call to __call__ merges the most recently finished result into
    state.detections, but only if that result is still younger than
    `ttl_seconds`; a stale or missing result contributes nothing rather
    than blocking the caller.
    """

    def __init__(self, plugin, every_n_frames: int = 5, ttl_seconds: float = 1.0):
        self._plugin = plugin
        self.every_n_frames = max(1, every_n_frames)
        self.ttl_seconds = ttl_seconds

        self._lock = threading.Lock()
        self._thread = None
        # Start saturated so the very first frame triggers a run instead of
        # waiting every_n_frames frames for the first result.
        self._frames_since_submit = self.every_n_frames

        self._cached_detections = []
        self._cached_errors = []
        self._cached_ts = None

        self.last_latency_ms = None
        self.runs = 0

    def _worker_idle(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def _run(self, frame_bgr, frame_index) -> None:
        start = time.perf_counter()
        private_state = FrameState(
            detections=[],
            attributes={},
            timers={},
            frame_index=frame_index,
        )
        try:
            self._plugin(frame_bgr, private_state)
        except Exception as exc:
            private_state.attributes.setdefault("errors", []).append(str(exc))
        latency_ms = (time.perf_counter() - start) * 1000.0

        with self._lock:
            self._cached_detections = private_state.detections
            self._cached_errors = list(private_state.attributes.get("errors", []))
            self._cached_ts = time.time()
            self.last_latency_ms = latency_ms
            self.runs += 1

    def __call__(self, frame_bgr, state: FrameState) -> None:
        self._frames_since_submit += 1
        with self._lock:
            cache_stale = self._cached_ts is None or (time.time() - self._cached_ts) >= self.ttl_seconds
        # Cadence caps the rate on a fast feed; staleness guarantees a fresh
        # result on a slow feed (a phone posting at 1 fps must not go blind).
        if self._worker_idle() and cache_stale:
            # Slow feed or first frame: nothing fresh to merge, so run inline
            # and pay the plugin's latency once (about 25 ms for the local
            # tools model) rather than hand this frame a blind result.
            self._frames_since_submit = 0
            self._run(frame_bgr, state.frame_index)
        elif self._worker_idle() and self._frames_since_submit >= self.every_n_frames:
            self._frames_since_submit = 0
            frame_copy = frame_bgr.copy()
            self._thread = threading.Thread(
                target=self._run,
                args=(frame_copy, state.frame_index),
                daemon=True,
            )
            self._thread.start()

        with self._lock:
            cached_ts = self._cached_ts
            detections = self._cached_detections
            errors = self._cached_errors

        if cached_ts is not None and (time.time() - cached_ts) < self.ttl_seconds:
            state.detections.extend(detections)
            if errors:
                state.attributes.setdefault("errors", []).extend(errors)


class LatestResult:
    """Holds the most recent FrameResult under a lock.

    This is the "detect slow, draw fast" trick: a renderer running at its
    own pace (e.g. a display loop redrawing every produced frame) can call
    get() at any time to reuse the newest judged boxes, without blocking or
    racing the consumer thread that produces them via set().
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._result = None
        self._ts = None

    def set(self, result, ts) -> None:
        with self._lock:
            self._result = result
            self._ts = ts

    def get(self):
        with self._lock:
            return self._result, self._ts


class StreamRunner:
    """Drains a Mailbox through a Spine and reports each result.

    run() loops forever until stop() is called; it is meant to be started
    in its own thread by the caller (a demo script or a server's
    background worker).
    """

    def __init__(self, spine, mailbox: Mailbox, on_result, latest_result: "LatestResult" = None):
        self.spine = spine
        self.mailbox = mailbox
        self.on_result = on_result
        self.latest_result = latest_result
        self._running = False
        self._last_ts = None

    def run(self) -> None:
        self._running = True
        while self._running:
            frame, ts = self.mailbox.get(timeout=0.5)
            if frame is None:
                continue
            dt = 0.0 if self._last_ts is None else max(0.0, ts - self._last_ts)
            self._last_ts = ts
            result = self.spine.process(frame, dt)
            if self.latest_result is not None:
                self.latest_result.set(result, ts)
            self.on_result(result, ts, result.latency_ms, self.mailbox.dropped)

    def stop(self) -> None:
        self._running = False


class Speaker:
    """Rate-limited text to speech via the macOS `say` command.

    Step 2 stand-in: on the glasses this becomes iPhone text-to-speech
    relayed over Bluetooth A2DP to the glasses' speakers. Here the Mac's
    own speaker plays the same role so the demo needs no phone or glasses.
    """

    def __init__(self, cooldown_seconds: float = 5.0, enabled: bool = True):
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self._last_spoken = {}  # sentence -> time.time() it was last spoken

    def say(self, sentence) -> bool:
        """Speak sentence, unless it repeats within cooldown_seconds.

        Returns True if this call actually announced the sentence (spoke it
        or, when muted, logged it), False if it was suppressed by the
        cooldown or sentence was None. Callers can use the return value to
        count real "spoken events" even when enabled=False.
        """
        if sentence is None:
            return False
        now = time.time()
        last = self._last_spoken.get(sentence)
        if last is not None and (now - last) < self.cooldown_seconds:
            return False
        self._last_spoken[sentence] = now

        if not self.enabled:
            print(f"[speaker] (muted) {sentence}")
            return True
        subprocess.Popen(
            ["say", sentence], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
