"""Live-feed plumbing for Step 1: a latest-frame-wins mailbox, a background
runner that drains it through a Spine, a cache wrapper that keeps a slow
plugin (hosted Roboflow calls: 250 ms to 2 s) off the hot path, and a
rate-limited speaker.

Nothing here is model-specific. This module only manages threads, time, and
queues so a 10 fps live feed never backs up behind a slow plugin call.
"""

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque

import cv2

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


class _ActiveClip:
    """One in-progress clip.

    Encoding runs on its own thread so a violation's disk I/O never blocks
    the ingest or judge path, the same idea as CachedPlugin keeping a slow
    model call off the hot path. feed() just hands a frame to a queue;
    the writer thread drains it.
    """

    def __init__(self, clips_dir: str, rule: str, start_ts: float, lead_in_frames: list, fps_hint: float):
        self.rule = rule
        self.path = os.path.join(clips_dir, f"{rule}_{int(start_ts)}.mp4")
        self.keyframe = lead_in_frames[-1][0] if lead_in_frames else None
        self._last_active_ts = start_ts
        self._queue = queue.Queue()

        if lead_in_frames:
            h, w = lead_in_frames[-1][0].shape[:2]
        else:
            h, w = 480, 640
        self._thread = threading.Thread(
            target=self._run, args=(lead_in_frames, w, h, fps_hint), daemon=True
        )
        self._thread.start()

    def _run(self, lead_in_frames: list, w: int, h: int, fps_hint: float) -> None:
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"avc1"), fps_hint, (w, h))
        if not writer.isOpened():
            writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps_hint, (w, h))
        for frame, _ts in lead_in_frames:
            writer.write(frame)
        while True:
            frame = self._queue.get()
            if frame is None:  # sentinel from close()
                break
            writer.write(frame)
        writer.release()

    def feed(self, frame_bgr) -> None:
        self._queue.put(frame_bgr)

    def mark_active(self, ts: float) -> None:
        self._last_active_ts = ts

    def idle_for(self, ts: float) -> float:
        return ts - self._last_active_ts

    def close(self) -> None:
        self._queue.put(None)

    def join(self, timeout: float = None) -> None:
        self._thread.join(timeout=timeout)


class Recorder:
    """Never-drop side of the ingest tee (see docs/ARCHITECTURE.md, 'Recording
    and the session record').

    Keeps the last `buffer_seconds` of raw frames in memory in a time-bounded
    deque. Nothing touches disk while every rule is inactive. When a rule
    fires, dumps that lead-up plus a live tail (frames keep being added until
    the rule clears plus `tail_seconds`) to
    sessions/<session_id>/clips/<rule>_<t>.mp4, a keyframe JPEG, and one line
    in events.jsonl.

    Two entry points, called from different threads:
      add_frame(frame, ts)   -- every incoming frame, before the mailbox
                                 might drop it (call at ingest, e.g. server.py
                                 POST /frame or a demo's producer loop).
      handle_result(result, ts) -- every judged FrameResult (call from the
                                 spine's on_result callback).
    """

    def __init__(
        self,
        session_id: str = None,
        base_dir: str = "sessions",
        buffer_seconds: float = 20.0,
        tail_seconds: float = 3.0,
        fps_hint: float = 15.0,
        record_all: bool = False,
    ):
        self.buffer_seconds = buffer_seconds
        self.tail_seconds = tail_seconds
        self.fps_hint = fps_hint
        self.session_id = session_id or time.strftime("%Y%m%dT%H%M%S")
        self.session_dir = os.path.join(base_dir, self.session_id)
        self.clips_dir = os.path.join(self.session_dir, "clips")
        self.keyframes_dir = os.path.join(self.session_dir, "keyframes")
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.keyframes_dir, exist_ok=True)
        self.events_path = os.path.join(self.session_dir, "events.jsonl")

        # record_all: also write every raw ingested frame as a JPEG to
        # sessions/<id>/frames/<ts_ms>.jpg, for supervised training runs
        # (docs/ARCHITECTURE.md, "--record-all"). Off by default, fills disk.
        self.record_all = record_all
        self.frames_dir = os.path.join(self.session_dir, "frames")
        self.frames_written = 0
        if record_all:
            os.makedirs(self.frames_dir, exist_ok=True)

        self._buf_lock = threading.Lock()
        self._buffer = deque()  # (frame_bgr, ts), oldest first, never older than buffer_seconds

        self._clip_lock = threading.Lock()
        self._active_clips = {}  # rule -> _ActiveClip
        self._all_clips = []  # every _ActiveClip this session has ever created, closed or not

        self._events_lock = threading.Lock()
        self._events = []  # ordered queue of every fired event this session, oldest first

    def add_frame(self, frame_bgr, ts: float) -> None:
        frame_copy = frame_bgr.copy()
        if self.record_all:
            # A 640px JPEG encode is about 2 ms, cheap enough to stay inline at
            # phone frame rates and keep frame order trivially correct.
            cv2.imwrite(os.path.join(self.frames_dir, f"{int(ts * 1000)}.jpg"), frame_copy)
            self.frames_written += 1
        with self._buf_lock:
            self._buffer.append((frame_copy, ts))
            cutoff = ts - self.buffer_seconds
            while len(self._buffer) > 1 and self._buffer[0][1] < cutoff:
                self._buffer.popleft()
        with self._clip_lock:
            clips = list(self._active_clips.values())
        for clip in clips:
            clip.feed(frame_copy)

    def handle_result(self, result, ts: float) -> None:
        fired_rules = {f["rule"] for f in result.fired}
        active_rules = {r.rule for r in result.rule_results if r.active}

        new_clips = []
        with self._clip_lock:
            for rule in fired_rules:
                if rule not in self._active_clips:
                    with self._buf_lock:
                        lead_in = list(self._buffer)
                    clip = _ActiveClip(self.clips_dir, rule, ts, lead_in, self.fps_hint)
                    self._active_clips[rule] = clip
                    self._all_clips.append(clip)
                    new_clips.append(clip)
            for rule in active_rules:
                clip = self._active_clips.get(rule)
                if clip is not None:
                    clip.mark_active(ts)
            stale = [
                rule
                for rule, clip in self._active_clips.items()
                if rule not in active_rules and clip.idle_for(ts) >= self.tail_seconds
            ]
            closing = [self._active_clips.pop(rule) for rule in stale]

        for clip in new_clips:
            fired_entry = next((f for f in result.fired if f["rule"] == clip.rule), None)
            self._write_event(ts, result, fired_entry, clip)
        for clip in closing:
            clip.close()

    def _write_event(self, ts: float, result, fired_entry: dict, clip: "_ActiveClip") -> None:
        keyframe_path = None
        if clip.keyframe is not None:
            keyframe_path = os.path.join(self.keyframes_dir, f"{clip.rule}_{int(ts * 1000)}.jpg")
            cv2.imwrite(keyframe_path, clip.keyframe)

        event = {
            "rule": clip.rule,
            "severity": fired_entry.get("severity") if fired_entry else None,
            "detail": fired_entry.get("detail") if fired_entry else None,
            "seconds_active": fired_entry.get("seconds_active") if fired_entry else None,
            "ts": ts,
            "clip_path": clip.path,
            "keyframe_path": keyframe_path,
            "detections": [
                {"class": d.class_name, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box], "source": d.source}
                for d in result.detections
            ],
        }
        with self._events_lock:
            self._events.append(event)
            with open(self.events_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def list_events(self) -> list:
        """The session's fired-event queue so far, oldest first. Each entry
        has a clip_path an end-of-session report can open and replay. A
        clip is only a complete, playable file once it's closed (the rule
        cleared plus the tail, or shutdown() ran) -- cv2.VideoWriter doesn't
        finalize the file until then.
        """
        with self._events_lock:
            return list(self._events)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Force-close any still-open clips, then block until EVERY clip
        this session ever wrote -- including ones that already closed
        naturally earlier (rule cleared plus the tail) -- has fully
        flushed to disk. A clip left open when the stream stops (the
        wearer was still mid-violation) would otherwise sit in a daemon
        thread's queue and be lost when the process exits; a clip that
        already closed still has no guarantee its writer thread finished
        draining its queue unless something joins it, which is exactly
        what this closes the gap on. Call this once, when the stream/
        session ends, before treating any clip_path as ready to read.
        """
        with self._clip_lock:
            still_open = list(self._active_clips.values())
            self._active_clips.clear()
            all_clips = list(self._all_clips)
        for clip in still_open:
            clip.close()
        for clip in all_clips:
            clip.join(timeout=timeout)


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
