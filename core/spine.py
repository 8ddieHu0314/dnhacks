"""Frame processing pipeline: run detectors + plugins, apply rules, debounce.

No person tracking in this step. The wearer checks themself (first-person
camera), so every Hands/Gloves/Tools detection in a frame is assumed to
belong to the wearer.
"""

import time
from dataclasses import dataclass

import cv2

from core.detectors import RoboflowDetector, SH17Detector
from core.rules import RULES, correct_detections, debounce, spoken_sentence


@dataclass
class FrameState:
    detections: list
    attributes: dict
    timers: dict  # persists across frames (same dict object as Spine.timers)
    frame_index: int


@dataclass
class FrameResult:
    detections: list
    rule_results: list
    fired: list
    sentence: object  # str | None
    timers: dict
    errors: list
    latency_ms: float = 0.0


def roboflow_tools_plugin(detector: RoboflowDetector = None, **detector_kwargs):
    """Factory for a plugin that wraps a RoboflowDetector and appends its
    detections (and any error) to the frame state. Returns a callable of
    the plugin signature: (frame_bgr, state) -> None.
    """
    det = detector or RoboflowDetector(**detector_kwargs)

    def _plugin(frame_bgr, state: FrameState) -> None:
        dets = det.detect(frame_bgr)
        state.detections.extend(dets)
        if det.last_error:
            state.attributes.setdefault("errors", []).append(f"roboflow: {det.last_error}")

    _plugin.detector = det  # expose for introspection/testing
    return _plugin


# --- Future plugin slots (not implemented in this step) ---------------------
#
# Pose plugin stub: a plugin that runs a pose model and adds keypoint
# attributes to state, so rules could reason about arm/wrist position.
#
#   def pose_plugin(frame_bgr, state):
#       state.attributes["pose"] = pose_model.predict(frame_bgr)
#
#   spine.register_plugin(pose_plugin, every_n_frames=2)
#
# Third-person tracking stub: this step has no ByteTrack / per-person
# tracking. All detections in a frame are treated as belonging to the
# single wearer. When third-person rules are added (checking OTHER people
# in frame, not just the wearer), a tracker would assign a track_id to each
# Person box, and rules would need to group Hands/Gloves/Tools detections
# by nearest track_id before applying bare_hands / tool_in_bare_hand per
# person. That grouping step slots in right before the RULES loop below.
# ------------------------------------------------------------------------


class Spine:
    def __init__(self, sh17: SH17Detector, plugins: list = None, debounce_seconds: float = 2.0):
        self.sh17 = sh17
        self.plugins = list(plugins) if plugins else []  # list of (fn, every_n_frames)
        self.debounce_seconds = debounce_seconds
        self.timers = {}
        self.frame_index = 0

    def register_plugin(self, fn, every_n_frames: int = 1) -> None:
        self.plugins.append((fn, every_n_frames))

    def reset(self) -> None:
        self.timers = {}
        self.frame_index = 0

    def process(self, frame_bgr, dt: float) -> FrameResult:
        start = time.perf_counter()
        state = FrameState(
            detections=[],
            attributes={},
            timers=self.timers,
            frame_index=self.frame_index,
        )

        state.detections.extend(self.sh17.detect(frame_bgr))

        for fn, every_n_frames in self.plugins:
            if every_n_frames <= 1 or self.frame_index % every_n_frames == 0:
                fn(frame_bgr, state)

        # Helmet beats gloves on the same region and vest beats gloves on the
        # same box (the detector's confusion runs one way only: yellow hard
        # hat or hi-vis vest called gloves), then weak witness
        # witnesses the detector let through are dropped. See rules.py.
        state.detections = correct_detections(state.detections, self.sh17.conf)

        rule_results = [rule(state.detections) for rule in RULES]
        fired = debounce(self.timers, rule_results, dt, threshold=self.debounce_seconds)
        sentence = spoken_sentence(fired)

        self.frame_index += 1
        latency_ms = (time.perf_counter() - start) * 1000.0

        return FrameResult(
            detections=state.detections,
            rule_results=rule_results,
            fired=fired,
            sentence=sentence,
            timers=dict(self.timers),
            errors=list(state.attributes.get("errors", [])),
            latency_ms=latency_ms,
        )


_COLOR_SH17 = (0, 255, 0)       # green (BGR)
_COLOR_ROBOFLOW = (0, 165, 255)  # orange (BGR)
_COLOR_GLOVES = (255, 0, 0)     # blue (BGR)
_COLOR_HANDS = (0, 255, 255)    # yellow (BGR)
_COLOR_HELMET = (0, 140, 255)   # orange (BGR)
_COLOR_VEST = (255, 0, 255)     # magenta (BGR)


def annotate(frame_bgr, detections):
    """Return a copy of frame_bgr with boxes and 'class conf' labels drawn."""
    out = frame_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)

        class_lower = d.class_name.lower()
        if class_lower == "gloves":
            color = _COLOR_GLOVES
        elif class_lower == "hands":
            color = _COLOR_HANDS
        elif class_lower == "helmet":
            color = _COLOR_HELMET
        elif class_lower == "safety-vest":
            color = _COLOR_VEST
        elif d.source == "roboflow":
            color = _COLOR_ROBOFLOW
        else:
            color = _COLOR_SH17

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{d.class_name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, label_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out
