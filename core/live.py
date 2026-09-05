"""Live-session pieces used by app.py's Live tab: tracking which rules are
firing, drawing the HUD, rendering the events table, and running a
persistent Spine across streamed webcam frames.

No Gradio imports here and no model construction, so this module is
testable with hand-built FrameResult/RuleResult objects and fake spines
and clocks.
"""

import time

import cv2

from core.spine import annotate


class EventTracker:
    """Tracks which rules are firing and for how long, across calls to
    update(). Same open/close bookkeeping process_video used to do inline.
    """

    def __init__(self):
        self.events = []
        self.open_events = {}

    def update(self, fired_rules: set, t: float) -> None:
        for rule in fired_rules:
            if rule not in self.open_events:
                self.open_events[rule] = {"rule": rule, "start": t}
        for rule in list(self.open_events):
            if rule not in fired_rules:
                ev = self.open_events.pop(rule)
                ev["end"] = t
                self.events.append(ev)

    def close_all(self, t: float) -> None:
        for rule in list(self.open_events):
            ev = self.open_events.pop(rule)
            ev["end"] = t
            self.events.append(ev)

    def rows(self, now: float) -> list:
        """Closed events plus open ones, each a dict with rule/start/end
        (None if ongoing) and duration, sorted by start.
        """
        out = []
        for ev in self.events:
            out.append(
                {
                    "rule": ev["rule"],
                    "start": ev["start"],
                    "end": ev["end"],
                    "duration": ev["end"] - ev["start"],
                }
            )
        for ev in self.open_events.values():
            out.append(
                {
                    "rule": ev["rule"],
                    "start": ev["start"],
                    "end": None,
                    "duration": now - ev["start"],
                }
            )
        out.sort(key=lambda r: r["start"])
        return out


def draw_hud(frame_bgr, result, fired_rules: set, t_label: str):
    """Draw the same HUD process_video draws: a t/judge-ms header line, one
    line per rule with its state and timer, and a red SAY line when the
    spine has a sentence. Draws in place and returns frame_bgr.
    """
    hud = [f"{t_label}  judge {result.latency_ms:.0f} ms"]
    for r in result.rule_results:
        timer = result.timers.get(r.rule, 0.0)
        if r.rule in fired_rules:
            state = "FIRED"
        elif r.active:
            state = "active"
        elif r.applicable:
            state = "pass"
        else:
            state = "unknown"
        hud.append(f"{r.rule}: {state} {timer:.1f}s")
    if result.sentence:
        hud.append(f'SAY: "{result.sentence}"')

    y = 22
    for i, line in enumerate(hud):
        color = (0, 0, 255) if (i == len(hud) - 1 and result.sentence) else (255, 255, 255)
        cv2.putText(frame_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22
    return frame_bgr


def events_table_html(rows: list) -> str:
    """Render the alert-events table process_video and the Live tab both
    show: Rule / Start / End / Duration, white background, dark text. An
    ongoing row (end is None) shows "ongoing" in the End column. Empty
    rows render one "no alerts fired" row.
    """
    trs = []
    for r in rows:
        end_str = "ongoing" if r["end"] is None else f"{r['end']:.1f} s"
        trs.append(
            f"<tr data-rule=\"{r['rule']}\">"
            f"<td style='padding:4px 8px;color:#111'>{r['rule']}</td>"
            f"<td style='padding:4px 8px;color:#111'>{r['start']:.1f} s</td>"
            f"<td style='padding:4px 8px;color:#111'>{end_str}</td>"
            f"<td style='padding:4px 8px;color:#111'>{r['duration']:.1f} s</td></tr>"
        )
    body = "".join(trs) or "<tr><td colspan='4' style='padding:4px 8px;color:#111'>no alerts fired</td></tr>"
    return (
        "<div style='background:#fff;color:#111;padding:8px'>"
        "<table data-testid=\"live-events-table\" style='border-collapse:collapse'>"
        "<tr><th style='text-align:left;padding:4px 8px;color:#111'>Rule</th><th style='color:#111'>Start</th>"
        "<th style='color:#111'>End</th><th style='color:#111'>Duration</th></tr>"
        f"{body}</table></div>"
    )


class LiveSession:
    """One Start..Stop run of the Live tab: a persistent Spine plus an
    EventTracker, with wall-clock dt so the spine's real debounce timers
    carry over correctly between streamed frames.
    """

    def __init__(self, spine, key, clock=time.perf_counter):
        self.spine = spine
        self.key = key
        self.clock = clock
        self.tracker = EventTracker()
        self.t0 = None
        self.last_ts = None
        self.frame_count = 0
        self.frame_times = []  # timestamps of judged frames, capped at 30

    def step(self, frame_bgr):
        now = self.clock()
        if self.t0 is None:
            self.t0 = now
            dt = 0.0
        else:
            dt = max(0.0, now - self.last_ts)
        self.last_ts = now
        t = now - self.t0

        result = self.spine.process(frame_bgr, dt)
        fired = {f["rule"] for f in result.fired}
        self.tracker.update(fired, t)

        annotated = annotate(frame_bgr, result.detections)
        draw_hud(annotated, result, fired, f"t={t:5.1f}s")

        self.frame_count += 1
        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        if len(self.frame_times) < 2:
            fps = 0.0
        else:
            span = self.frame_times[-1] - self.frame_times[0]
            # N timestamps span N-1 intervals.
            fps = (len(self.frame_times) - 1) / span if span > 0 else 0.0

        stats = (
            f"frame {self.frame_count} | judge {result.latency_ms:.0f} ms | {fps:.1f} fps effective "
            f"| debounce {self.spine.debounce_seconds:.1f} s"
        )
        for plugin, _ in self.spine.plugins:
            if hasattr(plugin, "runs"):
                stats += f" | tools runs {plugin.runs}"
                break

        rows = self.tracker.rows(t)
        return annotated, rows, stats

    def close(self):
        """Close any open events at the current t and return the final rows."""
        t = 0.0 if self.t0 is None else (self.clock() - self.t0)
        self.tracker.close_all(t)
        return self.tracker.rows(t)
