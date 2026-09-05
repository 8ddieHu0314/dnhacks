"""Integration tests for core/live.py's LiveSession, plus the Spine/
CachedPlugin wiring it depends on and app.py's build_spine.

Two kinds of test here:
1. LiveSession driven by a small scripted fake spine and an injected fake
   clock, so dt/timing math can be asserted exactly with no real model.
2. A real core.spine.Spine wired to a stub SH17 detector (hand-built
   Detection objects, no ultralytics) and a real CachedPlugin wrapping a
   counting fake plugin, to prove the cadence and debounce actually work
   together the way server.py relies on. No weights are loaded anywhere.
"""

import time

import numpy as np

from core.detectors import Detection
from core.live import LiveSession
from core.rules import RuleResult
from core.spine import FrameResult, Spine
from core.stream import CachedPlugin


class FakeClock:
    """A controllable clock: starts at `start`, advances only when told to."""

    def __init__(self, start=0.0):
        self.t = start

    def tick(self, dt):
        self.t += dt

    def __call__(self):
        return self.t


class ScriptedFakeSpine:
    """Matches the duck type LiveSession needs: process(frame, dt) ->
    FrameResult, plus .plugins and .debounce_seconds. Each call pops the
    next fired-rule set off a script so a test can control exactly when a
    rule opens and closes.
    """

    def __init__(self, fired_sequence, debounce_seconds=2.0, plugins=None, latency_ms=5.0):
        self.fired_sequence = list(fired_sequence)
        self.debounce_seconds = debounce_seconds
        self.plugins = plugins or []
        self.latency_ms = latency_ms
        self.calls = []  # dt values passed to process(), for assertions

    def process(self, frame_bgr, dt):
        self.calls.append(dt)
        fired_rules = self.fired_sequence.pop(0) if self.fired_sequence else set()
        fired = [{"rule": r} for r in fired_rules]
        rule_results = [RuleResult(rule=r, active=True, severity="high", detail="d", applicable=True) for r in fired_rules]
        if not rule_results:
            rule_results = [RuleResult(rule="BARE_HANDS", active=False, severity="high", detail="d", applicable=True)]
        return FrameResult(
            detections=[],
            rule_results=rule_results,
            fired=fired,
            sentence=None,
            timers={},
            errors=[],
            latency_ms=self.latency_ms,
        )


class FakePluginWithRuns:
    def __init__(self, runs):
        self.runs = runs


def _frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestLiveSessionDt:
    def test_dt_is_zero_on_first_frame_then_clock_delta(self):
        clock = FakeClock(100.0)
        spine = ScriptedFakeSpine(fired_sequence=[set(), set(), set()])
        session = LiveSession(spine, key="k", clock=clock)

        session.step(_frame())
        assert spine.calls[0] == 0.0

        clock.tick(0.5)
        session.step(_frame())
        assert spine.calls[1] == 0.5

        clock.tick(0.25)
        session.step(_frame())
        assert spine.calls[2] == 0.25


class TestLiveSessionEvents:
    def test_event_opens_then_closes_across_steps(self):
        clock = FakeClock(0.0)
        spine = ScriptedFakeSpine(fired_sequence=[{"BARE_HANDS"}, {"BARE_HANDS"}, set()])
        session = LiveSession(spine, key="k", clock=clock)
        frame = _frame()

        _, rows, _ = session.step(frame)  # t=0, fired -> opens
        assert len(rows) == 1 and rows[0]["end"] is None and rows[0]["start"] == 0.0

        clock.tick(1.0)
        _, rows, _ = session.step(frame)  # t=1, still fired -> still ongoing
        assert len(rows) == 1 and rows[0]["end"] is None

        clock.tick(1.0)
        _, rows, _ = session.step(frame)  # t=2, no longer fired -> closed
        assert len(rows) == 1
        assert rows[0]["start"] == 0.0
        assert rows[0]["end"] == 2.0

    def test_close_closes_open_events_at_current_t(self):
        clock = FakeClock(0.0)
        spine = ScriptedFakeSpine(fired_sequence=[{"BARE_HANDS"}])
        session = LiveSession(spine, key="k", clock=clock)
        session.step(_frame())

        clock.tick(3.0)
        rows = session.close()

        assert len(rows) == 1
        assert rows[0]["end"] == 3.0


class TestLiveSessionStats:
    def test_stats_line_contains_judge_ms_and_fps(self):
        clock = FakeClock(0.0)
        spine = ScriptedFakeSpine(fired_sequence=[set()], latency_ms=7.0)
        session = LiveSession(spine, key="k", clock=clock)

        _, _, stats = session.step(_frame())

        assert "judge 7 ms" in stats
        assert "fps effective" in stats
        assert "debounce 2.0 s" in stats

    def test_fps_is_zero_with_a_single_frame(self):
        clock = FakeClock(0.0)
        spine = ScriptedFakeSpine(fired_sequence=[set()])
        session = LiveSession(spine, key="k", clock=clock)

        _, _, stats = session.step(_frame())

        assert "0.0 fps effective" in stats

    def test_stats_line_includes_tools_runs_when_a_plugin_exposes_runs(self):
        clock = FakeClock(0.0)
        plugin = FakePluginWithRuns(runs=4)
        spine = ScriptedFakeSpine(fired_sequence=[set()], plugins=[(plugin, 5)])
        session = LiveSession(spine, key="k", clock=clock)

        _, _, stats = session.step(_frame())

        assert "tools runs 4" in stats

    def test_stats_line_omits_tools_runs_when_no_plugin_has_runs(self):
        clock = FakeClock(0.0)
        spine = ScriptedFakeSpine(fired_sequence=[set()], plugins=[])
        session = LiveSession(spine, key="k", clock=clock)

        _, _, stats = session.step(_frame())

        assert "tools runs" not in stats


class TestRealSpineCachedPluginWiring:
    def test_cadence_limits_plugin_runs_and_debounce_gates_firing(self):
        """12 frames at dt=0.5 with debounce=2.0: BARE_HANDS must not fire on
        frame 0 (timer only 0.5s) but must have fired by frame 4 (timer
        reaches 2.0s). The CachedPlugin, wrapping a trivial counting
        plugin, must run far fewer than 12 times (cadence every_n_frames=5,
        plus one inline run on the very first call before any cache exists).
        """
        bare_hand_detection = Detection(class_name="hands", conf=0.9, box=(0, 0, 10, 10), source="sh17")

        class StubSH17:
            def detect(self, frame_bgr):
                return [bare_hand_detection]

        class CountingPlugin:
            def __call__(self, frame_bgr, state):
                pass  # no detections to add, just something to count runs of

        cached = CachedPlugin(CountingPlugin(), every_n_frames=5, ttl_seconds=1.0)
        spine = Spine(sh17=StubSH17(), plugins=[(cached, 1)], debounce_seconds=2.0)

        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        fired_at = []
        for i in range(12):
            result = spine.process(frame, dt=0.5)
            if any(f["rule"] == "BARE_HANDS" for f in result.fired):
                fired_at.append(i)

        time.sleep(0.05)  # let any in-flight background plugin thread finish

        assert 0 not in fired_at, "BARE_HANDS should not fire on the first frame"
        assert fired_at, "BARE_HANDS should have fired at least once over 12 frames"
        assert fired_at[0] <= 4, f"expected BARE_HANDS to fire by frame index 4, fired at {fired_at}"
        assert 1 <= cached.runs <= 4, f"expected cadence-limited plugin runs, got {cached.runs}"


class TestBuildSpine:
    def test_cached_tools_true_wraps_plugin_in_cached_plugin(self, monkeypatch):
        import app as app_module

        class FakeSH17:
            def __init__(self, weights_path, device="mps", conf=0.4):
                pass

            def detect(self, frame_bgr):
                return []

        def fake_roboflow_tools_plugin(conf=0.4):
            def _plugin(frame_bgr, state):
                pass

            return _plugin

        monkeypatch.setattr(app_module, "SH17Detector", FakeSH17)
        monkeypatch.setattr(app_module, "roboflow_tools_plugin", fake_roboflow_tools_plugin)

        spine = app_module.build_spine("yolo8s", 0.4, True, 2.0, cached_tools=True)

        assert len(spine.plugins) == 1
        plugin, every_n = spine.plugins[0]
        assert isinstance(plugin, CachedPlugin)
        assert plugin.every_n_frames == 5
        assert plugin.ttl_seconds == 1.0

    def test_cached_tools_false_gives_a_plain_plugin(self, monkeypatch):
        import app as app_module

        class FakeSH17:
            def __init__(self, weights_path, device="mps", conf=0.4):
                pass

            def detect(self, frame_bgr):
                return []

        def fake_roboflow_tools_plugin(conf=0.4):
            def _plugin(frame_bgr, state):
                pass

            return _plugin

        monkeypatch.setattr(app_module, "SH17Detector", FakeSH17)
        monkeypatch.setattr(app_module, "roboflow_tools_plugin", fake_roboflow_tools_plugin)

        spine = app_module.build_spine("yolo8s", 0.4, True, 2.0, cached_tools=False)

        assert len(spine.plugins) == 1
        plugin, every_n = spine.plugins[0]
        assert not isinstance(plugin, CachedPlugin)

    def test_debounce_seconds_is_passed_through(self, monkeypatch):
        import app as app_module

        class FakeSH17:
            def __init__(self, weights_path, device="mps", conf=0.4):
                pass

            def detect(self, frame_bgr):
                return []

        monkeypatch.setattr(app_module, "SH17Detector", FakeSH17)

        spine = app_module.build_spine("yolo8s", 0.4, False, 3.5, cached_tools=False)

        assert spine.plugins == []
        assert spine.debounce_seconds == 3.5
