"""Unit tests for core/live.py: EventTracker, events_table_html, draw_hud.

Pure logic only. No models, no Gradio. FrameResult/RuleResult objects are
built by hand so these tests need no weights and no network.
"""

import numpy as np

from core.live import EventTracker, draw_hud, events_table_html
from core.rules import RuleResult
from core.spine import FrameResult


def _rule_result(rule, active=True, applicable=True, severity="high", detail="detail"):
    return RuleResult(rule=rule, active=active, severity=severity, detail=detail, applicable=applicable)


def _frame_result(rule_results, fired=None, sentence=None, timers=None, latency_ms=12.0):
    return FrameResult(
        detections=[],
        rule_results=rule_results,
        fired=fired or [],
        sentence=sentence,
        timers=timers or {},
        errors=[],
        latency_ms=latency_ms,
    )


class TestEventTracker:
    def test_open_event_reports_as_ongoing(self):
        tracker = EventTracker()
        tracker.update({"BARE_HANDS"}, t=1.0)

        rows = tracker.rows(now=3.0)

        assert len(rows) == 1
        row = rows[0]
        assert row["rule"] == "BARE_HANDS"
        assert row["start"] == 1.0
        assert row["end"] is None
        assert row["duration"] == 2.0

    def test_event_closes_once_rule_stops_firing(self):
        tracker = EventTracker()
        tracker.update({"BARE_HANDS"}, t=1.0)
        tracker.update(set(), t=4.0)

        rows = tracker.rows(now=10.0)

        assert len(rows) == 1
        row = rows[0]
        assert row["start"] == 1.0
        assert row["end"] == 4.0
        assert row["duration"] == 3.0

    def test_reopening_after_a_close_is_a_new_event(self):
        tracker = EventTracker()
        tracker.update({"BARE_HANDS"}, t=1.0)
        tracker.update(set(), t=2.0)
        tracker.update({"BARE_HANDS"}, t=5.0)

        rows = tracker.rows(now=6.0)

        assert len(rows) == 2
        closed, ongoing = rows[0], rows[1]
        assert closed["start"] == 1.0 and closed["end"] == 2.0
        assert ongoing["start"] == 5.0 and ongoing["end"] is None

    def test_close_all_closes_every_open_event(self):
        tracker = EventTracker()
        tracker.update({"BARE_HANDS", "TOOL_IN_BARE_HAND"}, t=1.0)

        tracker.close_all(t=9.0)
        rows = tracker.rows(now=1000.0)  # now must not matter once closed

        assert len(rows) == 2
        assert all(r["end"] == 9.0 for r in rows)
        assert tracker.open_events == {}

    def test_rows_are_sorted_by_start(self):
        tracker = EventTracker()
        tracker.update({"TOOL_IN_BARE_HAND"}, t=5.0)
        tracker.update({"TOOL_IN_BARE_HAND", "BARE_HANDS"}, t=6.0)

        rows = tracker.rows(now=10.0)

        assert [r["rule"] for r in rows] == ["TOOL_IN_BARE_HAND", "BARE_HANDS"]

    def test_multiple_rules_tracked_independently(self):
        tracker = EventTracker()
        tracker.update({"BARE_HANDS", "TOOL_IN_BARE_HAND"}, t=0.0)
        tracker.update({"BARE_HANDS"}, t=2.0)  # TOOL_IN_BARE_HAND clears

        rows = tracker.rows(now=5.0)

        by_rule = {r["rule"]: r for r in rows}
        assert by_rule["TOOL_IN_BARE_HAND"]["end"] == 2.0
        assert by_rule["BARE_HANDS"]["end"] is None
        assert by_rule["BARE_HANDS"]["duration"] == 5.0


class TestEventsTableHtml:
    def test_empty_rows_render_no_alerts_message(self):
        html = events_table_html([])

        assert "no alerts fired" in html
        assert 'data-testid="live-events-table"' in html

    def test_ongoing_row_shows_ongoing_marker(self):
        rows = [{"rule": "BARE_HANDS", "start": 1.0, "end": None, "duration": 2.0}]

        html = events_table_html(rows)

        assert "ongoing" in html
        assert 'data-rule="BARE_HANDS"' in html

    def test_closed_row_shows_end_and_duration(self):
        rows = [{"rule": "BARE_HANDS", "start": 1.0, "end": 3.5, "duration": 2.5}]

        html = events_table_html(rows)

        assert "3.5 s" in html
        assert "2.5 s" in html
        assert "ongoing" not in html

    def test_multiple_rows_each_get_a_data_rule_attribute(self):
        rows = [
            {"rule": "BARE_HANDS", "start": 1.0, "end": 2.0, "duration": 1.0},
            {"rule": "TOOL_IN_BARE_HAND", "start": 3.0, "end": None, "duration": 1.0},
        ]

        html = events_table_html(rows)

        assert 'data-rule="BARE_HANDS"' in html
        assert 'data-rule="TOOL_IN_BARE_HAND"' in html


class TestDrawHud:
    def test_draws_in_place_and_returns_the_frame_with_a_sentence(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        rule_results = [_rule_result("BARE_HANDS", active=True)]
        result = _frame_result(
            rule_results,
            fired=[{"rule": "BARE_HANDS"}],
            sentence="You aren't wearing gloves.",
            timers={"BARE_HANDS": 2.1},
        )

        out = draw_hud(frame, result, {"BARE_HANDS"}, "t=  1.0s")

        assert out is frame
        assert frame.any()  # some pixels were written

    def test_draws_without_a_sentence_and_without_crashing(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        rule_results = [_rule_result("BARE_HANDS", active=False, applicable=False)]
        result = _frame_result(rule_results, fired=[], sentence=None, timers={})

        out = draw_hud(frame, result, set(), "t=  0.0s")

        assert out is frame
        assert frame.any()

    def test_handles_multiple_rules_and_an_unapplicable_one(self):
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        rule_results = [
            _rule_result("BARE_HANDS", active=True),
            _rule_result("TOOL_IN_BARE_HAND", active=False, applicable=False),
        ]
        result = _frame_result(
            rule_results,
            fired=[{"rule": "BARE_HANDS"}],
            sentence="You aren't wearing gloves.",
            timers={"BARE_HANDS": 2.5},
        )

        out = draw_hud(frame, result, {"BARE_HANDS"}, "t=  3.0s")

        assert out is frame
        assert frame.any()
