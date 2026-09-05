"""NO_HELMET rule: pure function over hand-built detections."""

from core.detectors import Detection
from core.rules import RULES, debounce, no_helmet, spoken_sentence


def det(name, box, source="sh17"):
    return Detection(class_name=name, conf=0.9, box=box, source=source)


def test_no_head_is_unknown_not_pass():
    r = no_helmet([det("hands", (0, 0, 10, 10))])
    assert r.rule == "NO_HELMET"
    assert r.applicable is False
    assert r.active is False


def test_bare_head_is_active():
    r = no_helmet([det("head", (100, 100, 160, 170))])
    assert r.applicable is True
    assert r.active is True
    assert "without" in r.detail.lower()


def test_face_with_helmet_above_it_passes():
    # The helmet sits above the face box and barely touches it.
    face = det("face", (100, 140, 160, 200))
    helmet = det("helmet", (90, 90, 170, 145))
    r = no_helmet([face, helmet])
    assert r.applicable is True
    assert r.active is False


def test_head_with_overlapping_helmet_passes():
    head = det("head", (100, 100, 160, 170))
    helmet = det("helmet", (95, 90, 165, 130))
    assert no_helmet([head, helmet]).active is False


def test_helmet_only_counts_as_covered():
    r = no_helmet([det("helmet", (0, 0, 50, 30))])
    assert r.applicable is True
    assert r.active is False


def test_helmet_far_away_does_not_cover_a_head():
    head = det("head", (100, 100, 160, 170))
    helmet = det("helmet", (400, 400, 460, 440))
    assert no_helmet([head, helmet]).active is True


def test_one_bare_head_among_covered_heads_is_active():
    dets = [
        det("head", (100, 100, 160, 170)),
        det("helmet", (95, 90, 165, 130)),
        det("head", (300, 100, 360, 170)),
    ]
    assert no_helmet(dets).active is True


def test_rule_is_registered_and_spoken():
    assert no_helmet in RULES
    timers = {}
    fired = debounce(timers, [no_helmet([det("head", (0, 0, 10, 10))])], dt=1.0, threshold=0.0)
    assert [f["rule"] for f in fired] == ["NO_HELMET"]
    assert spoken_sentence(fired) == "Put your helmet on."


def test_tool_in_bare_hand_outranks_helmet():
    fired = [{"rule": "NO_HELMET"}, {"rule": "TOOL_IN_BARE_HAND"}]
    assert spoken_sentence(fired) == "Put gloves on before using that tool."
