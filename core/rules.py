"""PPE compliance rules.

Pure functions only: no model imports, no numpy. Operates on the plain
Detection objects produced by core/detectors.py (duck-typed here so this
module stays independent and fast to import/test).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.detectors import Detection


def iou(a: tuple, b: tuple) -> float:
    """Intersection over union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _is_tool(det) -> bool:
    # Mirrors core.detectors.is_tool without importing that (heavier) module.
    # Class name casing/pluralization varies by weights file (e.g. "Tools"
    # vs "tool"), so compare case-insensitively.
    return det.source == "roboflow" or det.class_name.lower() in ("tool", "tools")


def _is_class(det, name: str) -> bool:
    return det.class_name.lower() == name.lower()


def _center(box: tuple) -> tuple:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _expand_box(box: tuple, factor: float) -> tuple:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx, cy = _center(box)
    nw = w * (1.0 + factor)
    nh = h * (1.0 + factor)
    return (cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0)


def _point_in_box(pt: tuple, box: tuple) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _intersection_over_smaller(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return (iw * ih) / smaller if smaller > 0 else 0.0


def _tool_overlaps_hand(tool_box: tuple, hand_box: tuple) -> bool:
    # A held tool sticks out of the fist, so plain IoU is tiny even when the
    # grip is obvious. Demo heuristic: enough of the smaller box overlaps, or
    # the tool's center sits inside a generously expanded hand box.
    if _intersection_over_smaller(tool_box, hand_box) > 0.05:
        return True
    expanded = _expand_box(hand_box, 0.5)
    return _point_in_box(_center(tool_box), expanded)


@dataclass
class RuleResult:
    rule: str
    active: bool
    severity: str
    detail: str
    applicable: bool  # False = could not be judged; report as "unknown", not "pass"


# A helmet box only has to be weakly present to veto a gloves box on the same
# region, because the confusion runs one way. The detector keeps helmet boxes
# down to this confidence as witnesses; correct_detections drops the weak ones
# again after the veto so they never reach the rules or the overlay.
HELMET_WITNESS_CONF = 0.2


def correct_detections(detections: list, conf: float) -> list:
    """Detection-level corrections the Spine applies before the rules:
    helmet beats gloves on the same region (using any helmet box, including
    weak witnesses below `conf`), then everything below `conf` is dropped.
    Pure."""
    kept = suppress_gloves_on_helmet(detections)
    return [d for d in kept if d.conf >= conf]


def suppress_gloves_on_helmet(detections: list, overlap: float = 0.6) -> list:
    """Drop a `gloves` box that mostly sits inside a `helmet` box.

    The detector confuses the two in one direction only: a yellow hard hat
    gets called gloves far more often than a glove gets called helmet. So
    when both land on the same region, the helmet reading wins. Measured as
    intersection over the smaller box so a partial helmet box still counts.
    Pure: filters the list, touches nothing else.
    """
    helmets = [d.box for d in detections if _is_class(d, "helmet")]
    if not helmets:
        return detections
    return [
        d for d in detections
        if not (_is_class(d, "gloves") and any(_intersection_over_smaller(d.box, h) > overlap for h in helmets))
    ]


def bare_hands(detections: list) -> RuleResult:
    hands = [d for d in detections if _is_class(d, "hands")]
    gloves = [d for d in detections if _is_class(d, "gloves")]

    # SH17 labels a gloved hand as "gloves" and a bare hand as "hands", so a
    # frame with only gloves boxes is a gloved wearer, not an unknown.
    if not hands and not gloves:
        return RuleResult(
            rule="BARE_HANDS",
            active=False,
            severity="high",
            detail="No hands visible",
            applicable=False,
        )

    active = any(not any(iou(h.box, g.box) > 0.3 for g in gloves) for h in hands)
    detail = "Bare hand detected" if active else "Gloves detected on all visible hands"
    return RuleResult(
        rule="BARE_HANDS",
        active=active,
        severity="high",
        detail=detail,
        applicable=True,
    )


def tool_in_bare_hand(detections: list) -> RuleResult:
    tools = [d for d in detections if _is_tool(d)]
    hands = [d for d in detections if _is_class(d, "hands")]
    gloves = [d for d in detections if _is_class(d, "gloves")]

    if not tools or not (hands or gloves):
        return RuleResult(
            rule="TOOL_IN_BARE_HAND",
            active=False,
            severity="critical",
            detail="No tool + hand combination visible",
            applicable=False,
        )

    active = False
    detail = "Tool present, no bare hand on it"
    for h in hands:
        gloved = any(iou(h.box, g.box) > 0.3 for g in gloves)
        if gloved:
            continue
        for t in tools:
            if _tool_overlaps_hand(t.box, h.box):
                active = True
                detail = f"Tool '{t.class_name}' held by a bare hand"
                break
        if active:
            break

    return RuleResult(
        rule="TOOL_IN_BARE_HAND",
        active=active,
        severity="critical",
        detail=detail,
        applicable=True,
    )


RULES = [bare_hands, tool_in_bare_hand]


def debounce(timers: dict, results: list, dt: float, threshold: float = 2.0) -> list:
    """Accumulate active time per rule and report rules that have been
    active continuously for at least `threshold` seconds. threshold=0
    fires as soon as a rule is active (single-image mode).
    """
    fired = []
    for r in results:
        if r.active:
            timers[r.rule] = timers.get(r.rule, 0.0) + dt
        else:
            timers[r.rule] = 0.0

        if r.active and timers[r.rule] >= threshold:
            fired.append(
                {
                    "rule": r.rule,
                    "severity": r.severity,
                    "seconds_active": timers[r.rule],
                    "detail": r.detail,
                }
            )
    return fired


def spoken_sentence(fired: list):
    rules_fired = {f["rule"] for f in fired}
    if "TOOL_IN_BARE_HAND" in rules_fired:
        return "Put gloves on before using that tool."
    if "BARE_HANDS" in rules_fired:
        return "You aren't wearing gloves."
    return None
