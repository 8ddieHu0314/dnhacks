"""The part catalog as Claude's ground truth.

Loads docs/components/components.json (a teammate's researched records, one per kit part) and
renders the whole thing as one compact text block that goes into the cached system prompt of
every voice question. The wearer can then ask about any part on the table, any pin, any rating,
and the answer comes from the record rather than from the model's memory. With prompt caching
the block costs almost nothing after the first call; the trade is a slightly longer time to the
first token, which a hackathon can afford.

Field names are the record schema in docs/components/spec_extraction_prompt.md; the renderer is
tolerant of missing sections because not every record has every field.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATHS = [REPO_ROOT / "docs" / "components" / "components.json",
                 Path(__file__).with_name("components.json")]

records: dict[str, dict] = {}
prompt_text: str = ""
index_text: str = ""     # one line per part: id, names, what it looks like
source: str = "none"


def _flat(v) -> str:
    """Nested dicts/lists to one readable line."""
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_flat(x)}" for k, x in v.items() if x not in (None, "", [], {}))
    if isinstance(v, (list, tuple)):
        return "; ".join(_flat(x) for x in v if x not in (None, "", [], {}))
    return str(v)


def _clip(v, n: int) -> str:
    t = re.sub(r"\s+", " ", _flat(v)).strip() if v not in (None, "", [], {}) else ""
    if len(t) <= n:
        return t
    cut = t[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:(") + " ..."


def render_record(rec: dict) -> str:
    """One part, about 2 KB: identity, what it does, what it looks like, pins, electrical,
    specs, wiring, safety, troubleshooting. Sections are clipped so the whole catalog stays
    around 30k tokens."""
    d = rec.get("details") or {}
    ident = d.get("identity") or {}
    pid = rec.get("id")
    name = rec.get("canonical_name") or ident.get("canonical_name") or rec.get("name") or pid
    head = f"### {pid} | {name}"
    on_kit = rec.get("name_on_kit") or ident.get("name_on_kit")
    if on_kit:
        head += f' | on the kit lid: "{on_kit}"'
    mpn = rec.get("mpn") or ident.get("manufacturer_part_number")
    if mpn:
        head += f" | part number: {_clip(mpn, 60)}"
    cat = rec.get("category") or ident.get("category")
    if cat:
        head += f" | category: {cat}"
    lines = [head]

    def add(label, v, n):
        t = _clip(v, n)
        if t:
            lines.append(f"{label}: {t}")

    aliases = ident.get("aliases")
    if aliases:
        add("Also called", aliases, 120)
    fn = d.get("function") or {}
    add("Function", " ".join(x for x in (fn.get("one_sentence_plain_english"), fn.get("what_it_is_used_for")) if x)
        or rec.get("function"), 220)
    add("How it works", fn.get("how_it_works_briefly"), 150)
    vis = d.get("visual_identification") or {}
    add("Looks like", {"shape": vis.get("shape_and_size"), "markings": vis.get("color_and_markings"),
                       "printed text": vis.get("printed_text_to_look_for"), "confused with": vis.get("easily_confused_with")}, 280)
    pins = d.get("pins") or {}
    pin_count = rec.get("pin_count") or pins.get("pin_count")
    add("Pins", {"count": pin_count, "pin 1": pins.get("pin_1_identification"),
                 "polarity sensitive": pins.get("polarity_sensitive"), "pinout": pins.get("pinout")}, 640)
    add("Electrical", d.get("electrical") or rec.get("voltage"), 260)
    add("Key specs", d.get("key_specs") or rec.get("key_specs"), 280)
    add("Interface", rec.get("interface"), 80)
    add("Wiring to the Uno", d.get("wiring_to_uno"), 320)
    safety = d.get("safety") or {}
    add("Safety", {"hazards": safety.get("hazards"), "handling": safety.get("handling_notes"),
                   "common mistakes": safety.get("common_mistakes")}, 260)
    add("Troubleshooting", d.get("troubleshooting"), 220)
    price = rec.get("price_usd") or (d.get("market") or {}).get("unit_price_usd")
    if price not in (None, ""):
        lines.append(f"Price: about {price} USD each")
    conf = (d.get("confidence") or {})
    if conf.get("fields_uncertain"):
        add("Uncertain fields", conf.get("fields_uncertain"), 160)
    return "\n".join(lines)


def index_line(rec: dict) -> str:
    d = rec.get("details") or {}
    ident = d.get("identity") or {}
    vis = d.get("visual_identification") or {}
    name = rec.get("canonical_name") or ident.get("canonical_name") or rec.get("name") or rec.get("id")
    on_kit = rec.get("name_on_kit") or ident.get("name_on_kit") or ""
    look = " ".join(x for x in (vis.get("shape_and_size"), vis.get("color_and_markings")) if x)
    return f"- {rec.get('id')}: {_clip(name, 50)}" + (f' (kit lid: "{_clip(on_kit, 40)}")' if on_kit else "") + f" | looks: {_clip(look, 150) or 'n/a'}"


def load() -> str:
    """Read the catalog and render the prompt block. Returns a one-line status for the log."""
    global records, prompt_text, index_text, source
    for p in CATALOG_PATHS:
        if p.exists():
            raw = json.loads(p.read_text())
            recs = raw.get("components") or raw.get("parts") or raw if isinstance(raw, dict) else raw
            if isinstance(recs, dict):
                recs = [dict(v, id=v.get("id", k)) for k, v in recs.items()]
            records = {str(r.get("id") or r.get("slug") or r.get("name")): r for r in recs}
            prompt_text = "\n\n".join(render_record(r) for r in records.values())
            index_text = "\n".join(index_line(r) for r in records.values())
            source = str(p)
            return f"{len(records)} parts from {p} ({len(prompt_text) // 1024} KB of prompt)"
    records, prompt_text, index_text, source = {}, "", "", "none"
    return "no catalog file found; answers come from the image alone"


def record(pid: str | None) -> dict | None:
    """Catalog record for an id, tolerating near misses ("ds3231" -> "rtc-ds3231")."""
    if not pid:
        return None
    rec = records.get(str(pid))
    if rec:
        return rec
    want = str(pid).lower()
    for k in records:
        if want in k or k in want:
            return records[k]
    return None
