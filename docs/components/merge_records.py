#!/usr/bin/env python3
"""Merge docs/components/records/*.json into components.json and write REPORT.md.

For each record file:
  - validate shape against the schema keys
  - check pin_count vs pinout length (unless pins flagged uncertain)
  - check quantity matches the seed
  - require at least one datasheet or manufacturer source
  - attach the full record under `details` on the matching seed entry
  - promote seed fields (mpn, pin_count, price_usd, canonical_name) from the record
  - set status = verified (or verified_with_gaps if fields_uncertain is non-empty)
"""
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
COMP = HERE / "components.json"
RECS = HERE / "records"
REPORT = HERE / "REPORT.md"

TOP_KEYS = ["id", "kit", "identity", "function", "pins", "electrical", "key_specs",
            "visual_identification", "wiring_to_uno", "safety", "troubleshooting",
            "market", "sources", "confidence"]


def main():
    data = json.loads(COMP.read_text())
    seeds = {c["id"]: c for c in data["components"]}
    problems, uncertain, merged, missing = [], [], [], []

    for cid, seed in seeds.items():
        path = RECS / f"{cid}.json"
        if not path.exists():
            missing.append(cid)
            continue
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            problems.append((cid, f"invalid JSON: {e}"))
            continue

        errs = []
        for k in TOP_KEYS:
            if k not in rec:
                errs.append(f"missing key {k}")
        if errs:
            problems.append((cid, "; ".join(errs)))
            continue

        pins = rec["pins"]
        unc = rec["confidence"].get("fields_uncertain", []) or []
        pins_flagged = any("pin" in str(u).lower() for u in unc)
        if pins.get("pinout") and pins.get("pin_count") != len(pins["pinout"]) and not pins_flagged:
            errs.append(f"pin_count {pins.get('pin_count')} != pinout length {len(pins['pinout'])}")

        q = rec["kit"].get("quantity_in_kit")
        if q != seed["qty"]:
            errs.append(f"quantity_in_kit {q} != seed qty {seed['qty']}")

        types = {s.get("type") for s in rec.get("sources", [])}
        if not types & {"datasheet", "manufacturer"}:
            errs.append("no datasheet or manufacturer source")

        n_src = len(rec.get("sources", []))
        if n_src == 0:
            errs.append("no sources")

        if errs:
            problems.append((cid, "; ".join(errs)))
        if unc:
            uncertain.append((cid, unc))

        # promote and attach
        ident = rec["identity"]
        seed["canonical_name"] = ident.get("canonical_name") or seed["canonical_name"]
        seed["mpn"] = ident.get("manufacturer_part_number") or seed["mpn"]
        if isinstance(pins.get("pin_count"), int):
            seed["pin_count"] = pins["pin_count"]
        price = rec["market"].get("unit_price_usd")
        if isinstance(price, (int, float)) and price > 0:
            seed["price_usd"] = price
        seed["voltage"] = rec["electrical"].get("operating_voltage") or seed["voltage"]
        seed["interface"] = rec["electrical"].get("interface") or seed["interface"]
        seed["function"] = rec["function"].get("one_sentence_plain_english") or seed["function"]
        seed["confidence"] = rec["confidence"].get("overall", "unknown")
        seed["source_count"] = n_src
        seed["status"] = "verified_with_gaps" if (unc or errs) else "verified"
        seed["details"] = rec
        merged.append(cid)

    data["_meta"]["last_merge"] = date.today().isoformat()
    data["_meta"]["status_values"] = ["seed", "verified", "verified_with_gaps"]
    COMP.write_text(json.dumps(data, indent=2) + "\n")

    lines = [f"# Component merge report, {date.today().isoformat()}", "",
             f"Merged {len(merged)} of {len(seeds)} records.", ""]
    if missing:
        lines += ["## Missing record files", ""] + [f"- {m}" for m in missing] + [""]
    if problems:
        lines += ["## Validation problems", ""] + [f"- **{c}**: {p}" for c, p in problems] + [""]
    if uncertain:
        lines += ["## Fields flagged uncertain by the researcher", ""]
        lines += [f"- **{c}**: {', '.join(map(str, u))}" for c, u in uncertain] + [""]
    lines += ["## Per record", "", "| id | status | confidence | sources | price USD |", "|---|---|---|---|---|"]
    for cid in seeds:
        s = seeds[cid]
        lines.append(f"| {cid} | {s['status']} | {s.get('confidence','')} | {s.get('source_count','')} | {s['price_usd']} |")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"merged={len(merged)} missing={len(missing)} problems={len(problems)} uncertain={len(uncertain)}")
    for c, p in problems:
        print(f"  PROBLEM {c}: {p}")


if __name__ == "__main__":
    main()
