#!/usr/bin/env python3
"""Query components.json from the shell. Stdlib only.

Examples:
  python3 docs/components/query.py                       # list everything
  python3 docs/components/query.py --search relay        # substring over all text fields
  python3 docs/components/query.py --category sensor     # exact category
  python3 docs/components/query.py --pins 16             # exact pin count
  python3 docs/components/query.py --min-pins 5 --max-pins 8
  python3 docs/components/query.py --kit super --fields id,pin_count,price_usd
  python3 docs/components/query.py --id hc-sr04 --json   # full record as JSON
  python3 docs/components/query.py --stats               # counts and total kit price
"""
import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).with_name("components.json")
DEFAULT_FIELDS = ["id", "canonical_name", "category", "qty", "pin_count", "price_usd"]


def load():
    with DATA.open() as f:
        return json.load(f)


def matches(rec, a):
    if a.id and rec["id"] != a.id:
        return False
    if a.category and rec["category"] != a.category:
        return False
    if a.kit and rec["kit"] != a.kit:
        return False
    if a.pins is not None and rec["pin_count"] != a.pins:
        return False
    if a.min_pins is not None and rec["pin_count"] < a.min_pins:
        return False
    if a.max_pins is not None and rec["pin_count"] > a.max_pins:
        return False
    if a.search:
        blob = " ".join(str(v) for v in rec.values()).lower()
        if a.search.lower() not in blob:
            return False
    return True


def table(rows, fields):
    if not rows:
        print("no matches")
        return
    widths = {f: max(len(f), *(len(str(r.get(f, ""))) for r in rows)) for f in fields}
    line = "  ".join(f.ljust(widths[f]) for f in fields)
    print(line)
    print("  ".join("-" * widths[f] for f in fields))
    for r in rows:
        print("  ".join(str(r.get(f, "")).ljust(widths[f]) for f in fields))


def stats(recs):
    by_cat, by_kit = {}, {}
    total = 0.0
    for r in recs:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        by_kit[r["kit"]] = by_kit.get(r["kit"], 0) + 1
        total += r["qty"] * r["price_usd"]
    print(f"records: {len(recs)}")
    print("by kit:", ", ".join(f"{k}={v}" for k, v in sorted(by_kit.items())))
    print("by category:", ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    print(f"sum of qty x unit price (seed estimates): ${total:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search", help="case-insensitive substring over every field")
    p.add_argument("--id")
    p.add_argument("--category")
    p.add_argument("--kit", choices=["super", "most_complete"])
    p.add_argument("--pins", type=int)
    p.add_argument("--min-pins", type=int)
    p.add_argument("--max-pins", type=int)
    p.add_argument("--fields", help="comma-separated columns to show")
    p.add_argument("--json", action="store_true", help="print matching records as JSON")
    p.add_argument("--stats", action="store_true")
    a = p.parse_args()

    recs = [r for r in load()["components"] if matches(r, a)]
    if a.stats:
        stats(recs)
        return
    if a.json:
        json.dump(recs, sys.stdout, indent=2)
        print()
        return
    fields = a.fields.split(",") if a.fields else DEFAULT_FIELDS
    table(recs, fields)


if __name__ == "__main__":
    main()
