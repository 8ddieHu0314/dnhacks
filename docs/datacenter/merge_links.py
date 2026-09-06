#!/usr/bin/env python3
"""Merge docs/datacenter/links/*.json into hardware_links.json and print a summary.

Validates each group file, drops nothing, but flags items with fetched_ok false,
missing urls, or duplicate urls across groups.
"""
import json
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "hardware_links.json"
REPORT = HERE / "LINKS_REPORT.md"
REQUIRED = ["id", "category", "vendor", "doc_type", "title", "url", "fetched_ok",
            "what_a_tech_uses_it_for", "safety_relevant"]


def main():
    items, problems, groups = [], [], []
    for path in sorted((HERE / "links").glob("*.json")):
        try:
            g = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            problems.append((path.name, f"invalid JSON: {e}"))
            continue
        groups.append((g.get("group", path.stem), len(g.get("items", []))))
        for it in g.get("items", []):
            missing = [k for k in REQUIRED if k not in it]
            if missing:
                problems.append((it.get("id", "?"), f"missing {', '.join(missing)}"))
            if not it.get("fetched_ok"):
                problems.append((it.get("id", "?"), "fetched_ok is false"))
            it.setdefault("group", g.get("group", path.stem))
            items.append(it)

    seen = Counter(i.get("url") for i in items)
    for url, n in seen.items():
        if n > 1:
            problems.append((url, f"duplicate url in {n} items"))

    OUT.write_text(json.dumps({
        "_meta": {"merged_on": date.today().isoformat(), "groups": dict(groups),
                  "item_count": len(items)},
        "items": items,
    }, indent=2) + "\n")

    cats = Counter(i.get("category") for i in items)
    vendors = Counter(i.get("vendor") for i in items)
    ok = sum(1 for i in items if i.get("fetched_ok"))
    lines = [f"# Data center link merge report, {date.today().isoformat()}", "",
             f"{len(items)} items across {len(groups)} groups, {ok} fetched ok.", "",
             "## By category", ""] + [f"- {k}: {v}" for k, v in sorted(cats.items())] + \
            ["", "## By vendor", ""] + [f"- {k}: {v}" for k, v in vendors.most_common()] + \
            ["", "## Problems", ""] + ([f"- **{a}**: {b}" for a, b in problems] or ["- none"])
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"items={len(items)} groups={len(groups)} fetched_ok={ok} problems={len(problems)}")
    for a, b in problems:
        print(f"  PROBLEM {a}: {b}")


if __name__ == "__main__":
    main()
