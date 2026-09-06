#!/usr/bin/env python3
"""Query hardware_links.json. Stdlib only.

  python3 docs/datacenter/query_links.py                     # everything
  python3 docs/datacenter/query_links.py --category pdu
  python3 docs/datacenter/query_links.py --vendor Dell
  python3 docs/datacenter/query_links.py --search "hot swap"  # over all text incl. procedures
  python3 docs/datacenter/query_links.py --safety             # only safety_relevant items
  python3 docs/datacenter/query_links.py --urls               # one url per line, for a fetcher
"""
import argparse
import json
from pathlib import Path

DATA = Path(__file__).with_name("hardware_links.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category")
    p.add_argument("--vendor")
    p.add_argument("--doc-type")
    p.add_argument("--search")
    p.add_argument("--safety", action="store_true")
    p.add_argument("--urls", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    items = json.loads(DATA.read_text())["items"]
    out = []
    for it in items:
        if a.category and it.get("category") != a.category:
            continue
        if a.vendor and a.vendor.lower() not in str(it.get("vendor", "")).lower():
            continue
        if a.doc_type and it.get("doc_type") != a.doc_type:
            continue
        if a.safety and not it.get("safety_relevant"):
            continue
        if a.search and a.search.lower() not in json.dumps(it).lower():
            continue
        out.append(it)

    if a.json:
        print(json.dumps(out, indent=2))
        return
    if a.urls:
        for it in out:
            print(it["url"])
        return
    if not out:
        print("no matches")
        return
    for it in out:
        flag = " [safety]" if it.get("safety_relevant") else ""
        print(f"{it['id']}  ({it.get('category')}, {it.get('vendor')}, {it.get('doc_type')}){flag}")
        print(f"    {it.get('title')}")
        print(f"    {it.get('url')}")
        print(f"    use: {it.get('what_a_tech_uses_it_for')}")
    print(f"\n{len(out)} items")


if __name__ == "__main__":
    main()
