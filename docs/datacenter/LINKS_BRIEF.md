# Brief for data center documentation link scrapers

You are collecting official documentation links for hardware that a data center technician
touches with their hands. These links will be fetched later, chunked, and embedded into a
vector database so smart glasses can answer "what is this, how do I service it, is it safe"
while the tech is standing at the rack.

## Output
One JSON file per category group: `docs/datacenter/links/<group>.json`, containing:

```json
{
  "group": "<group>",
  "collected_on": "2026-09-05",
  "items": [
    {
      "id": "<group>/<slug>",
      "category": "server | switch | optic | pdu | ups | rack | cabling | storage | memory | cooling | safety | standard | tool | management",
      "vendor": "",
      "product_family": "",
      "model_examples": [],
      "doc_type": "owner_manual | spec_sheet | quick_start | installation_guide | safety_notice | standard | datasheet | pinout | product_page",
      "title": "",
      "url": "",
      "format": "html | pdf",
      "fetched_ok": true,
      "approx_pages": 0,
      "what_a_tech_uses_it_for": "one plain sentence",
      "key_procedures_inside": [],
      "safety_relevant": true,
      "notes": ""
    }
  ]
}
```

## Rules
1. Prefer the vendor's own documentation site or a standards body. Retailer pages only for
   commodity parts that have no vendor doc.
2. WebFetch every URL before listing it. Set `fetched_ok` honestly. If a PDF is huge and the
   fetch truncates, that still counts as ok if the title matches. If it 403s or 404s, try one
   alternative, then drop it.
3. Target 8 to 14 links per group. Depth beats breadth: one owner's manual that covers
   hot-swapping a PSU is worth more than five marketing pages.
4. Cover the two or three market-leading vendors per category, not one.
5. `key_procedures_inside` lists the hands-on tasks the doc actually covers (rail install,
   DIMM replacement, PSU hot swap, optic insertion, breaker reset, LOTO steps), as seen in
   its table of contents.
6. Plain English. No em dashes anywhere.
7. Budget: at most 20 searches and 30 fetches for the whole group.

## When you finish
Validate the JSON parses. Commit exactly your one file:
git add docs/datacenter/links/<group>.json && git commit -m "data: data center doc links, <group>" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
Retry on index.lock up to 5 times. Do not push. Reply with: item count, how many fetched_ok,
vendors covered, commit hash. Do not paste the JSON.
