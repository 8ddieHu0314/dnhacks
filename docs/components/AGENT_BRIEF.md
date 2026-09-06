# Brief for component research subagents

You are researching a handful of electronic components for a technician-facing knowledge base.
Smart glasses will read your answers aloud to someone holding the part. A wrong pin or voltage
is worse than an empty field.

## Inputs
- Seed records: `docs/components/components.json` (find your ids in the `components` array).
  Seeds are guesses. Confirm or correct them, never copy them blindly.
- Full record schema: the JSON block under "Record schema" in
  `docs/components/spec_extraction_prompt.md`. Your output must match that shape exactly,
  with the seed's `id` and `kit` values carried over.

## Output
One file per component: `docs/components/records/<id>.json`, containing exactly one JSON
object in the schema. Nothing else in the file.

## Rules
1. Use WebSearch to find sources and WebFetch to actually open them. Only list a URL in
   `sources` if you fetched it and it loaded. Never list a URL from memory.
2. Source priority: ELEGOO's own tutorial PDF or product page for this kit first (to learn
   which exact variant ships), then the manufacturer datasheet for the chip or sensor, then a
   major retailer page (DigiKey, Mouser, Adafruit, SparkFun, Amazon) for price.
3. Every number in `pins`, `electrical`, and `key_specs` must come from a source you list, and
   the matching source entry's `used_for` must say which fields it backed.
4. Never guess a pinout. If unconfirmed, omit that pin entry and add the field name to
   `confidence.fields_uncertain`.
5. `visual_identification` describes what a camera sees: colour, shape, printed text, lead
   lengths, stripe or notch. Not what the part does.
6. `serial_number` is "not_applicable" unless the part truly carries one.
7. `market.unit_price_usd` is single-unit retail from a named retailer with the URL and
   today's date (2026-09-05). If only sold in packs, divide and say so in `confidence.notes`.
8. Plain English throughout. Define jargon inline. No em dashes anywhere.
9. Kit-specific generic parts (jumper wires, USB cable, breadboard) still get a full record.
   Use the ELEGOO listing plus a generic datasheet or retailer page.
10. Budget: at most 4 searches and 6 fetches per component. If a source is paywalled or dead,
    move on and record the gap.

## When you finish
Reply with a short table: id, confidence.overall, fields_uncertain, number of sources.
Then stop. Do not print the JSON in your reply.
