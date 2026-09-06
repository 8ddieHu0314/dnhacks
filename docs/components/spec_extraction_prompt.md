# Component spec extraction, orchestrator prompt

Paste everything below the line to the agent that will run the job.

---

You are building a component knowledge base that will be chunked and embedded into a vector
database. A separate system uses smart glasses to help data center technicians identify parts
and follow procedures, and it will answer questions by retrieving these records. Precision
matters more than coverage: a wrong pin number in this database will be read out loud to a
technician holding a live board.

## Job

Produce one JSON record per line item in the ELEGOO UNO R3 Super Starter Kit (35 items, listed
below). Fan out one subagent per item (run up to 8 in parallel), then merge, validate, and
write the results.

## Kit line items (name as printed on the lid, quantity)

1. UNO R3 Controller Board, 1
2. LCD1602 Module (with pin header), 1
3. Prototype Expansion Module, 1
4. Power Supply Module, 1
5. ULN2003 Stepper Motor Driver Module, 1
6. Stepper Motor, 1
7. Servo Motor SG90, 1
8. 5V Relay, 1
9. IR Receiver Module, 1
10. Joystick Module, 1
11. DHT11 Temperature and Humidity Module, 1
12. Ultrasonic Sensor, 1
13. Fan Blade and 3-6V Motor, 1
14. Active Buzzer, 1
15. Passive Buzzer, 1
16. 74HC595 IC, 1
17. L293D IC, 1
18. Button, 5
19. Potentiometer 10K, 1
20. 1 Digit 7-Segment Display, 1
21. 4 Digit 7-Segment Display, 1
22. Tilt Ball Switch, 1
23. Remote Control, 1
24. 830 Tie-Points Breadboard, 1
25. USB Cable, 1
26. Female-to-Male Dupont Wire, 10
27. Breadboard Jumper Wire, 65
28. 9V Battery with Snap-on Connector Clip, 1
29. Resistor, 120
30. LED, 25
31. RGB LED, 2
32. Thermistor, 1
33. Diode Rectifier, 2
34. Photoresistor (Photocell), 2
35. NPN Transistor PN2222, 2

## Record schema (every subagent returns exactly this shape)

```json
{
  "id": "elegoo-uno-r3-kit/<slug>",
  "kit": { "name": "ELEGOO UNO R3 Super Starter Kit", "sku": "EL-KIT-003 (verify)", "quantity_in_kit": 0 },
  "identity": {
    "name_on_kit": "",
    "canonical_name": "",
    "aliases": [],
    "category": "microcontroller | sensor | actuator | driver_ic | passive | display | power | wiring | tool",
    "manufacturer": "",
    "manufacturer_part_number": "",
    "generic_equivalents": [],
    "serial_number": "not_applicable | <value>",
    "package": "DIP-16 | TO-92 | 5mm through-hole | module | cable | ..."
  },
  "function": {
    "one_sentence_plain_english": "",
    "what_it_is_used_for": "",
    "how_it_works_briefly": ""
  },
  "pins": {
    "pin_count": 0,
    "pinout": [ { "pin": 1, "name": "", "role": "", "notes": "" } ],
    "pin_1_identification": "how to find pin 1 or the positive lead by looking at the part",
    "polarity_sensitive": true
  },
  "electrical": {
    "operating_voltage": "",
    "absolute_max_voltage": "",
    "typical_current": "",
    "max_current": "",
    "logic_level": "",
    "interface": "digital | analog | PWM | I2C | SPI | UART | 1-wire | none"
  },
  "key_specs": {},
  "visual_identification": {
    "shape_and_size": "",
    "color_and_markings": "",
    "printed_text_to_look_for": "",
    "easily_confused_with": []
  },
  "wiring_to_uno": {
    "example_connections": [ { "part_pin": "", "uno_pin": "" } ],
    "required_extra_parts": [],
    "arduino_library": ""
  },
  "safety": {
    "hazards": [],
    "handling_notes": [],
    "common_mistakes": []
  },
  "troubleshooting": {
    "symptoms_and_causes": [ { "symptom": "", "likely_cause": "", "check": "" } ],
    "how_to_test_with_a_multimeter": ""
  },
  "market": {
    "unit_price_usd": 0.0,
    "price_source_url": "",
    "price_checked_on": "YYYY-MM-DD",
    "lifecycle_status": "active | NRND | EOL | unknown",
    "typical_suppliers": []
  },
  "sources": [ { "title": "", "url": "", "type": "datasheet | manufacturer | retailer | tutorial | wiki", "used_for": "" } ],
  "confidence": {
    "overall": "high | medium | low",
    "fields_uncertain": [],
    "notes": ""
  }
}
```

## Subagent brief (send this to each subagent, filling in the item)

> You are researching ONE component for a technician-facing knowledge base: **<name_on_kit>**,
> quantity **<qty>**, from the ELEGOO UNO R3 Super Starter Kit. Return one JSON object matching
> the schema you were given. Rules:
>
> 1. Identify the exact part ELEGOO ships (module vs bare component matters for pin count).
>    Use the ELEGOO tutorial PDF for this kit and the ELEGOO product page as the first source,
>    then the manufacturer datasheet for the underlying chip or sensor.
> 2. Every number in `pins`, `electrical`, and `key_specs` must be traceable to a datasheet or
>    manufacturer page listed in `sources`. Tutorials and blogs may fill `function`, `wiring`,
>    and `troubleshooting` only.
> 3. Never guess a pin. If the pinout cannot be confirmed from a datasheet, set `pin_count` to
>    what you can confirm, leave the unconfirmed entries out, and list the field in
>    `confidence.fields_uncertain`.
> 4. `visual_identification` is written for someone looking at the part through a camera:
>    describe what they can see (colour, shape, printed text, lead length), not what it does.
> 5. `serial_number` is `not_applicable` unless the part actually carries one.
> 6. Price is the current single-unit retail price in USD from a named retailer (Amazon,
>    DigiKey, Mouser, Adafruit, SparkFun). Record the URL and today's date. If sold only in
>    packs, compute the per-unit price and say so in `confidence.notes`.
> 7. Plain English throughout. A reader with zero electronics background must be able to
>    follow `function` and `safety`. Define any jargon inline.
> 8. Return only the JSON. No prose before or after it.

## Merge and validation (you, the orchestrator, do this after all subagents return)

1. Write each record to `docs/components/records/<slug>.json`.
2. Validate every record: all schema keys present, `pin_count` equals the length of `pinout`
   unless `fields_uncertain` mentions pins, `quantity_in_kit` matches the list above, at least
   one `datasheet` or `manufacturer` source per record, and `price_checked_on` is today.
3. Write `docs/components/index.json` with one line per record: id, canonical_name, MPN,
   category, pin_count, unit_price_usd, confidence.overall.
4. Write `docs/components/REPORT.md` listing: records that failed validation, every field
   flagged uncertain, and any item where subagents disagreed on the MPN.
5. Do not invent data to satisfy validation. A record with honest gaps beats a complete one
   that is wrong.

## Chunking hint for whoever loads the vector database

Embed each record as three chunks: (a) identity + function + visual_identification, (b) pins +
electrical + wiring, (c) safety + troubleshooting. Prefix every chunk with the canonical name
and MPN so retrieval works when a technician only knows one of them.
