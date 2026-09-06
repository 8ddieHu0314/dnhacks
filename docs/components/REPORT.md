# Component merge report, 2026-09-05

Merged 47 of 47 records.

## Validation problems

- **psu-mb102**: no datasheet or manufacturer source
- **stepper-28byj48**: no datasheet or manufacturer source
- **button**: no datasheet or manufacturer source
- **breadboard-830**: pin_count 830 != pinout length 4
- **usb-cable**: no datasheet or manufacturer source
- **battery-clip-9v**: no datasheet or manufacturer source
- **led-5mm**: no datasheet or manufacturer source
- **led-rgb**: no datasheet or manufacturer source
- **water-level**: no datasheet or manufacturer source

## Fields flagged uncertain by the researcher

- **uno-r3**: exact USB-to-serial chip and MCU package on this specific ELEGOO kit bundle revision (CH340C+SMD confirmed for the standalone board vs ATMEGA16U2+DIP-28 seen on an Amazon listing of the same product name); ELEGOO's own product pages returned HTTP 403 to the fetch tool on this research pass, so this could not be resolved further, board dimensions and weight are the official Arduino Uno Rev3 reference numbers, not independently confirmed for the ELEGOO clone, 5V/3.3V rail and per-pin current limits are official Arduino Uno Rev3 numbers, assumed to apply to the ELEGOO clone but not independently verified
- **lcd1602**: electrical.absolute_max_voltage (not found in a fetched source), backlight max current draw for this exact module (figure given is a typical reference value, not from a fetched datasheet for the ELEGOO-bundled unit), exact controller IC (HD44780 vs a pin-compatible clone) for the specific unit ELEGOO ships, since kit-bundled LCDs are commonly unbranded
- **proto-shield**: exact manufacturer / part number (still generic/unbranded; ELEGOO does not sell this piece as a standalone SKU, and no single OEM is confirmed across the many equivalent listings), electrical.absolute_max_voltage and electrical.max_current (the shield is a passive pass-through board with no rating of its own beyond the Uno underneath it; left blank intentionally rather than guessed)
- **psu-mb102**: exact regulator IC used (commonly cited as an AMS1117-family part in generic sources, but not confirmed via a fetched datasheet), absolute maximum input voltage before permanent damage (12V is the rated recommended max input, not an explicitly stated absolute maximum in the sources found), no manufacturer or datasheet-type source exists for this generic module; the three cited sources are a tutorial, a wiki-style parts database, and a retailer listing
- **uln2003-driver**: exact board-level brand/manufacturer (this is an unbranded generic design), typical_current for the assembled board while actually driving a motor load, ELEGOO's own specific Arduino pin choice in their kit tutorial (their tutorial and wiki pages returned 403/404 errors during research; a third-party DigiKey wiring example was used instead)
- **stepper-28byj48**: max_current (total draw at the connector under full 2-phase-on drive), exact manufacturer (this is a generic, multi-sourced part), absolute_max_voltage (no formal datasheet was found for this generic part)
- **servo-sg90**: exact stall current (no fetched source gave a specific number), whether the specific unit ELEGOO bundles is a genuine TowerPro part or a compatible clone
- **relay-5v**: exact numbered pin order printed on this specific board footprint (the datasheet gives a dimensioned drawing and a wiring-diagram icon rather than explicit pin numbers 1-5), coil pin non-polarity is confirmed from the datasheet's symmetric coil symbol, but silkscreen labeling can vary slightly between manufacturing batches
- **ir-receiver**: identity.manufacturer_part_number (ELEGOO's own wiki/product pages for this module returned empty or 404 on fetch, so I could not confirm whether the physical board is silkscreened VS1838B, HX1838, or AX-1838HS; a sibling ELEGOO kit's own datasheet folder on GitHub uses the name AX-1838HS for the same role), wiring_to_uno.example_connections (OUT pin number is a common illustrative example, not confirmed from an ELEGOO-specific sketch), market.unit_price_usd (converted from an Indian retailer's INR sale price using an approximate 87 INR/USD rate; Amazon, eBay, Makerfabs, and Abra-Electronics pages all failed to load through automated fetch this session)
- **joystick**: electrical.absolute_max_voltage and current figures (no formal manufacturer datasheet publishes these for this generic breakout; current is a calculated estimate, not a measured or published spec), market.unit_price_usd (converted from an Indian retailer's INR price using an approximate 87 INR/USD rate, and excludes local GST; Amazon listings would not render price through automated fetch this session)
- **dht11**: electrical.absolute_max_voltage (not stated in the sources checked), market.unit_price_usd (converted from an Indian retailer's INR price using an approximate 87 INR/USD rate; Adafruit's USD listing is for a different, discontinued breakout)
- **hc-sr04**: electrical.absolute_max_voltage and max_current (not separately specified beyond the 5V/15mA working figures in the datasheet checked), wiring_to_uno.example_connections (Trig/Echo pin numbers are a common illustrative example, not confirmed from a specific ELEGOO sketch)
- **dc-motor-fan**: electrical.typical_current, electrical.max_current, electrical.absolute_max_voltage, key_specs.no_load_speed_rpm
- **buzzer-active**: pins.pin_1_identification (exact ELEGOO lead length not independently confirmed), electrical.max_current, wiring_to_uno.example_connections (the exact UNO pin used in ELEGOO's active-buzzer lesson wiring diagram is an image and could not be extracted as text; the passive-buzzer lesson's D8 is given by analogy, not confirmed for this lesson)
- **buzzer-passive**: electrical.typical_current, electrical.max_current, electrical.absolute_max_voltage
- **74hc595**: identity.manufacturer_part_number (ELEGOO does not disclose which specific manufacturer's chip is bundled; several second-source this exact part)
- **button**: electrical.absolute_max_voltage, electrical.max_current, key_specs.actuation_force, key_specs.travel_distance, key_specs.pin_pitch, identity.manufacturer_part_number
- **pot-10k**: exact shaft length/knob variant ELEGOO ships (datasheet lists several shaft lengths and mounting styles; ELEGOO's own tutorial/product pages returned HTTP 403 on every fetch attempt so the specific suffix could not be confirmed), identity.manufacturer_part_number (WH148 is a widely cloned generic family name, not necessarily the literal branded part ELEGOO sources)
- **7seg-1digit**: identity.manufacturer (the fetched datasheet is from XLITX Technology, one of several factories making a 5161AS-numbered part; ELEGOO's actual supplier is unconfirmed), market.unit_price_usd (converted from Philippine peso at an approximate exchange rate, not a native USD retail listing)
- **7seg-4digit**: identity.manufacturer (the fetched datasheet is from XLITX Technology, one of several factories making a 3461BS-numbered part; ELEGOO's actual supplier is unconfirmed), market.unit_price_usd (converted from South African rand at an approximate exchange rate, not a native USD retail listing)
- **tilt-switch**: electrical.max_current (not explicit in the fetched manufacturer datasheet; a secondary retailer states <5 mA for the same part number but this is unconfirmed against the primary datasheet), electrical.operating_voltage and absolute_max_voltage (primary datasheet says 12V; a different retailer listing for the same part number claims 24V; the discrepancy is not resolved), key_specs.mechanical_life (sources disagree: 50,000 vs 100,000+ cycles), market.unit_price_usd (converted from Indian rupee at an approximate exchange rate, not a native USD listing; every Amazon fetch attempt failed with HTTP 500)
- **ir-remote**: identity.manufacturer_part_number, key_specs.key_count (strong new photo and retailer-listing evidence points to 17-18 keys, directly contradicting the seed's 21-key guess, but this was not confirmed on the literal UNO R3 Super Starter Kit's own tutorial PDF, only on a sibling ELEGOO kit's tutorial that appears to bundle the identical accessory), electrical.absolute_max_voltage, electrical.typical_current, electrical.max_current, visual_identification.shape_and_size (exact millimeter dimensions), market.unit_price_usd (best available price is for a closest-equivalent generic remote, not a listing of the literal ELEGOO-branded unit)
- **breadboard-830**: identity.manufacturer_part_number (MB-102 is the common industry name for this style of board, not a confirmed ELEGOO-specific SKU), row letter labeling convention (commonly a through e and f through j) is standard practice on this style of board but was not spelled out in the text of the fetched datasheets, only visible in their diagrams
- **usb-cable**: exact cable length shipped inside this specific ELEGOO kit (ELEGOO's own product page and tutorial blog both failed to load in this session, returning HTTP 403; the length given is from a generic retail USB A-B cable, not the ELEGOO-bundled one), confirmation that the UNO R3 board's onboard connector is specifically USB Type-B (this is standard, widely documented Arduino hardware knowledge, but no ELEGOO, Arduino, or datasheet source loaded successfully in this session to confirm it directly), manufacturer_part_number (this is a generic, unbranded cable with no confirmable MPN)
- **dupont-fm**: key_specs (length ~20cm appears in search-result snippets and the seed only, not confirmed by a page that loaded), market.unit_price_usd (retailer pages returned server errors when fetched; figure carried from the seed estimate, not independently verified today), identity.manufacturer_part_number (a generic wire like this has no MPN)
- **jumper-wire**: identity.manufacturer, identity.manufacturer_part_number, electrical.typical_current, electrical.max_current, electrical.absolute_max_voltage, key_specs.typical_length_range (exact ELEGOO 65-piece breakdown not confirmed, inferred from a comparable 140-piece commercial kit), market.unit_price_usd (computed from a generic-equivalent kit's pack price, not the exact ELEGOO-branded wire)
- **battery-clip-9v**: identity.manufacturer_part_number (ELEGOO does not publish one for this accessory), whether an actual 9V battery ships in the box (a search snippet of one Amazon listing title claims the kit includes a 9V battery, but a direct fetch of that page and of ELEGOO's own product page could not confirm the wording, and other sources note some kit revisions removed the battery), electrical.typical_current and electrical.max_current (no source found rates this connector's wire), key_specs.wire_length_cm (used a generic equivalent part's spec since ELEGOO's exact cable length could not be confirmed)
- **resistor-assorted**: kit.sku, identity.manufacturer, identity.manufacturer_part_number, key_specs.tolerance_percent, key_specs.resistance_range (exact list of values in this specific 120-piece ELEGOO bundle), electrical.absolute_max_voltage, electrical.max_current (exact figure, only the general formula is confirmed)
- **led-5mm**: kit.quantity_in_kit, key_specs.kit_colors_confirmed_by_sources (white color and total of 25), electrical.absolute_max_voltage, identity.manufacturer, identity.manufacturer_part_number, market.unit_price_usd (reference part, not confirmed identical to ELEGOO's bundled LEDs)
- **led-rgb**: identity.manufacturer, identity.manufacturer_part_number, kit.sku, electrical.absolute_max_voltage, electrical.max_current (as distinct from the typical value), wiring_to_uno resistor value specifically used in ELEGOO's own tutorial, key_specs luminous intensity and lens_type (sourced from a reference SparkFun part, not confirmed identical to the ELEGOO-bundled part)
- **thermistor**: identity.manufacturer, identity.manufacturer_part_number, key_specs.tolerance_resistance, key_specs.operating_temperature_range_c, market.unit_price_usd, wiring_to_uno.example_connections (exact analog pin ELEGOO's own tutorial uses is not confirmed)
- **diode-1n4007**: market.unit_price_usd, kit.sku, identity.manufacturer
- **photoresistor**: identity.manufacturer (no single OEM confirmed; GL55 series is widely second-sourced), kit.sku (ELEGOO's exact SKU not verified; elegoo.com blocked automated fetches with HTTP 403 on both the product page and the tutorial blog page), confirmation that the ELEGOO Super Starter Kit specifically ships the GL5528 variant rather than a sibling like GL5516 or GL5537 (could not confirm directly from ELEGOO's own materials; based on the seed record and consistent third-party documentation of ELEGOO kits, not on a fetched ELEGOO source), wiring_to_uno.required_extra_parts divider resistor value (commonly 10 kOhm in generic Arduino tutorials, not confirmed from an ELEGOO-specific source), electrical.typical_current and electrical.max_current (not given as explicit device ratings by the datasheet, only power and voltage maximums), market.unit_price_usd (see notes, this is a proxy from a similar but not confirmed-identical part)
- **pn2222**: kit.sku, identity.manufacturer (multiple manufacturers produce parts marked PN2222A; which one ELEGOO actually ships was not confirmed), whether ELEGOO ships the plain PN2222 or the tighter-spec PN2222A variant printed on the kit lid as just 'PN2222', market.unit_price_usd (derived by dividing a 10-pack price since single-unit retail listings were not accessible)
- **mpu6050**: identity.manufacturer_part_number (MPU-6050 vs QMI8658, both apparently used under the same "GY-521" board name per ELEGOO's own current tutorial), electrical.operating_voltage and absolute_max_voltage at the module level (only the bare-chip VDD figures were independently confirmed), electrical.max_current, visual_identification.color_and_markings (exact PCB color as shipped by ELEGOO not independently confirmed)
- **rtc-ds3231**: identity.manufacturer_part_number: still not confirmed for every physical unit. Second pass found two independent sources (an independently revised GitHub copy of this exact kit's tutorial, and a search-snippet match of ELEGOO's own current wiki page) naming the part a DS3231/ZS-042 module with an AT24C32N EEPROM and CR2032 battery, which now outweighs ELEGOO's own printed PDF tutorial text and required library name (both say "DS1307"). A separate genuinely ELEGOO-branded spare part named "DS1307 Module V03" (CR1220 battery) keeps a real DS1307 chip from being fully ruled out for some production runs., electrical.typical_current and electrical.max_current: sourced only from the DS1307 datasheet; DS3231-specific current draw figures were not confirmed by a fetched source this pass, electrical.absolute_max_voltage for a DS3231-based board specifically (only the DS1307 figure is confirmed), key_specs.backup_ram for a DS3231-based board (the DS3231 chip itself has no onboard battery-backed RAM; this field is confirmed only for the DS1307 chip), visual_identification.color_and_markings
- **rfid-rc522**: electrical.logic_level 5V-tolerance claim (widely repeated by sellers but not confirmed in the NXP datasheet, which specifies a 3.3V-class supply), visual_identification.color_and_markings, whether the ELEGOO kit's card-plus-fob bundle composition (1 card, 1 fob) still matches the seed data; not re-verified from a fetched source this pass
- **water-level**: electrical.absolute_max_voltage (no manufacturer datasheet exists for this generic part), electrical.max_current beyond the tutorial's less-than-20mA typical figure, visual_identification.color_and_markings, the exact manufacturer or model of the specific board ELEGOO bundles, since this is a generic, unbranded sensor design shared across many kit makers
- **sound-sensor**: identity.manufacturer_part_number (no MPN printed on ELEGOO's board), pins.pin_1_identification (physical left-to-right pin order on ELEGOO's specific board), electrical.absolute_max_voltage, electrical.typical_current, electrical.max_current, electrical.logic_level
- **keypad-4x4**: identity.manufacturer_part_number (generic part with no printed MPN)
- **led-matrix-8x8**: identity.manufacturer_part_number (the underlying 8x8 matrix chip's exact part number, for example 1088AS, was not confirmed for ELEGOO's specific board), electrical.absolute_max_voltage, electrical.typical_current (as distinct from the max figure), electrical.logic_level, visual_identification exact PCB dimensions of ELEGOO's specific board
- **pir-hc-sr501**: identity.manufacturer_part_number (the underlying sensing IC, commonly BISS0001, was not directly confirmed for ELEGOO's specific board), electrical.max_current (only one current figure was found, not explicitly labeled as typical versus max)
- **cap-ceramic-22pf**: identity.manufacturer, identity.manufacturer_part_number, key_specs.tolerance_percent (confirmed only for a comparable retail part, not ELEGOO's exact unit), electrical.absolute_max_voltage (same caveat), wiring_to_uno.example_connections (generic crystal-load wiring, not tied to a specific numbered ELEGOO lesson)
- **cap-ceramic-104**: identity.manufacturer, identity.manufacturer_part_number, electrical.absolute_max_voltage (confirmed only for a comparable retail part, not ELEGOO's exact unit), wiring_to_uno.example_connections (generic decoupling wiring; ELEGOO does not appear to dedicate a standalone tutorial lesson to this part on its own)
- **cap-electrolytic-10uf**: identity.manufacturer, identity.manufacturer_part_number, key_specs.tolerance_percent (from a comparable retail part, not ELEGOO's exact unit), key_specs.case_size (same caveat)
- **cap-electrolytic-100uf**: identity.manufacturer, identity.manufacturer_part_number, key_specs.tolerance_percent (not published for this exact part; omitted rather than guessed), identity.package (exact case size not independently confirmed for ELEGOO's unit)

## Per record

| id | status | confidence | sources | price USD |
|---|---|---|---|---|
| uno-r3 | verified_with_gaps | medium | 9 | 13.55 |
| lcd1602 | verified_with_gaps | medium | 3 | 9.95 |
| proto-shield | verified_with_gaps | medium | 5 | 2.95 |
| psu-mb102 | verified_with_gaps | medium | 3 | 1.99 |
| uln2003-driver | verified_with_gaps | medium | 4 | 1.5 |
| stepper-28byj48 | verified_with_gaps | medium | 3 | 3.16 |
| servo-sg90 | verified_with_gaps | high | 2 | 2.0 |
| relay-5v | verified_with_gaps | high | 2 | 0.44 |
| ir-receiver | verified_with_gaps | medium | 3 | 0.11 |
| joystick | verified_with_gaps | high | 3 | 0.38 |
| dht11 | verified_with_gaps | high | 4 | 0.68 |
| hc-sr04 | verified_with_gaps | high | 3 | 3.95 |
| dc-motor-fan | verified_with_gaps | medium | 3 | 1.95 |
| buzzer-active | verified_with_gaps | medium | 3 | 0.95 |
| buzzer-passive | verified_with_gaps | high | 3 | 1.5 |
| 74hc595 | verified_with_gaps | high | 3 | 0.98 |
| l293d | verified | high | 3 | 8.95 |
| button | verified_with_gaps | medium | 2 | 0.13 |
| pot-10k | verified_with_gaps | high | 2 | 0.5 |
| 7seg-1digit | verified_with_gaps | medium | 2 | 0.35 |
| 7seg-4digit | verified_with_gaps | medium | 2 | 0.3 |
| tilt-switch | verified_with_gaps | medium | 2 | 0.12 |
| ir-remote | verified_with_gaps | medium | 5 | 1.75 |
| breadboard-830 | verified_with_gaps | medium | 3 | 7.95 |
| usb-cable | verified_with_gaps | medium | 2 | 3.95 |
| dupont-fm | verified_with_gaps | medium | 2 | 0.1 |
| jumper-wire | verified_with_gaps | medium | 2 | 0.06 |
| battery-clip-9v | verified_with_gaps | medium | 1 | 2.95 |
| resistor-assorted | verified_with_gaps | medium | 3 | 0.02 |
| led-5mm | verified_with_gaps | medium | 4 | 0.2 |
| led-rgb | verified_with_gaps | medium | 2 | 2.3 |
| thermistor | verified_with_gaps | medium | 2 | 4.0 |
| diode-1n4007 | verified_with_gaps | medium | 2 | 0.05 |
| photoresistor | verified_with_gaps | medium | 2 | 0.95 |
| pn2222 | verified_with_gaps | medium | 2 | 0.2 |
| mpu6050 | verified_with_gaps | medium | 3 | 1.4 |
| rtc-ds3231 | verified_with_gaps | medium | 7 | 1.04 |
| rfid-rc522 | verified_with_gaps | medium | 3 | 3.89 |
| water-level | verified_with_gaps | medium | 2 | 3.99 |
| sound-sensor | verified_with_gaps | medium | 4 | 1.25 |
| keypad-4x4 | verified_with_gaps | high | 4 | 5.95 |
| led-matrix-8x8 | verified_with_gaps | medium | 4 | 8.99 |
| pir-hc-sr501 | verified_with_gaps | high | 4 | 1.99 |
| cap-ceramic-22pf | verified_with_gaps | medium | 5 | 0.14 |
| cap-ceramic-104 | verified_with_gaps | medium | 5 | 0.32 |
| cap-electrolytic-10uf | verified_with_gaps | medium | 4 | 0.195 |
| cap-electrolytic-100uf | verified_with_gaps | medium | 4 | 0.79 |
