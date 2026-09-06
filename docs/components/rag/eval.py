#!/usr/bin/env python3
"""JSON lookup vs BM25 vs embeddings on the real question: a fuzzy description comes out of a
vision model, which record is it? Prints hit@1, hit@3 and latency per mode and class."""
import json
import time
from collections import defaultdict

from search import Searcher

# (class, query, acceptable ids)
EVAL = [
    # A. what a vision model would say about an unlabeled part
    ("vision", "small blue rectangular plastic block with five metal legs underneath, printed SONGLE on the side", {"relay-5v"}),
    ("vision", "blue cube with 5 pins", {"relay-5v"}),
    ("vision", "black chip with 16 legs, printed 74HC595", {"74hc595"}),
    ("vision", "long black chip with 16 pins, notch at one end, text unreadable", {"74hc595", "l293d"}),
    ("vision", "blue board with two silver cylinders that look like eyes", {"hc-sr04"}),
    ("vision", "tiny blue box with a white grid pattern and three pins", {"dht11"}),
    ("vision", "green board with a white plastic socket, four small LEDs and a black chip", {"uln2003-driver"}),
    ("vision", "small silver round motor with a flat gearbox and five colored wires ending in a white plug", {"stepper-28byj48"}),
    ("vision", "little blue plastic motor with a white plastic arm and brown red orange wires", {"servo-sg90"}),
    ("vision", "black round disc with a hole in the middle, two legs, sticker on top", {"buzzer-active", "buzzer-passive"}),
    ("vision", "black square base with a round dome on top like a thumb stick", {"joystick"}),
    ("vision", "black cylinder with two wire legs that rattles when shaken", {"tilt-switch"}),
    ("vision", "black rectangle showing one figure eight of red segments, ten legs", {"7seg-1digit"}),
    ("vision", "four figure eights side by side, red segments, twelve pins", {"7seg-4digit"}),
    ("vision", "green screen with 16 metal pins along the top edge, shows two rows of text", {"lcd1602"}),
    ("vision", "clear plastic bulb with four legs of different lengths", {"led-rgb"}),
    ("vision", "small black bead on two thin wire legs", {"thermistor"}),
    ("vision", "small black cylinder with a silver stripe at one end and two wires", {"diode-1n4007"}),
    ("vision", "flat round disc with a zigzag red line on top and two legs", {"photoresistor"}),
    ("vision", "small black half cylinder with three legs and a flat face", {"pn2222"}),
    ("vision", "white board with hundreds of tiny holes in rows", {"breadboard-830"}),
    ("vision", "blue board with a black chip, a pin header and a big round coin battery", {"rtc-ds3231"}),
    ("vision", "blue square board with a coil trace around the edge, comes with a white card", {"rfid-rc522"}),
    ("vision", "small board with a tiny chip and 8 pins labeled SCL SDA XDA XCL", {"mpu6050"}),
    ("vision", "white plastic dome on a green board with two orange dials", {"pir-hc-sr501"}),
    ("vision", "black remote with white round buttons and a red power button at the top", {"ir-remote"}),
    ("vision", "tan cylinder with colored stripes and a wire out of each end", {"resistor-assorted"}),
    ("vision", "blue knob on a black body with three legs", {"pot-10k"}),
    ("vision", "tiny black square button with four legs at the corners", {"button"}),
    ("vision", "grid of 64 small red dots on a board with a chip labeled MAX7219", {"led-matrix-8x8"}),
    # B. how a technician would actually ask
    ("question", "which part reads badge cards", {"rfid-rc522"}),
    ("question", "what do I use to make a motor spin in both directions", {"l293d"}),
    ("question", "which one beeps as soon as it gets power without any code", {"buzzer-active"}),
    ("question", "which sensor measures how far away something is", {"hc-sr04"}),
    ("question", "thing that keeps the time even when the power is off", {"rtc-ds3231"}),
    ("question", "part that lets 5 volts switch a wall powered lamp", {"relay-5v"}),
    ("question", "how do I know which leg of the LED is positive", {"led-5mm"}),
    ("question", "what senses a person walking past", {"pir-hc-sr501"}),
    ("question", "which capacitor has polarity and can blow if backwards", {"cap-electrolytic-10uf", "cap-electrolytic-100uf"}),
    ("question", "what turns a 9 volt battery into 5 volts for the breadboard", {"psu-mb102"}),
]

# C. exact field lookups, where the answer is a full set. JSON does these with a filter.
FIELD = [
    ("which parts have 16 pins", lambda c: c["pin_count"] == 16),
    ("parts that talk over I2C", lambda c: "i2c" in str(c["interface"]).lower()),
    ("everything in the most complete kit only", lambda c: c["kit"] == "most_complete"),
]


def main():
    s = Searcher()
    modes = ["json", "bm25", "embed", "hybrid"]
    stats = {m: defaultdict(lambda: [0, 0, 0, 0.0]) for m in modes}  # cls -> [n, hit1, hit3, ms]
    misses = {m: [] for m in modes}
    for cls, q, ok in EVAL:
        for m in modes:
            t = time.time()
            hits = [h[0] for h in s.search(q, m, k=3)]
            ms = (time.time() - t) * 1000
            st = stats[m][cls]
            st[0] += 1
            st[3] += ms
            if hits and hits[0] in ok:
                st[1] += 1
            if any(h in ok for h in hits):
                st[2] += 1
            else:
                misses[m].append((cls, q, sorted(ok), hits))

    print("## Fuzzy retrieval: does the right record come back?\n")
    print("| class | mode | n | hit@1 | hit@3 | avg ms |")
    print("|---|---|---|---|---|---|")
    for cls in ("vision", "question"):
        for m in modes:
            n, h1, h3, ms = stats[m][cls]
            print(f"| {cls} | {m} | {n} | {h1}/{n} ({100*h1/n:.0f}%) | {h3}/{n} ({100*h3/n:.0f}%) | {ms/n:.1f} |")

    print("\n## Exact field lookups: JSON filter vs RAG top-k\n")
    print("| query | truth (JSON filter) | bm25 recall | embed recall | hybrid recall |")
    print("|---|---|---|---|---|")
    for q, pred in FIELD:
        truth = {c["id"] for c in s.comps if pred(c)}
        k = len(truth)
        row = []
        for m in ("bm25", "embed", "hybrid"):
            got = {h[0] for h in s.search(q, m, k=k)}
            row.append(f"{len(got & truth)}/{k}")
        print(f"| {q} | {len(truth)} exact | {row[0]} | {row[1]} | {row[2]} |")

    print("\n## Embedding misses (top 3 shown)\n")
    for cls, q, ok, hits in misses["embed"]:
        print(f"- [{cls}] \"{q}\" wanted {ok}, got {hits}")
    print("\n## BM25 misses\n")
    for cls, q, ok, hits in misses["bm25"]:
        print(f"- [{cls}] \"{q}\" wanted {ok}, got {hits}")
    print("\n## Hybrid misses\n")
    for cls, q, ok, hits in misses["hybrid"]:
        print(f"- [{cls}] \"{q}\" wanted {ok}, got {hits}")


if __name__ == "__main__":
    main()
