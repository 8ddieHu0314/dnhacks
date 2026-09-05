# PPE Sentinel roadmap

Plain-English list of what is built, what is next, and what we learned that changes
the plan. Architecture lives in `docs/ARCHITECTURE.md`. This file is about order and
intent, not design.

## Done

- **Spine, rules, debounce.** Local SH17 model plus the local tools ONNX model, three
  rules (bare hands, tool in bare hand, no helmet), 2 second debounce, spoken sentence.
- **Playground.** Gradio app with Image, Video and Live webcam tabs. The Live tab runs
  the real spine with real wall-clock debounce and the tools model on its background
  cadence, so it is a faithful stand-in for the phone stream.
- **Phone endpoint.** `server.py` accepts posted frames into a latest-wins mailbox.

## Next

### 1. Glasses audio (the real alert path)

Finding from the Meta DAT SDK docs (searched 2026-09-05): there is no Meta speech API
and no way to change what Meta AI says. Third-party apps get the glasses as a Bluetooth
speaker, nothing more. So the design is:

- The Mac produces the sentence (already done, `spoken_sentence` in `core/rules.py`).
- The iPhone app speaks it with Apple's on-device text to speech (`AVSpeechSynthesizer`)
  through an audio session in playback mode. iOS routes it to the paired glasses over
  A2DP, the high quality media profile.
- Changing what the glasses say means editing the sentence strings in `core/rules.py`.
  Voice, rate and language are app settings on the phone.
- Stay on A2DP for alerts. Capturing the wearer's voice from the glasses mic switches
  the link to HFP, which drops playback to 8 kHz mono for the whole session.
- Meta AI's own voice shares the speaker but is not scriptable. If Meta ships an
  assistant extension API later, revisit.

Owner: cofounder (iPhone DAT app). Mac side needs nothing new beyond `/latest`.

### 2. Vest detection and rule

- SH17 already detects `safety-vest`, but at a desk webcam the box flickers when worn.
  Diagnose first: lower the confidence slider to 0.25 and try yolo8m. If the box
  steadies, it is a threshold problem, not a model problem.
- Then add per-class confidence for `safety-vest` in `SH17Detector` (keep 0.4 for
  everything else) and a sticky-presence plugin that re-adds the last vest box for
  about 1.5 seconds when the detector drops a frame.
- Then a `NO_VEST` rule shaped like `no_helmet`: applicable when a `person` box is
  visible, active when no vest box sits inside it, sentence "Put your vest on."

### 3. Person tracker for `other` rules

`NO_HELMET` (and `NO_VEST`) fire once per frame, not once per person. With several
people in view that is "someone here has no helmet". ByteTrack over `person` boxes,
timers keyed by `(rule, track_id)`. Slot is marked in `core/spine.py`.

### 4. Render after the fact (session recap)

Session folder, events log, clips around each violation, then the report agent and
the knowledge graph. Design is in `docs/ARCHITECTURE.md` under "Recording and the
session record". Nothing built yet.

## Parking lot

- Canonical class-name mapping in each detector (today rules compare raw lowercase
  names). Do this before adding a third detector.
- Measured accuracy numbers per class on our own footage. Nothing in the repo has
  benchmarked SH17 on webcam or glasses frames.
