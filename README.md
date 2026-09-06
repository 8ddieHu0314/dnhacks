# Vest Practices

**A journeyman electrician in your ear, running on Ray-Ban Meta glasses.**

Look at an electrical part. The glasses see it, the system finds it in a curated parts
knowledge base, and a voice in the glasses tells you what it is, what its pins do, and what
will break it. Hands stay on the work. Nothing to hold, nothing to type.

Built in one weekend at DNHacks 2026 for the Energy & Industrialization track.

> Demo video: _[link goes here before submission]_
> Website: https://vestpractices.vercel.app/

## Why

The trade is taught one apprentice at a time, and there are not enough journeymen left to
do the teaching. Skilled-labor shortages are the leading cause of construction delay in the
United States (AGC of America 2025 workforce survey, 1,342 firms). The workers who are
showing up often do not know the parts in front of them, and the person who could tell them
is on another site.

Vest Practices is a personal trainer for that worker. It answers "what is this and how do I
use it" at the moment the question comes up, from a knowledge base the team curated and
cited. Because the reasoning layer is an AI model reading from that knowledge base, the same
loop adapts to a new trade or a new employer's parts list by swapping the data, not the code.

## What it does today

1. **Wear the glasses, open the app.** The iPhone app pulls the glasses camera stream and
   relays frames to a Mac on the same Wi-Fi. The Mac is found automatically over Bonjour.
2. **Look at a part.** A small detector on the Mac, fine-tuned on our own footage, spots
   the component and waits for the wearer to hold still.
3. **It finds the part.** The frame is matched against a 47-part knowledge base built from
   the Elegoo Uno R3 kit: pinouts, electrical limits, wiring examples, and a source for each
   spec. Fields nobody could verify are flagged, not guessed.
4. **It speaks.** The name and one key spec are spoken into the glasses within a few
   seconds. Ask "describe" and you get the fuller answer. It stays quiet while the part in
   view has not changed.
5. **It keeps a record.** Every identification is written to a session report with the
   frame, the answer, and the confidence, so a supervisor can review the shift afterwards.

There is also a scene mode that narrates a walk through a site, hazards first, and a
laptop-webcam mode for demoing without the glasses.

## How it is built

```text
 Ray-Ban Meta glasses          iPhone                    Mac (:8787)
 ┌──────────────────┐   BT    ┌───────────────┐  Wi-Fi  ┌─────────────────────────────┐
 │ camera           │ ──────► │ Glasses       │ ──────► │ relay_receiver               │
 │                  │         │ Inspector app │  JPEG   │  detector (YOLOv8n, ~43 ms)  │
 │ speakers  ◄──────│ ◄────── │ text-to-speech│ ◄────── │  part identification         │
 └──────────────────┘  audio  └───────────────┘ sentences│  parts knowledge base (47)   │
                                                         │  live dashboard + report     │
                                                         └─────────────────────────────┘
```

| Piece | Where | What it is |
|---|---|---|
| Glasses app | `clients/ios/GlassesInspector` | Fork of Meta's Device Access Toolkit camera sample. Hardware video decode, throttled JPEG relay, USB-first then Wi-Fi transport, speech routed to the glasses over Bluetooth. |
| Mac brain | `services/relay_receiver` | Single-file FastAPI server. Receives frames, runs the detector, identifies parts, streams spoken sentences back as they land, serves the dashboard and the session report. |
| Component detector | `weights/components_v3.onnx`, `tools/components` | Single-class YOLOv8n at 960 px, fine-tuned on 553 of our own webcam frames plus 250 relabelled public images. Labels came from Grounding DINO proposals verified by a vision model, then audited by hand. |
| Parts knowledge base | `docs/components` | 47 cited records for the Elegoo Uno R3 kit, a merge and validation script, and a retrieval layer with its own eval. |
| Field notes | `docs/meta-glasses-field-notes.md` | The running log of every measurement, outage and root cause from the weekend. |

Numbers we measured, not guessed:

| Measurement | Value |
|---|---|
| Phone to Mac, 504x896 over venue Wi-Fi | ~15 fps, 171 ms per frame |
| Phone to Mac, 720x1280 | ~12 fps, 252 ms per frame |
| Detector, CPU, 1280x720 frame | 43 ms |
| Detector, held-out own frames | 53% box recall, 63% of frames boxed, ~0.15 stray boxes per empty frame |
| Knowledge base retrieval, visual descriptions | 80% hit@1, 93% hit@3 (BM25, 30 queries) |
| Fake round trip, first spoken sentence | 0.7 s |

## What we learned

**Streaming off the glasses is hard, and the weekend made it harder.** The glasses talk to
the phone over Bluetooth Classic, and that link is the ceiling. Push it and Wi-Fi collapses
while Bluetooth streams. A free Apple developer account cannot sign the Wi-Fi hotspot
entitlement, so Bluetooth it is. On the phone, one main-thread hop per frame froze the UI at
720p until every frame went through a single latest-frame slot instead. iOS refuses to
resolve the default Bonjour hostname, so the Mac had to advertise itself by plain hostname
or the phone sat in "preparing" forever. Each of these cost hours we did not have, and each
one is written down in the field notes with the fix.

**The detector was only as good as its labels.** The public Arduino-parts model fired on
furniture and shirts and missed the real parts through a webcam. We built our own data
instead. A label audit found 131 of 791 boxes were furniture, clothing or bare cable, and
recall rose from 42% to 53% once the labels were honest. Overnight runs of bigger models did
not beat the small one on the audited set, so the small one shipped.

**A knowledge base should refuse to guess.** Every spec field in the parts records carries
a source. Where the sources disagreed or no datasheet existed, the field is flagged as
uncertain rather than filled in. The merge report lists every such field. A trainee's
first wiring mistake should not come from a made-up pin.

## What is next

- **Step-by-step guidance.** "Show me a red wire. Land it on COIL+." The camera verifies
  colour and placement before the next step. The state machine that advances steps is
  designed, not yet built.
- **Gear check before starting.** Eye protection and gloves confirmed across several
  frames. Prototyped on a separate branch, not part of this demo.
- **A second sense before danger.** A field-sensing probe on the temple arm confirms a
  conductor reads ambient before the camera's "plug is out" is trusted. The Arduino probe
  circuit exists in `firmware/context_node` and responds to touch, and its integration is
  blocked on hardware.
- **More trades, more parts.** The knowledge base format and the identification loop are
  independent of the Arduino kit. A breaker panel catalog is the same pipeline with new records.

## Run it

You need a Mac, an iPhone, Ray-Ban Meta glasses paired to the Meta AI app, and Xcode 26.4 or
newer. Without the glasses, the laptop-webcam mode still exercises everything on the Mac.

Mac side:

```bash
services/relay_receiver/run.sh              # creates a venv, installs, starts :8787
INSPECT_FAKE=1 services/relay_receiver/run.sh   # canned answers, no API key needed
```

Put the vision model API key in `services/relay_receiver/.env`. Open http://localhost:8787
for the live dashboard. Any JPEG posted to `POST /frame` shows up there, which is how the
webcam demo works.

Phone side: open `clients/ios/GlassesInspector/CameraAccess.xcodeproj` in Xcode, run it on
the phone, pick the Mac from the receiver list, and press Preview. The relay chip shows fps
and latency live.

Knowledge base:

```bash
python3 docs/components/query.py --search i2c              # exact filters over the records
python3 docs/components/rag/search.py "blue cube with 5 pins"   # fuzzy retrieval
cd docs/components/rag && python3 eval.py                 # retrieval eval (downloads a small embedding model)
```

Detector fine-tune and eval tooling is under `tools/components`, with the workflow written up
in `services/relay_receiver/README.md`.

## Also in this repo

- `services/vision_api`: an earlier, tested FastAPI scaffold with sessions, versioned
  workflow packages and a pluggable vision engine. Not wired to the phone app. Its README is
  in that folder.
- `firmware/context_node`: the Arduino field-probe sketch and its bench notes.

## Team

- [Henrik Gombos](https://www.linkedin.com/in/henrikgombos), design
- [Anant Gupta](https://www.linkedin.com/in/anant0/), hardware
- [Gabe Meredith](https://www.linkedin.com/in/gabriel-meredith/), machine learning
- [Eddie Hu](https://www.linkedin.com/in/eddie-hu-6ab561270), Ray-Ban glasses and AI interaction
- [Hari Gridharan](https://www.linkedin.com/in/harigridharan1/), product

Built on Meta's Wearables Device Access Toolkit. The public Arduino-parts dataset used for
the detector's warm start is `arduino-lcxdx` on Roboflow Universe, CC BY 4.0.
