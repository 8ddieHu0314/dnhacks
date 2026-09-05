# PPE Sentinel architecture

Worker safety and training on Meta Ray-Ban glasses. The wearer's camera streams to
an iPhone, the iPhone posts frames to a Mac, the Mac runs detection and rules, and
the verdict is spoken back through the glasses. Every alert is recorded so a report
agent can write up the session afterwards and feed a knowledge graph.

This file is the contract. If you change a boundary here, update this file in the
same commit.

## The one picture

```
glasses camera --Bluetooth--> iPhone DAT app --JPEG over Wi-Fi--> server.py POST /frame
                                                                        |
                                                            tee at ingest (core/stream.py)
                                                           /                          \
                                              Mailbox (1 slot, latest wins)      Recorder (ring buffer, never drops)
                                                      |                                   |
                                              Spine.process(newest frame)           clip on violation
                                              detectors -> plugins -> rules                |
                                                      |                                   v
                                              LatestResult + Speaker             session folder on disk
                                                      |                          (events.jsonl, clips/, timeline)
                                              sentence -> iPhone TTS -> glasses          |
                                                                                 report agent at session end
                                                                                          |
                                                                                 knowledge graph
```

Two paths, two clocks, on purpose:

- **Realtime path** answers "what is the wearer's state right now." It uses the
  mailbox, drops stale frames, and must stay under about 100 ms per judged frame.
- **Record path** answers "what happened this session." It never drops. It stores
  the realtime path's verdicts and, around every violation, the raw video.

The realtime path is latency bound, so there is no batching in it. Batching only
belongs in offline passes over saved clips after the session.

## Layers

| Layer | File | Job | Rule of the layer |
|---|---|---|---|
| Detectors | `core/detectors.py` | Frame in, list of `Detection` out | Each detector maps its raw class names to the canonical vocabulary below |
| Plugins | `core/spine.py` (registry), `core/stream.py` (`CachedPlugin`) | Anything that adds detections or attributes to a frame | Slow plugins run on a cadence in a background thread and cache their result |
| Rules | `core/rules.py` | Pure functions over detections and attributes | No model imports, no numpy, no I/O. Must be unit-testable with hand-built inputs |
| Spine | `core/spine.py` | Runs detectors, plugins, rules, debounce for one frame | Holds the per-rule timers across frames |
| Stream | `core/stream.py` | Mailbox, StreamRunner, LatestResult, Speaker, Recorder | Detector-agnostic and rule-agnostic |
| Live session | `core/live.py` | EventTracker (fired-rule start/end), HUD drawing, LiveSession (one Spine plus tracker across streamed frames) | Model-agnostic and Gradio-agnostic. Shared by the Video tab (offline) and the Live tab (webcam) |
| Surfaces | `app.py`, `stream_demo.py`, `server.py` | Gradio playground (Image, Video, Live tabs), local demo, phone endpoint | Thin. No rule logic lives here. `app.py` wires detectors in one function, `build_spine` |

## Canonical vocabulary

Rules only ever match these strings. Detectors translate into them. Add a class here
before any rule references it.

| Canonical | SH17 raw | Roboflow tools raw | Notes |
|---|---|---|---|
| `person` | person | | third person only |
| `head` | head | | |
| `helmet` | helmet | | |
| `face` | face | | |
| `face_mask` | face-mask | | |
| `face_guard` | face-guard | | |
| `safety_glasses` | glasses | | |
| `ear_muffs` | ear-mufs | | SH17 spelling is a typo, keep the mapping |
| `hands` | hands | | SH17 means a BARE hand |
| `gloves` | gloves | | SH17 means a GLOVED hand. There is no `hands` box under a `gloves` box |
| `tool` | tool | every class (wrench, hammer, drill, ...) | Roboflow keeps the specific name in `Detection.subtype` |
| `foot` | foot | | |
| `shoes` | shoes | | |
| `safety_vest` | safety-vest | | |
| `safety_suit` | safety-suit | | |
| `medical_suit` | medical-suit | | |

Current code compares lowercase raw names directly. Introducing the mapping is the
first change to make before adding a third detector, so that two models both saying
"tool" or one saying "hard hat" never reaches a rule.

## Viewpoints: what a rule can and cannot see

The wearer's camera cannot see the wearer's own head, chest, or eyes. Every rule
declares which viewpoint it needs.

| Viewpoint | Examples | Needs a tracker | Status |
|---|---|---|---|
| `self` | bare hands, tool in bare hand, ESD wrist strap, sleeves, shoes | No. All boxes in frame belong to the wearer | Built |
| `other` | coworker without helmet, vest, safety glasses, face mask | Yes. Boxes must be grouped by person so a rule fires once per person | Slot documented in `core/spine.py`, not built |
| `environment` | open floor tile, ladder, cable across walkway, hot aisle zone | Usually no | Needs new models |

Do not promise a wearer-self helmet or vest rule. It is only possible as a mirror check
at session start or from a buddy's glasses.

## How to add things

### A new rule

1. Write a pure function in `core/rules.py` with the signature
   `(detections: list) -> RuleResult`. Return `applicable=False` when the frame cannot
   be judged (the thing to check is not visible). `applicable=False` is reported as
   UNKNOWN, never as PASS.
2. Give it a severity, a one-line `detail`, and append it to `RULES`.
3. Add its sentence to `spoken_sentence` in priority order. Most severe first.
4. Add one photo where it should fire and one where it should pass to `test_images/`,
   and confirm with `.venv/bin/python smoke.py test_images/<file>.jpg`.
5. Confirm the debounce still behaves with `.venv/bin/python stream_demo.py --no-speak`.

Rules never import models, never call the network, and never read the clock. The
spine passes `dt` and owns the timers.

### A new detector (a new model)

1. Add a class in `core/detectors.py` with `detect(frame_bgr) -> list[Detection]`.
   Set `source` to a short unique name. Map raw class names to the canonical
   vocabulary above.
2. Fail soft. Catch exceptions, store the message in `self.last_error`, return `[]`.
   A dead model must not take the alert path down.
3. Run it locally. Hosted inference is for one-image tests only. Measure its latency
   on this Mac with 10 warm calls before wiring it in.
4. If it is slower than about 30 ms, do not add it to the per-frame path. Wrap it as a
   plugin with `CachedPlugin(every_n_frames=N, ttl_seconds=1.0)` in `core/stream.py`
   so it runs on a cadence in the background and its last result is merged in.
5. Register it in `Spine` (primary detector) or via `spine.register_plugin`. There are
   three places that build a Spine, and a new detector goes into each of them once:
   `build_spine` in `app.py` (every playground tab, including the Live webcam tab, goes
   through this one function, so the Live tab needs no extra wiring), the module-level
   setup in `server.py`, and `stream_demo.py`. In `build_spine`, add the plugin behind
   `CachedPlugin` when `cached_tools` is true (the live and server path) and inline
   otherwise (offline passes where latency does not matter).
6. Prove it live before trusting it: open the Live tab in `app.py`, point a webcam at the
   thing the detector should see, and watch the HUD line for its rule and the stats line
   (judge ms and effective fps). If the judge latency climbs past about 100 ms, the
   detector is on the per-frame path and needs to move behind `CachedPlugin`.

### A new attribute plugin (not boxes: pose, zone, depth)

A plugin is a callable `(frame_bgr, state: FrameState) -> None`. It may append to
`state.detections` or write into `state.attributes`. Register with
`spine.register_plugin(fn, every_n_frames=N)`. Rules read attributes by key. Keep
attribute keys namespaced by plugin, for example `state.attributes["pose"]`.

### The tracker (for `other` rules)

Add a plugin that runs ByteTrack over the `person` boxes and stamps `track_id` on
every detection by nearest person. Then `other` rules group detections by `track_id`
before evaluating, and timers are keyed by `(rule, track_id)` instead of `rule`. The
slot is marked in `core/spine.py` right before the rules loop.

### The Roboflow tools model: how it runs and how to refresh it

Roboflow does not offer a `.pt` download for Universe models. What it does offer is a
cache: its `inference` runtime fetches a bundle (`weights.onnx`, `class_names.txt`,
`inference_config.json`) into `/tmp/cache/models-cache/` on first load. We copied that
bundle into `weights/tools_yolov5n.onnx` and `weights/tools_yolov5n.classes.txt` and
run it with plain `onnxruntime` in `ToolsOnnxDetector`. No Roboflow package is
installed in the main venv. The key is not needed at runtime.

To refresh or fetch a different Roboflow model:

```
python3.11 -m venv .venv-inf && uv pip install --python .venv-inf/bin/python inference python-dotenv
.venv-inf/bin/python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); \
  from inference import get_model; get_model(model_id='<workspace-model>/<version>', api_key=os.environ['ROBOFLOW_API_KEY'])"
cp /tmp/cache/models-cache/v2-<model>-*/*/weights.onnx weights/<name>.onnx
cp /tmp/cache/models-cache/v2-<model>-*/*/class_names.txt weights/<name>.classes.txt
```

Then read the bundle's `inference_config.json` for input size, resize mode, and color
order, and check `ToolsOnnxDetector` preprocessing matches. Install the runtime only
in a scratch venv: it pins an older FastAPI and installs a headless OpenCV that shadows
the GUI build and breaks the overlay window.

Weights files are gitignored. `setup.sh` fetches SH17; the tools ONNX must be copied
in by hand or refreshed as above.

### A model for something no public model knows (ESD strap, floor tile)

Label a few hundred frames from real glasses footage in Roboflow, train there,
download or cache the weights, and run them locally like the tools model. Map the new
class into the vocabulary table above.

## Recording and the session record

Everything the report agent needs is written to disk during the session by code, not
by a model. Facts first, narrative later.

### Ring buffer

`Recorder` in `core/stream.py` keeps the last N seconds of raw frames in memory in a
fixed-size deque. New frames overwrite the oldest. Nothing touches disk while the
wearer is compliant.

### Clip on violation

When a rule fires (the debounce passes), the recorder dumps the ring buffer to
`sessions/<session_id>/clips/<rule>_<t>.mp4` and keeps appending frames until the rule
clears plus a tail of a few seconds. Result: full frame rate video of the lead-up,
the violation, and the recovery, for every incident, and nothing else.

`--record-all` writes the whole session as one file for supervised training runs.
It is off by default because it fills a disk quickly.

### Session folder layout

```
sessions/<session_id>/
  session.json        id, worker (may be null), device, zone, start, end, models and versions, thresholds
  events.jsonl        one line per fired rule: rule, severity, start, end, duration_s,
                      tool subtype if any, keyframe path, clip path, detections at fire time
  timeline.jsonl      one line per second: active rules and their timer values, judged fps, dropped count
  keyframes/          one JPEG per event, the frame at the moment the rule fired
  clips/              one MP4 per event
  report.md           written by the report agent at session end
  extraction.json     structured insights from the report agent, kept separate from facts
```

Session boundaries follow the glasses stream: start when the stream starts, end when
it stops or the wearer takes the glasses off.

### Report agent (session end)

Input: `session.json`, `events.jsonl`, the timeline, and the keyframes. Output:
`report.md` (plain English, for the worker and the safety lead, with coaching notes)
and `extraction.json` (things code cannot know: what the worker appeared to be doing,
whether an event looks habitual, suggested training focus). The agent cites event ids
from `events.jsonl` for every claim. Use a vision-capable Claude model so it can see
the keyframes.

### Knowledge graph

Nodes: Worker, Session, Violation, Rule, Tool, PPEItem, Zone, Clip.
Edges: worker HAD session, session CONTAINS violation, violation BREAKS rule,
violation INVOLVES tool, session IN zone, violation EVIDENCED_BY clip.

Facts from `events.jsonl` become nodes and edges directly. Insights from
`extraction.json` go in as a distinct edge type (`INFERRED_...`) so a query can always
separate what was observed from what a model concluded. Start with an embedded graph
persisted to JSON and an interactive web view. Move to Neo4j only if queries outgrow it.

## Latency budget on the realtime path

| Stage | Measured on this Mac |
|---|---|
| SH17 yolov8s, 640 px, Apple GPU | 12 to 30 ms |
| Roboflow tools, hosted (removed from code) | 250 to 2100 ms |
| Roboflow tools, Roboflow's in-process runtime (scratch venv, 203 packages) | 5.9 ms |
| Roboflow tools, plain onnxruntime on the cached ONNX (what ships) | 12.6 ms including resize and NMS |
| Tools plugin on every 5th frame inside the stream | 19 to 24 ms |
| Judge loop median at 30 fps in | 12 ms, 22 of 180 frames dropped |
| Judge loop median at 10 fps in | 22 ms, 0 dropped after warm-up |

The demo overlay draws the last known boxes on every incoming frame, so the screen
runs at feed rate while the judge loop runs at 10 to 15 fps.

The Live tab in `app.py` is a stand-in for the phone stream: the browser's webcam posts
frames to Gradio, which calls the spine once per frame with the real wall-clock `dt`,
the real debounce, and the tools model behind `CachedPlugin` on the same cadence as
`server.py`. Its stats line reports judge latency and the effective fps actually
achieved. Expect that fps to be well below the offline Video tab's judge rate, because
each frame makes a browser round trip. The judge latency number is the one that has to
stay under budget; the fps number tells you how much the browser transport costs.

## Non-goals right now

- Batching in the realtime path.
- Person tracking for the self viewpoint.
- Any rule about the wearer's own head or torso.
- Hosted inference on the per-frame path.
