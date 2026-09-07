# Team plan: three tracks to the demo

State on the evening of Sept 5: glasses -> phone -> Mac streaming works (Bonjour-discovered,
USB attempted then Wi-Fi), Claude analysis streams back to the phone sentence by sentence and
the phone speaks it into the glasses. Real Claude is untested on the glasses (no API key on the
Mac yet); the receiver runs in fake mode until then. A shared-relay fix for interface lag just
landed and needs confirming on the phone.

The three layers meet at fixed interfaces, so they can be worked in parallel:

- Phone -> Mac: `WS /ws/ingest`, binary = 8-byte big-endian capture ms + JPEG; text = JSON
  commands `{"type":"inspect"}`, `{"type":"narrate","enabled":bool,"interval":s}`,
  `{"type":"voice",...}`, `{"type":"detector",...}`, `{"type":"report_page",...}`, and since Sept 6
  `{"type":"ask","text":...,"heard_at":ms}` (a sentence the wearer said, recognized on the phone)
  and `{"type":"hush"}` (stop talking, drop queued sentences); since Sept 6 afternoon
  `{"type":"session","id":...}` (one id per app launch, sent with the preferences; a new id resets
  the Mac's conversation memory and any guided build, a repeat keeps them), and
  `{"type":"demo","enabled":bool}` (Demo mode: the breadboard build in Claude's prompt; the Mac
  echoes it as `demo_mode` in its status), and `{"type":"mac_speak","enabled":bool}` (read every
  sentence aloud on the Mac's speaker too; echoed as `mac_speak`). On the reactive branch the Mac's
  hello and status message is `{"type":"status",...}`; `reactive`, `narrate` and `models` are
  gone. The full list is in CLAUDE.md.
- Mac -> phone (same socket): `{"type":"speak","text":...}` per sentence, then
  `{"type":"speak_end"}`; `{"type":"narration","enabled":..,"interval":..}`; `{"type":"reactive",...}`
  status; ElevenLabs PCM as `audio` / `audio_end`; `speak_stop` after a hush.
- Mac HTTP: `POST /reset` (new conversation), `POST /demo {enabled}`, `POST /mac_speak {enabled}`, `POST /inspect {question}`, `POST /ask {text}`, `POST/GET /narrate`, `GET /report`, `GET /health`,
  dashboard on `/`. Report rows land in `services/relay_receiver/report.jsonl`.

Change an interface only with the other side's owner in the loop, and update this file.

## Track A: phone + glasses (needs the iPhone, the glasses, Xcode) — owner: Eddie

Folder: `clients/ios/GlassesInspector`. Branch: `track-phone`.

1. Confirm the lag fix: app smooth at rest and at Low; note where it returns (Medium/High).
2. Voice: "Test voice on glasses" plays through the glasses, not the phone. If it plays on the
   phone, the audio session category is the fix (Speaker.swift).
3. Describe loop in fake mode: sentences spoken through the glasses, video keeps flowing.
4. Demo defaults: resolution, relay cap, speak on, narration interval. Bake them in.
5. Robustness: app relaunch reconnects by itself; Preview survives a 2-minute run; caption and
   Hush button behave. Kill the toolkit's "compat undefined" chip noise if it distracts.
6. Nice-to-have: voice trigger ("describe") via the phone mic; a huge one-tap Describe.
7. Record a 60-second backup video of the working loop from the dashboard + phone.

Done when: a cold start to spoken description takes under 30 s of taps and works three times
in a row on venue Wi-Fi.

## Track B: the Mac brain (Python, needs an Anthropic key) — owner: teammate 2

Folder: `services/relay_receiver`. Branch: `track-brain`. Start: `./run.sh` (put the key in
`services/relay_receiver/.env`; `INSPECT_FAKE=1` to work without one).

1. Turn on real Claude; check latency to first sentence and total. Downscale the frame sent
   to Claude (max 768 px long side) and trim `max_tokens` if it is slow.
2. Prompt tuning for the three demo objects (see Track C): hazards first, read labels and
   gauges verbatim, one next action, under 60 words, no preamble. Keep it in `SYSTEM_PROMPT`
   and `NARRATION_PROMPT` in `app.py`.
3. Narration: make "no change" reliable (previous text as context is already passed); add a
   minimum gap after speech ends so it never talks over itself.
4. Report: `GET /report` already stores frame + text per inspection. Add a printable
   session report page (`/report.html`): timestamped frames, spoken text, flagged hazards,
   downloadable JSON. This is what judges see after the live demo.
5. Optional, only if Track C decides to use it: tee frames to Gabe's PPE server
   (`feat/gabe-eddie-merged-work`, same `/ws/ingest` wire format) and feed its detections
   into the Claude prompt as structured context.

Done when: Inspect returns a spoken, correct description of each demo object in under 6 s,
and the report page shows the session.

## Track C: demo, story, integration (no special hardware) — owner: teammate 3

Folders: `docs/`, repo root. Branch: `track-demo`.

1. Pick the three demo objects and stage them: e.g. a breaker panel or extension cord
   (hazard), a printed gauge or nameplate (reading), a fire extinguisher or PPE item
   (compliance). Photograph them so Track B can tune the prompt tonight.
2. Write the 3-minute demo script: cold start, Describe on each object, one continuous
   narration walk, show the report page. Time it with a stopwatch.
3. Pitch: the track is Energy & Industrialization; the story is hands-free inspection with
   hazards spoken into the wearer's ear and a report written for them. Two slides max.
4. Architecture diagram + README for judges (one picture: glasses -> phone -> Mac -> Claude
   -> glasses). Reuse `docs/meta-glasses-field-notes.md` for the "what we learned" beat.
5. Fallback plan: the backup video from Track A, and a laptop-webcam mode if the glasses
   misbehave on stage (the dashboard accepts any JPEG on `POST /frame`).
6. Repo hygiene: decide whether Gabe's PPE branch is part of the demo; if yes, agree the
   split with Track B and merge; if no, say so in the README.

Done when: a full dry run has been done twice, with a timer, on the venue Wi-Fi.

## Working agreements

- Each track works on its own branch off `main`, small commits,
  rebase on it before merging. Only the owner touches their folder.
- Interface changes are announced in the group chat and recorded above.
- Branch history: the glasses work lived on `meta-glasses-display-access` until 2026-09-06, when `main` was fast-forwarded to it (the previous `main` is archived as `anant/hardware-archive`).

Integration checkpoints: one tonight after Track A confirms voice on glasses and Track B
  has real Claude; one dry run in the morning; one on the demo network.
- Ask each other before restarting the receiver on the demo Mac; it holds the phone's socket.
