"""Guided build: the visitor demo. The relay keeps which step the wearer is on; Claude judges
the frames against that step and speaks only the next instruction, never the whole list.

One procedure exists: on the one breadboard on the bench, three wires make a fan motor run
while a tactile button is held, then a test. The human-facing design doc (circuit, the team's
setup checklist, design note) is docs/procedures/button-fan.md; the spoken lines below are the
ones the glasses say, keep the two in step.

PROMPT is static and goes in the cached system block. turn_note() is the dynamic part (which
step, the verdict protocol) and goes in the per-question user message, so the catalog cache
stays byte-identical. The reply's first word is the verdict (DONE or STAY); app.py strips it
before speaking and calls advance() on DONE.
"""
import re
import time

NAME = "button-fan"

STEPS = [
    {"title": "motor to ground",
     "say": "Push the motor's black wire into the blue minus row of the top rail, above the printed twelve, between the ten and the fifteen.",
     "look_for": "a black lead ending in a hole of the top rail's lower row, the blue line, between the printed 10 and 15; the fan still",
     "hints": ["Lead in the red row: That is the red plus row. The black wire goes in the blue row just under it, on the same rail.",
               "Lead in the bottom rail or in the board: That is the wrong rail. Use the top one, the rail next to the button, blue row."]},
    {"title": "motor to the button",
     "say": "Push the motor's red wire into hole a twelve: row a is the top row of holes, and column twelve is two holes from the printed ten toward the fifteen.",
     "look_for": "the red lead ending in row a, column 12, in line with the button's far pair of legs; the fan still (it cannot spin yet, strip 10 is not on the plus rail until wire three)",
     "hints": ["Lead in column 10 or 11: One column over. Count from the printed ten: two holes toward the fifteen is column twelve.",
               "Lead in row b, c or d of column 12: that works too, the whole column is one strip; count it DONE and say so.",
               "Lead below the groove, rows f to j: That is below the groove, a different strip. Use the top half, row a."]},
    {"title": "plus rail to the button",
     "say": "Take the red jumper. Push one end into hole a ten, the top hole under the printed ten, first. Then push the other end into the red plus row of the top rail above it.",
     "look_for": "a red jumper from a10 to the top rail's upper row, the red line, above column 9 to 11; the fan still",
     "hints": ["Rail end went in first: Do the board end first. A wire hanging off the plus row can touch the blue row and short the module.",
               "Rail end in the blue row: That end is in the blue row. Move it up one row to the red plus row.",
               "Board end in column 11 or 12: That is the motor's column. The jumper goes under the printed ten, column ten.",
               "Fan spinning with nothing pressed: The fan runs with nothing pressed, so the two board wires are on the same side of the switch. Check that the jumper is in column ten and the motor's red wire in column twelve; if they are, the button is turned the wrong way, so pull the jumper out and turn the button a quarter turn."]},
    {"title": "test",
     "say": "Wiring done. Press and hold the button.",
     "look_for": "the fan spinning while the button is held, still on release",
     "hints": ["Nothing happens: one check per exchange, in this order: is the small light on the module lit; are both rail wires on the top rail, black in blue and red in red; is the jumper in column ten and the motor's red wire in column twelve, both in the top half; press each wire deeper into its hole; try a fresh battery.",
               "Fan weak or twitching: The module's top jumper is probably on three point three volts, or the battery is low.",
               "Fan blowing the wrong way: Swap the motor's two wires. The motor has no fixed polarity."]},
]
INTRO = "I will guide you through three wires, one at a time, then we test it."
FINISH = "That is it. Your button switches the fan."
TIMEOUT_SECONDS = 15 * 60   # no build talk for this long and the guide lapses
MEMORY_SECONDS = 10 * 60    # earlier exchanges kept in the user message while guiding

PROMPT = """Guided build (the demo). One procedure exists: on the one breadboard on the bench, three wires make a fan motor run while a tactile button is held. Before any build, the wearer is simply looking around the bench: they pick up parts, ask what something is, how many legs it has, what voltage it takes, and you answer those as the ordinary conversation they are. Do not bring up the build, do not suggest wiring anything, and do not ask whether they want to build; the build only begins when the wearer says they want to build, wire or make the circuit, or asks how to get the fan working, and the relay then attaches a bracketed "Guided build" note to the user message. You are the guide only while that note is present: the relay decides when a build starts and which step is current, and it strips the verdict word from your reply before anything is spoken. Without the note, never start guiding on your own and never begin a reply with the words DONE or STAY, because they would be spoken aloud. With the note: one step per reply, never the whole list; judge the frames against the step named there and begin your reply with the verdict word the note asks for. While guiding, the step's instruction is the answer: say it exactly as written, with at most one short sentence before it; questions about parts during the build are still answered as usual.

The circuit: top red rail (five volts) to hole a10, the button between strip 10 and strip 12, hole a12 to the motor's red lead, the motor's black lead to the top blue rail (ground). The power module clipped to the right end of the board already feeds the top rail, and the rails stay live for the whole build, which is why the order of the wires matters: ground first, then the motor to the open switch, the plus rail last and board end first. Nothing can spin or short until the last wire is home.

Board conventions: column 1 is at the module end and column 60 at the far end. Rows a to e are the top half next to the top rail, then the groove, then rows f to j. The five holes of one column in one half are one strip. The printed 10 sits at column 10; column 12 is two holes from it toward the printed 15. The rail rows run along the board in groups of five holes; the upper row of the top rail is red (plus), the lower row blue (minus); "near column ten" means the rail hole above column ten. The button sits in the top half with one pair of legs in strip 10 and the other pair in strip 12, legs in rows c to e, so row a is free. Name holes by row letter and column number anchored to the printed numbers, never by left or right, because the wearer may face the board from either side.

The layout when finished:
   col:   13  12  11  10   9
 red  +    .   .   .   .   .    five volts from the module
 blue -    .   M-  .   .   .    M- motor black lead (wire one)
 row a     .   M+  .   W   .    M+ motor red lead (wire two), W red jumper (wire three)
 row c     .   B   .   B   .    button legs, two in strip 10, two in strip 12
 row e     .   B   .   B   .
   ------------- groove -------------

The steps, in order:
""" + "\n".join(
    f"{i}. {s['title'].capitalize()}. Say: \"{s['say']}\" Done when: {s['look_for']}. If wrong: " + " ".join(s["hints"])
    for i, s in enumerate(STEPS, 1)
) + f"""
Opening line when the build starts: "{INTRO}" Closing line after the test: "{FINISH}"

Judging the frames: a wire in the bottom rail or below the groove is wrong. When the frames do not settle which column a wire is in, say STAY and ask the wearer to hold still over the board or to say which printed number the wire is nearest; do not guess a column. When the wearer says they did it and the frames do not contradict them, count the step done. Questions about the parts themselves during the build (what is this, how many legs, what voltage) are answered from the catalog as usual, and the step does not advance."""

# Explicit build intent only. Looking at parts and asking about them is ordinary conversation and
# must never start the build, so generic phrases ("what should I do", "help me", "next step") are
# deliberately not triggers.
START = re.compile(
    r"\b(get|make|have|want|let'?s get|let'?s make|how (do|can|would) i (get|make)|how to (get|make))\b.{0,24}\b(fan|motor)\b.{0,24}\b(work|working|spin|spinning|run|running|turn on|go|going|start|started)\b"
    r"|\bturn (on )?the (fan|motor)\b"
    r"|\bput (the |this |my )?(breadboard|circuit|fan|motor|project) together\b"
    r"|\b(build|wire|wiring|make|assemble|set up|hook up|connect|finish|complete|put together)\b.{0,40}\b(fan|motor|circuit|breadboard|board|project)\b"
    r"|\b(start|begin|do|try|walk me through|talk me through|guide me through|help me with|help me) (the |a |this |my )?(build|circuit|project|wiring|breadboard)\b"
    r"|\b(want|like|ready|time) to (build|wire|make|assemble|start building|start wiring)\b",
    re.I)
RESTART = re.compile(r"\b(start over|start again|from the beginning|restart|reset)\b", re.I)
QUIT = re.compile(r"\b(quit|exit|leave|end|forget|never ?mind|abandon|skip)\b.{0,24}\b(build|guide|guiding|fan|circuit|steps?|procedure)\b|\b(quit|exit)\b", re.I)

state = {"step": 0, "since": 0.0, "starts": 0, "finished": 0}


def active() -> bool:
    return state["step"] > 0 and time.time() - state["since"] < TIMEOUT_SECONDS


def start():
    state.update(step=1, since=time.time(), starts=state["starts"] + 1)


def stop():
    state["step"] = 0


def set_step(n: int):
    """Demo desk control: jump to a step (1 to len(STEPS)); 0 ends the build."""
    n = max(0, min(int(n), len(STEPS)))
    state.update(step=n, since=time.time())


def advance() -> bool:
    """The current step is done. Returns True when that was the last one (the build is finished)."""
    if not active():
        return False
    state["since"] = time.time()
    if state["step"] >= len(STEPS):
        state["step"] = 0
        state["finished"] += 1
        return True
    state["step"] += 1
    return False


def route(low: str) -> str | None:
    """Update the guide from the wearer's words (lowercased, punctuation stripped). Returns
    "start", "restart" or "quit" when those happened, else None. Advancing is Claude's call."""
    if active():
        if QUIT.search(low):
            stop()
            return "quit"
        if RESTART.search(low):
            start()
            return "restart"
        state["since"] = time.time()
        return None
    if START.search(low):
        start()
        return "start"
    return None


def turn_note(kind: str | None) -> str:
    """The dynamic part of the user message while a build is in progress."""
    i = state["step"]
    s = STEPS[i - 1]
    nxt = STEPS[i] if i < len(STEPS) else None
    if kind in ("start", "restart"):
        return (f"[Guided build {'restarting' if kind == 'restart' else 'starting'}: step 1 of {len(STEPS)}. "
                f"Begin your reply with the word STAY, then say: \"{INTRO}\" and then step 1 exactly: \"{s['say']}\"]")
    then = (f"On DONE: confirm in a few words, then say step {i + 1} exactly: \"{nxt['say']}\"" if nxt
            else f"On DONE: say exactly: \"{FINISH}\"")
    return "\n".join([
        f"[Guided build in progress: step {i} of {len(STEPS)}, {s['title']}. The instruction they were given: \"{s['say']}\"",
        f"Look for in the frames: {s['look_for']}.",
        "If wrong: " + " ".join(s["hints"]),
        "Begin your reply with exactly one word: DONE if the frames show this step complete, or if the frames cannot settle it and the wearer says they did it; otherwise STAY. Then speak.",
        then,
        "On STAY: say what to fix in one or two sentences, using the hint that matches what you see. If the wearer asked something unrelated to the build, answer that instead and still begin with STAY.]",
    ])


def info() -> dict:
    a = active()
    i = state["step"]
    return {"name": NAME, "active": a, "step": i if a else 0, "steps": len(STEPS),
            "title": STEPS[i - 1]["title"] if a else None,
            "idle_s": round(time.time() - state["since"]) if a else None,
            "starts": state["starts"], "finished": state["finished"]}
