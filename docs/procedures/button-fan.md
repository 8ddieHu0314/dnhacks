# Demo: press the button, the fan spins

The guided build visitors try on the glasses. The bench starts with the breadboard, the power
module clipped to its right end with the top rail live (red row five volts, blue row ground),
one tactile button in the top half of the board, the fan motor loose, and one red jumper wire.
The visitor makes three connections and the fan runs while the button is held. No Arduino,
no code.

The three wiring steps are what has to go right. Anything before them (the visitor holding up
the button or the fan and asking what it is, how many legs, what voltage) is the ordinary ask
path: the inspector matches the part to its catalog record and answers from it, and never
brings up the build on its own. The build starts only when the visitor says they want to build
or wire the circuit, or asks how to get the fan working, so nobody feels pushed into it. The
test at the end is a formality.

The inspector speaks one step at a time, judges the frames, and only advances when the step is
done. Spoken lines follow the voice rules: one or two sentences, numbers in words, no lists.
Every hole is named by row letter and column number and anchored to the printed numbers, never
by left or right, so the instruction holds whichever way the visitor faces the board.

Parts, by catalog id: `psu-mb102`, `button`, `dc-motor-fan`, `battery-clip-9v`.

The relay runs this only while Demo mode is on (the phone's gear menu, Inspector section, off by
default); off, nothing about the breadboard reaches Claude. It runs it from
`services/relay_receiver/guide.py`: the spoken lines there are what
the glasses say, so a change to a "Say" line here must be made there too. The relay keeps the
step, Claude judges the frames and answers with a verdict word (DONE or STAY) that is stripped
before speaking; `GET /guide` shows where the visitor is and `POST /guide {"reset": true}`
clears it between visitors.

## The circuit

```
top red rail (+5 V) --- a10 ---[ button ]--- a12 --- motor red lead
                                                      motor black lead --- top blue rail (GND)
```

Board conventions: column 1 at the module end, column 60 at the far end; rows a to e in the top
half next to the top rail, the groove, then rows f to j. Each column's five holes in one half are
one strip. The printed "10" sits at column 10; column 12 is two holes from it toward the printed
"15". The rail rows run along the board in groups of five holes; "near column ten" means a rail
hole above column ten.

```
   col:   13  12  11  10   9
 red  +    .   .   .   .   .    five volts from the module
 blue -    .   M-  .   .   .    M- = motor black lead (wire 1)
 row a     .   M+  .   W   .    M+ = motor red lead (wire 2), W = red jumper (wire 3)
 row b     .   .   .   .   .
 row c     .   B   .   B   .    B = button legs, two in strip 10, two in strip 12
 row d     .   .   .   .   .
 row e     .   B   .   B   .
 ------------- groove -------------
```

Strip 10 is one side of the button, strip 12 the other. Pressing joins them, so current runs
red rail, W, strip 10, button, strip 12, M+, motor, M-, blue rail. The rails stay live for the
whole build; the order of the three wires (ground, then motor to the open switch, then the plus
rail last, board end first) means nothing can spin or short until the last wire is home.

## Setup by the team before each visitor

- Module on, top jumper on 5V, the module light lit. Confirm with a multimeter: about five volts
  between the red and blue rows of the top rail near column 10.
- Button in the top half with one pair of legs in column 10 and the other pair in column 12, legs
  in rows c to e so row a is free. The two legs on one edge of the button are one side of the
  switch (catalog record `button`), so the leg edges must face the module and the far end of the
  board, not the rails. Confirm with continuity: a10 to a12 beeps only while pressed. If it beeps
  unpressed, turn the button a quarter turn and re-test.
- Both motor leads end in male pins. Bare stranded wire will not hold in a hole.
- One red jumper wire on the bench. Nothing else in strips 10 to 12 or the top rail near them.
- Board placed with the module on the visitor's right, as in the reference photo, so the printed
  numbers read the way the frames expect.

## Wire 1 of 3: motor to ground

Say: "Push the motor's black wire into the blue minus row of the top rail, above the printed
twelve, between the ten and the fifteen."

Look for: a black lead ending in a hole of the top rail's lower row (the blue line), somewhere
between the printed 10 and 15. Fan still.

If the lead is in the red row: "That is the red plus row. The black wire goes in the blue row
just under it, on the same rail."

If the lead is in the bottom rail or on the board: "That is the wrong rail. Use the top one,
the rail next to the button, blue row."

## Wire 2 of 3: motor to the button

Say: "Push the motor's red wire into hole a twelve: row a is the top row of holes, and column
twelve is two holes from the printed ten toward the fifteen."

Look for: the red lead ending in row a, column 12, in line with the button's far pair of legs.
Fan still. (It cannot spin yet: strip 10 is not connected to the plus rail until wire 3.)

If it is in column 10 or 11: "One column over. Count from the printed ten: two holes toward
the fifteen is column twelve."

If it is in row b or lower, still column 12: fine, any free hole in strip 12 works; say "That
works too, that whole column is one strip."

If it is in the bottom half (rows f to j): "That is below the groove, a different strip. Use
the top half, row a."

## Wire 3 of 3: plus rail to the button

Say: "Take the red jumper. Push one end into hole a ten, the top hole under the printed ten,
first. Then push the other end into the red plus row of the top rail above it."

Look for: a red jumper from a10 to the top rail's upper row (the red line), above column 9 to
11. Fan still.

If the rail end went in first: "Do the board end first. A wire hanging off the plus row can
touch the blue row and short the module."

If the rail end is in the blue row: "That end is in the blue row. Move it up one row to the
red plus row."

If the board end is in column 12 or 11: "That is the motor's column. The jumper goes under the
printed ten, column ten."

If the fan starts spinning now: "The fan runs with nothing pressed, so the two board wires are
on the same side of the switch. Check that the jumper is in column ten and the motor's red wire
in column twelve; if they are, the button is turned the wrong way, so pull the jumper out and
turn the button a quarter turn."

## Test

Say: "Wiring done. Press and hold the button."

Look for: the fan spinning while the button is held, still on release. Say: "That is it. Your
button switches the fan."

If nothing happens, one check per exchange, in order:

1. "Is the small light on the module lit?" (module off, or the barrel plug loose in the jack)
2. "Both rail wires on the top rail, black in blue, red in red?"
3. "Jumper in column ten, motor's red wire in column twelve, both in the top half?"
4. "Press each wire deeper into its hole." (a loose lead)
5. "Try a fresh battery." (a tired nine volt battery sags under the motor)

Fan weak or twitching: the top jumper on 3.3V, or a low battery. Fan blowing the wrong way:
swap the motor's two wires; the motor has no fixed polarity.

## Design note for the demo desk

The tactile button is a signal switch. Its contacts are rated for tens of milliamps, and the fan
draws about a hundred to two hundred and fifty milliamps running and more at start. For a
short demo the button holds up; the contacts just wear faster. The textbook version puts a
transistor (PN2222 with a flyback diode) or the kit's L293D between the button and the motor,
which is also what the catalog's motor record recommends. The module's regulator drops nine
volts to five and gets warm if the fan runs for minutes, so keep presses short.
