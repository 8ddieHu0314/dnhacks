## Fuzzy retrieval: does the right record come back?

| class | mode | n | hit@1 | hit@3 | avg ms |
|---|---|---|---|---|---|
| vision | json | 30 | 0/30 (0%) | 0/30 (0%) | 0.1 |
| vision | bm25 | 30 | 24/30 (80%) | 28/30 (93%) | 0.2 |
| vision | embed | 30 | 11/30 (37%) | 16/30 (53%) | 4.4 |
| vision | hybrid | 30 | 18/30 (60%) | 24/30 (80%) | 4.5 |
| question | json | 10 | 0/10 (0%) | 0/10 (0%) | 0.1 |
| question | bm25 | 10 | 7/10 (70%) | 8/10 (80%) | 0.2 |
| question | embed | 10 | 9/10 (90%) | 9/10 (90%) | 4.1 |
| question | hybrid | 10 | 8/10 (80%) | 9/10 (90%) | 4.2 |

## Exact field lookups: JSON filter vs RAG top-k

| query | truth (JSON filter) | bm25 recall | embed recall | hybrid recall |
|---|---|---|---|---|
| which parts have 16 pins | 3 exact | 1/3 | 1/3 | 2/3 |
| parts that talk over I2C | 3 exact | 3/3 | 1/3 | 2/3 |
| everything in the most complete kit only | 12 exact | 4/12 | 4/12 | 4/12 |

## Embedding misses (top 3 shown)

- [vision] "blue cube with 5 pins" wanted ['relay-5v'], got ['led-rgb', '74hc595', 'psu-mb102']
- [vision] "blue board with two silver cylinders that look like eyes" wanted ['hc-sr04'], got ['lcd1602', 'proto-shield', '7seg-4digit']
- [vision] "tiny blue box with a white grid pattern and three pins" wanted ['dht11'], got ['proto-shield', '7seg-1digit', 'keypad-4x4']
- [vision] "green board with a white plastic socket, four small LEDs and a black chip" wanted ['uln2003-driver'], got ['led-rgb', '7seg-1digit', 'proto-shield']
- [vision] "little blue plastic motor with a white plastic arm and brown red orange wires" wanted ['servo-sg90'], got ['dc-motor-fan', 'stepper-28byj48', 'l293d']
- [vision] "black round disc with a hole in the middle, two legs, sticker on top" wanted ['buzzer-active', 'buzzer-passive'], got ['tilt-switch', 'button', 'proto-shield']
- [vision] "black square base with a round dome on top like a thumb stick" wanted ['joystick'], got ['button', 'tilt-switch', 'servo-sg90']
- [vision] "small black cylinder with a silver stripe at one end and two wires" wanted ['diode-1n4007'], got ['tilt-switch', 'dupont-fm', '7seg-1digit']
- [vision] "flat round disc with a zigzag red line on top and two legs" wanted ['photoresistor'], got ['tilt-switch', 'proto-shield', 'joystick']
- [vision] "small black half cylinder with three legs and a flat face" wanted ['pn2222'], got ['tilt-switch', 'dc-motor-fan', 'button']
- [vision] "blue board with a black chip, a pin header and a big round coin battery" wanted ['rtc-ds3231'], got ['psu-mb102', 'breadboard-830', 'proto-shield']
- [vision] "small board with a tiny chip and 8 pins labeled SCL SDA XDA XCL" wanted ['mpu6050'], got ['breadboard-830', 'proto-shield', 'rfid-rc522']
- [vision] "white plastic dome on a green board with two orange dials" wanted ['pir-hc-sr501'], got ['button', 'breadboard-830', '7seg-1digit']
- [vision] "tan cylinder with colored stripes and a wire out of each end" wanted ['resistor-assorted'], got ['dupont-fm', 'jumper-wire', 'led-rgb']
- [question] "what do I use to make a motor spin in both directions" wanted ['l293d'], got ['dc-motor-fan', 'stepper-28byj48', 'joystick']

## BM25 misses

- [vision] "tiny blue box with a white grid pattern and three pins" wanted ['dht11'], got ['proto-shield', 'led-rgb', 'breadboard-830']
- [vision] "black round disc with a hole in the middle, two legs, sticker on top" wanted ['buzzer-active', 'buzzer-passive'], got ['thermistor', 'button', '7seg-1digit']
- [question] "what do I use to make a motor spin in both directions" wanted ['l293d'], got ['diode-1n4007', 'uln2003-driver', 'dc-motor-fan']
- [question] "which one beeps as soon as it gets power without any code" wanted ['buzzer-active'], got ['relay-5v', 'pn2222', 'dupont-fm']

## Hybrid misses

- [vision] "tiny blue box with a white grid pattern and three pins" wanted ['dht11'], got ['proto-shield', 'lcd1602', 'led-rgb']
- [vision] "green board with a white plastic socket, four small LEDs and a black chip" wanted ['uln2003-driver'], got ['7seg-4digit', '7seg-1digit', 'led-5mm']
- [vision] "black round disc with a hole in the middle, two legs, sticker on top" wanted ['buzzer-active', 'buzzer-passive'], got ['button', '7seg-1digit', 'breadboard-830']
- [vision] "small black cylinder with a silver stripe at one end and two wires" wanted ['diode-1n4007'], got ['tilt-switch', 'pn2222', 'battery-clip-9v']
- [vision] "flat round disc with a zigzag red line on top and two legs" wanted ['photoresistor'], got ['breadboard-830', 'cap-ceramic-104', 'proto-shield']
- [vision] "white plastic dome on a green board with two orange dials" wanted ['pir-hc-sr501'], got ['led-5mm', 'led-rgb', 'breadboard-830']
- [question] "what do I use to make a motor spin in both directions" wanted ['l293d'], got ['dc-motor-fan', 'diode-1n4007', 'uln2003-driver']
