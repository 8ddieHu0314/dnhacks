# Touch Sensor Demo
## Goal

Use the Arduino Uno and included kit parts to demonstrate a **USB-powered,
non-contact touch/proximity signal**. It was intended as a context input for
the worker-assist prototype, not as a voltage detector or safety decision tool.

## Intended circuit
The breadboard circuit uses two PN2222 transistors as a Darlington amplifier:

```text
free antenna wire -- 100 kOhm -- Q1 base
Q1 base -- 1 MOhm -- GND
Q1 emitter -- Q2 base
Q1 collector + Q2 collector -- Arduino A0
A0 -- 10 kOhm -- Arduino 5V
Q2 emitter -- Arduino GND
```

Touching or approaching the free antenna wire changes the amplifier state.
Because the signal is inverted, a lower `A0` reading represents stronger
coupling. The sketch reports `capacitive_level` and `signal_detected` as JSON
at 115200 baud.

## What worked

- Idle readings were typically near `1023`.
- Touch samples dropped to values including `706`, `252`, and `140`.
- A threshold of `850` therefore identifies the observed touch response.
- A USB-only 5V pull-up check read near `118`, confirming the amplifier stage
  and analog-input path responded.

## Current firmware

`context_node.ino` was simplified to emit only continuous `A0` diagnostics
every 500 ms. This removes unrelated tilt, distance, and buzzer inputs while
testing the antenna behavior.

## Current hardware blocker

The Uno is visible to macOS as `/dev/cu.usbmodem101`, but it will not accept an
upload or complete an RX/TX loopback test with pins 0 and 1 connected while the
main chip is held in reset. The touch circuit is not connected during this
test. A working or borrowed Uno is the fastest path to resume the demo.

## Important boundary

This is an interaction/context demo only. It must not be used to detect,
measure, or declare anything about electrical voltage or de-energized equipment.
