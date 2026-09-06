# ELEGOO UNO R3 Super Starter Kit, inventory seed

Source: photo of the kit lid (35 line items). Kit SKU is most likely EL-KIT-003, confirm on the box.
Whole kit retails around 40 USD (Amazon, 2026). Prices below are approximate single-unit
retail in USD and must be re-checked by the extraction pass. Pin counts are for the part as
shipped in this kit (module vs bare component matters).

| # | Item on lid | Likely MPN / generic | Qty | Pins | Function (plain English) | Key spec | ~Unit price |
|---|---|---|---|---|---|---|---|
| 1 | UNO R3 Controller Board | ELEGOO UNO R3 (ATmega328P + CH340 USB) | 1 | 14 digital I/O (6 PWM), 6 analog in, ~32 header pins | The brain. Runs your code, talks to every other part | 5V logic, 16 MHz, 32 KB flash | 15 |
| 2 | LCD1602 Module (pin header) | 1602A, HD44780 controller | 1 | 16 | Shows 2 lines of 16 characters of text | 5V, parallel 4/8-bit | 4 |
| 3 | Prototype Expansion Module | UNO proto shield + mini breadboard | 1 | Stacks on all UNO headers | Sits on top of the UNO so you can build circuits without loose wires | 5V | 4 |
| 4 | Power Supply Module | MB102 breadboard PSU | 1 | 8 output pins (2 rails x 4) | Turns 6.5 to 12V (barrel or USB) into clean 3.3V or 5V for the breadboard | 3.3V/5V selectable, ~700 mA | 3 |
| 5 | ULN2003 Stepper Motor Driver Module | ULN2003 (16-pin DIP on a board) | 1 | 4 inputs + 2 power + 5-pin motor socket | Amplifies weak UNO signals so they can drive the stepper motor coils | 500 mA/channel | 2 |
| 6 | Stepper Motor | 28BYJ-48 | 1 | 5 wires | Motor that turns in precise steps, not continuously | 5V, 64:1 gearbox, 2048 steps/rev | 3 |
| 7 | Servo Motor SG90 | SG90 | 1 | 3 wires (signal, V+, GND) | Motor that swings to a chosen angle | 4.8 to 6V, 0 to 180 deg, 1.8 kg·cm | 3 |
| 8 | 5V Relay | SRD-05VDC-SL-C | 1 | 5 (2 coil, COM, NO, NC) | Electrically controlled switch, lets 5V control a big load | 10 A at 250 VAC | 1 |
| 9 | IR Receiver Module | VS1838B on breakout | 1 | 3 (OUT, GND, VCC) | Hears the remote control's infrared blinks | 38 kHz | 1 |
| 10 | Joystick Module | KY-023 | 1 | 5 (GND, +5V, VRx, VRy, SW) | Thumbstick, two knobs plus a push button | 2 x 10K pots | 2 |
| 11 | DHT11 Temperature and Humidity Module | DHT11 (3-pin module) | 1 | 3 (VCC, DATA, GND) | Measures air temperature and humidity | 0 to 50 C, 20 to 90 %RH, 1-wire protocol | 2 |
| 12 | Ultrasonic Sensor | HC-SR04 | 1 | 4 (VCC, Trig, Echo, GND) | Measures distance with sound, like a bat | 2 to 400 cm | 2 |
| 13 | Fan Blade and 3-6V Motor | brushed DC motor + blade | 1 | 2 terminals | Plain spinning motor | 3 to 6V | 2 |
| 14 | Active Buzzer | 5V active buzzer | 1 | 2 (polarized) | Beeps as soon as it gets power | 5V, fixed tone | 0.5 |
| 15 | Passive Buzzer | passive piezo | 1 | 2 | Only makes sound when you feed it a tone signal | needs PWM | 0.5 |
| 16 | 74HC595 IC | SN74HC595N | 1 | 16 (DIP) | Shift register, turns 3 UNO pins into 8 outputs | 2 to 6V | 0.4 |
| 17 | L293D IC | L293D | 1 | 16 (DIP) | Dual H-bridge, lets the UNO spin motors both directions | 600 mA/channel, 4.5 to 36V | 1.5 |
| 18 | Button | 6x6 mm tactile switch | 5 | 4 | Push button | momentary | 0.1 |
| 19 | Potentiometer 10K | 10K linear pot | 1 | 3 | Twist knob that gives a variable voltage | 10 kΩ | 0.5 |
| 20 | 1 Digit 7-Segment Display | 5161AS (common cathode) | 1 | 10 | Shows one number 0 to 9 | red | 0.5 |
| 21 | 4 Digit 7-Segment Display | 3461BS (common anode) | 1 | 12 | Shows four numbers | multiplexed | 1 |
| 22 | Tilt Ball Switch | SW-520D | 1 | 2 | Closes when tipped, a crude orientation sensor | | 0.3 |
| 23 | Remote Control | 21-key NEC IR remote | 1 | none | Sends infrared codes to the IR receiver | CR2025 battery | 2 |
| 24 | 830 Tie-Points Breadboard | MB-102 | 1 | 830 holes | Plug-in board for building circuits without solder | | 4 |
| 25 | USB Cable | USB A to B | 1 | | Powers and programs the UNO from a computer | | 2 |
| 26 | Female-to-Male Dupont Wire | 20 cm Dupont | 10 | | Wires from a module's pins to the breadboard | | 1 |
| 27 | Breadboard Jumper Wire | solid core jumpers | 65 | | Short wires for the breadboard | | 2.5 |
| 28 | 9V Battery with Snap-on Connector Clip | 9V snap to barrel jack | 1 | 2 wires | Runs the UNO without a computer | 9V | 1 |
| 29 | Resistor | 1/4 W 5% assorted | 120 | 2 | Limits current, e.g. protects an LED | 10 Ω to 1 MΩ, colour bands | 2.5 total |
| 30 | LED | 5 mm assorted | 25 | 2 (long leg = +) | Little light | ~20 mA | 0.1 |
| 31 | RGB LED | 5 mm common cathode | 2 | 4 | One LED that can be any colour | | 0.3 |
| 32 | Thermistor | 10K NTC | 1 | 2 | Resistor that changes with temperature | 10 kΩ at 25 C | 0.3 |
| 33 | Diode Rectifier | 1N4007 | 2 | 2 (stripe = cathode) | One-way valve for current | 1 A, 1000 V | 0.05 |
| 34 | Photoresistor (Photocell) | GL5528 LDR | 2 | 2 | Resistor that changes with light | ~10 kΩ light, ~1 MΩ dark | 0.3 |
| 35 | NPN Transistor PN2222 | PN2222A, TO-92 | 2 | 3 (E, B, C) | Tiny electronic switch or amplifier | 40 V, 600 mA | 0.1 |

Notes
- None of these parts carry a serial number. Hobby components are identified by manufacturer
  part number (MPN), package, and sometimes a date or lot code. Only the UNO board could be
  given an asset tag. Keep `serial_number` in the schema anyway so the same records work for
  real data center gear (servers, PDUs, optics) where serials matter.
- "Supply" means quantity in this kit. Market availability (in stock, lifecycle active or EOL)
  is a separate field the extraction pass should fill.
