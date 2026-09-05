// Kit-only context sensor node. It does not measure or clear voltage.
const byte LDR_PIN = A0;
const byte TILT_PIN = 3;
const byte BUZZER_PIN = 8;
const byte TRIG_PIN = 9;
const byte ECHO_PIN = 10;
const int LIGHT_ON_THRESHOLD = 500;
const int ZONE_DISTANCE_CM = 80;

long distanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 25000);
  return duration ? duration * 0.0343 / 2 : -1;
}

void printReading(int light, bool tilted, long distance) {
  Serial.print(F("{\"indicator_state\":\""));
  Serial.print(light >= LIGHT_ON_THRESHOLD ? F("on") : F("off"));
  Serial.print(F("\",\"light_level\":"));
  Serial.print(light);
  Serial.print(F(",\"tilt_detected\":"));
  Serial.print(tilted ? F("true") : F("false"));
  Serial.print(F(",\"distance_cm\":"));
  Serial.print(distance);
  Serial.println(F("}"));
}

void setup() {
  Serial.begin(115200);
  pinMode(TILT_PIN, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  int light = analogRead(LDR_PIN);
  bool tilted = digitalRead(TILT_PIN) == LOW;
  long distance = distanceCm();
  if (distance > 0 && distance <= ZONE_DISTANCE_CM) tone(BUZZER_PIN, 2000, 80);
  printReading(light, tilted, distance);
  delay(250);
}
