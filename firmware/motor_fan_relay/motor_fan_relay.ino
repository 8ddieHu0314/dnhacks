#include <Arduino.h>

constexpr uint8_t BUTTON_PIN = 2;
constexpr uint8_t RELAY_PIN = 8;
constexpr unsigned long DEBOUNCE_MS = 25;

bool lastReading = false;
bool stablePressed = false;
unsigned long changedAt = 0;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  Serial.begin(115200);
  Serial.println(F("{\"event\":\"ready\",\"relay\":false}"));
}

void loop() {
  const bool reading = digitalRead(BUTTON_PIN) == LOW;
  if (reading != lastReading) {
    lastReading = reading;
    changedAt = millis();
  }
  if (millis() - changedAt >= DEBOUNCE_MS && reading != stablePressed) {
    stablePressed = reading;
    digitalWrite(RELAY_PIN, stablePressed ? HIGH : LOW);
    Serial.print(F("{\"event\":\"button\",\"pressed\":"));
    Serial.print(stablePressed ? F("true") : F("false"));
    Serial.print(F(",\"relay\":"));
    Serial.print(stablePressed ? F("true") : F("false"));
    Serial.println('}');
  }
}
