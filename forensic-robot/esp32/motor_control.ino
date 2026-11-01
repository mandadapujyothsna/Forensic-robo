/*
 * RoboForensic ESP32 Motor Controller Firmware
 * Receives Single-Character Commands over Serial or Bluetooth:
 *   'F' -> FORWARD
 *   'B' -> BACKWARD
 *   'L' -> TURN LEFT
 *   'R' -> TURN RIGHT
 *   'S' -> STOP
 */

#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

// Motor A Driver Pins (Left Motor)
const int ENA = 14;
const int IN1 = 27;
const int IN2 = 26;

// Motor B Driver Pins (Right Motor)
const int ENB = 32;
const int IN3 = 25;
const int IN4 = 33;

// Base Motor Speed (0 - 255)
int motorSpeed = 160;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("RoboForensic_Rover"); // Bluetooth Device Name
  Serial.println("RoboForensic Motor Controller Ready.");

  // Motor Driver Control Pins as Outputs
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotors();
}

void moveForward() {
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void moveBackward() {
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void turnLeft() {
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH); // Left backwards
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);  // Right forwards
}

void turnRight() {
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);  // Left forwards
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH); // Right backwards
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}

void processCommand(char cmd) {
  switch (cmd) {
    case 'F':
      Serial.println("Action: FORWARD");
      moveForward();
      break;
    case 'B':
      Serial.println("Action: BACKWARD");
      moveBackward();
      break;
    case 'L':
      Serial.println("Action: TURN LEFT");
      turnLeft();
      break;
    case 'R':
      Serial.println("Action: TURN RIGHT");
      turnRight();
      break;
    case 'S':
      Serial.println("Action: STOP");
      stopMotors();
      break;
    default:
      break;
  }
}

void loop() {
  // Check USB Serial input
  if (Serial.available()) {
    char cmd = Serial.read();
    processCommand(cmd);
  }

  // Check Bluetooth Serial input
  if (SerialBT.available()) {
    char cmd = SerialBT.read();
    processCommand(cmd);
  }
}
