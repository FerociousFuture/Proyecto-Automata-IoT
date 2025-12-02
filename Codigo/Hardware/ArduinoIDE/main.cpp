#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

int16_t ax, ay, az;
int16_t gx, gy, gz;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  Serial.println("--- ESP32 + MPU6050 ---");
  mpu.initialize();

  if (mpu.testConnection()) {
    Serial.println("MPU6050 OK");
  } else {
    Serial.println("ERROR: No se detecta el MPU6050");
  }
}

void loop() {
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  Serial.print("ACCEL: ");
  Serial.print(ax); Serial.print(" | ");
  Serial.print(ay); Serial.print(" | ");
  Serial.println(az);

  Serial.print("GYRO: ");
  Serial.print(gx); Serial.print(" | ");
  Serial.print(gy); Serial.print(" | ");
  Serial.println(gz);

  Serial.println("-------------------------");

  delay(100);
}

