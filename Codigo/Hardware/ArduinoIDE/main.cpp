#include <Arduino.h>
#include <Wire.h> 
#include <MPU6050.h> 

MPU6050 mpu;

// Constante de escala para convertir datos crudos a G's
const float ACCEL_SCALE = 16384.0;

int16_t ax, ay, az; 

// Variables para controlar el muestreo (20 Hz)
unsigned long previousMillis = 0;
const long interval = 50; 

void setup() {
  Serial.begin(115200);
  Wire.begin(); 
  
  delay(100); 

  // Imprimir encabezado con unidades de G y la Magnitud
  Serial.println("Timestamp (ms),Ax (G),Ay (G),Az (G),Magnitud (G)");

  mpu.initialize();

  if (mpu.testConnection()) {
    Serial.println("CONEXION_MPU_OK");
  } else {
    Serial.println("ERROR_MPU");
  }
}

void loop() {
  unsigned long currentMillis = millis();

  // Muestrear cada 50 ms
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // 1. Lectura de datos crudos (solo aceleración)
    // Se ignoran los giroscopios para simplificar la visualización del movimiento
    int16_t gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz); 

    // 2. Conversión a G's (valores decimales)
    float acc_x_g = (float)ax / ACCEL_SCALE;
    float acc_y_g = (float)ay / ACCEL_SCALE;
    float acc_z_g = (float)az / ACCEL_SCALE;

    // 3. Cálculo de la Magnitud Vectorial (Fuerza total)
    // Magnitud = sqrt(x² + y² + z²)
    float magnitude = sqrt(
      acc_x_g * acc_x_g + 
      acc_y_g * acc_y_g + 
      acc_z_g * acc_z_g
    );

    // 4. Salida en formato CSV con precisión decimal
    Serial.print(currentMillis); Serial.print(",");
    
    // Usamos dos decimales para una mejor lectura
    Serial.print(acc_x_g, 2); Serial.print(",");
    Serial.print(acc_y_g, 2); Serial.print(",");
    Serial.print(acc_z_g, 2); Serial.print(",");
    
    // El valor clave: la Magnitud
    Serial.print(magnitude, 2); Serial.println();
  }
}