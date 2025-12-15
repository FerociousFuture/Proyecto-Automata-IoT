
#include <Arduino.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino_JSON.h>
#include "LittleFS.h"

// credenciales de red
const char* ssid = "INFINITUM8895";
const char* password = "333NmGEs4s";

// Nombre del archivo CSV
const char* DATA_FILE = "/sensor_data.csv"; 
// Intervalo de registro CSV (cada 500 ms)
unsigned long lastTimeDataRecord = 0;
const unsigned long DATA_RECORD_DELAY = 500; 

// === NUEVA VARIABLE DE ESTADO ===
bool dataRecording = false; // Controla si se está grabando activamente

// Create AsyncWebServer object on port 80
AsyncWebServer server(80);
// Create an Event Source on /events
AsyncEventSource events("/events");
// Json Variable to Hold Sensor Readings
JSONVar readings;

// Timer variables
unsigned long lastTime = 0;  
unsigned long lastTimeTemperature = 0;
unsigned long lastTimeAcc = 0;
unsigned long gyroDelay = 10;
unsigned long temperatureDelay = 1000;
unsigned long accelerometerDelay = 200;

// Create a sensor object
Adafruit_MPU6050 mpu;

sensors_event_t a, g, temp;

float gyroX, gyroY, gyroZ;
float accX, accY, accZ;
float temperature;

//Gyroscope sensor deviation
float gyroXerror = 0.14;
float gyroYerror = 0.06;
float gyroZerror = 0.02;

// --- Funciones de Archivo LittleFS ---

// Escribe datos al final del archivo especificado
void appendFile(fs::FS &fs, const char * path, const char * message){
    Serial.printf("Appending to file: %s\n", path);
    File file = fs.open(path, FILE_APPEND);
    if(!file){
        Serial.println("Failed to open file for appending");
        return;
    }
    if(file.print(message)){
        // Serial.println("Message appended"); 
    } else {
        Serial.println("Append failed");
    }
    file.close();
}

// Función para registrar los datos actuales en el CSV
void recordData() {
  // Asegúrate de obtener los últimos datos antes de grabar
  mpu.getEvent(&a, &g, &temp);

  String dataMessage = "";
  
  // Encabezados: Timestamp_ms,Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z,Temperature_C
  
  // Agregar un timestamp
  dataMessage += String(millis()) + ",";
  
  // Datos de Gyroscopio (X, Y, Z) - Usamos los valores integrados
  dataMessage += String(gyroX) + ",";
  dataMessage += String(gyroY) + ",";
  dataMessage += String(gyroZ) + ",";
  
  // Datos de Acelerómetro (X, Y, Z)
  dataMessage += String(a.acceleration.x) + ",";
  dataMessage += String(a.acceleration.y) + ",";
  dataMessage += String(a.acceleration.z) + ",";
  
  // Temperatura
  dataMessage += String(temp.temperature) + "\n";
  
  appendFile(LittleFS, DATA_FILE, dataMessage.c_str());
}

// --- Funciones de Inicialización ---

// Init MPU6050
void initMPU(){
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");
}

void initLittleFS() {
  if (!LittleFS.begin()) {
    Serial.println("An error has occurred while mounting LittleFS");
  }
  Serial.println("LittleFS mounted successfully");
}

// Initialize WiFi
void initWiFi() {
  WiFi.mode(WIFI_STA);
  // Deshabilitar el modo de ahorro de energía para estabilidad
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);
  Serial.println("");
  Serial.print("Connecting to WiFi...");
  // Bucle de espera de conexión
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println("");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

// --- Funciones de Lectura de Sensores Originales (sin cambios) ---

String getGyroReadings(){
  mpu.getEvent(&a, &g, &temp);
  float gyroX_temp = g.gyro.x;
  if(abs(gyroX_temp) > gyroXerror)  {
    gyroX += gyroX_temp/50.00;
  }
  float gyroY_temp = g.gyro.y;
  if(abs(gyroY_temp) > gyroYerror) {
    gyroY += gyroY_temp/70.00;
  }
  float gyroZ_temp = g.gyro.z;
  if(abs(gyroZ_temp) > gyroZerror) {
    gyroZ += gyroZ_temp/90.00;
  }
  readings["gyroX"] = String(gyroX);
  readings["gyroY"] = String(gyroY);
  readings["gyroZ"] = String(gyroZ);
  String jsonString = JSON.stringify(readings);
  return jsonString;
}

String getAccReadings() {
  mpu.getEvent(&a, &g, &temp);
  accX = a.acceleration.x;
  accY = a.acceleration.y;
  accZ = a.acceleration.z;
  readings["accX"] = String(accX);
  readings["accY"] = String(accY);
  readings["accZ"] = String(accZ);
  String accString = JSON.stringify (readings);
  return accString;
}

String getTemperature(){
  mpu.getEvent(&a, &g, &temp);
  temperature = temp.temperature;
  return String(temperature);
}

// --- Setup y Loop ---

void setup() {
  Serial.begin(115200);
  initWiFi();
  initLittleFS();
  initMPU();

  // 1. ELIMINAR y ESCRIBIR ENCABEZADOS del CSV solo en el primer inicio
  if (LittleFS.exists(DATA_FILE)) {
    LittleFS.remove(DATA_FILE);
    Serial.println("Existing data file removed.");
  }
  // Encabezados del archivo CSV
  const char* header = "Timestamp_ms,Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z,Temperature_C\n";
  appendFile(LittleFS, DATA_FILE, header);


  // --- HANDLERS ESPECÍFICOS (DEBEN IR PRIMERO) ---
  
  // === NUEVOS HANDLERS PARA GRABACIÓN CSV ===
  server.on("/start_record", HTTP_GET, [](AsyncWebServerRequest *request){
    dataRecording = true;
    Serial.println("CSV Recording Started!");
    request->send(200, "text/plain", "Recording Started");
  });

  server.on("/stop_record", HTTP_GET, [](AsyncWebServerRequest *request){
    dataRecording = false;
    Serial.println("CSV Recording Stopped!");
    request->send(200, "text/plain", "Recording Stopped");
  });
  // ===========================================

  // Handlers para resetear las variables del giroscopio
  server.on("/reset", HTTP_GET, [](AsyncWebServerRequest *request){
    gyroX=0;
    gyroY=0;
    gyroZ=0;
    request->send(200, "text/plain", "OK");
  });
  server.on("/resetX", HTTP_GET, [](AsyncWebServerRequest *request){
    gyroX=0;
    request->send(200, "text/plain", "OK");
  });
  server.on("/resetY", HTTP_GET, [](AsyncWebServerRequest *request){
    gyroY=0;
    request->send(200, "text/plain", "OK");
  });
  server.on("/resetZ", HTTP_GET, [](AsyncWebServerRequest *request){
    gyroZ=0;
    request->send(200, "text/plain", "OK");
  });
  
  // Endpoint para ver/descargar el archivo CSV
  server.on("/data", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send(LittleFS, DATA_FILE, "text/csv");
  });


  // --- HANDLERS GENÉRICOS ---

  // Servir la página principal (index.html)
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send(LittleFS, "/index.html", "text/html");
  });

  // Servir todos los demás archivos estáticos (style.css, script.js, etc.)
  server.serveStatic("/", LittleFS, "/");

  // Handle Web Server Events
  events.onConnect([](AsyncEventSourceClient *client){
    if(client->lastId()){
      Serial.printf("Client reconnected! Last message ID that it got is: %u\n", client->lastId());
    }
    client->send("hello!", NULL, millis(), 10000);
  });
  server.addHandler(&events);

  server.begin();
}

void loop() {
  // --- Eventos Server-Sent (para el Dashboard Web) ---
  if ((millis() - lastTime) > gyroDelay) {
    events.send(getGyroReadings().c_str(),"gyro_readings",millis());
    lastTime = millis();
  }
  if ((millis() - lastTimeAcc) > accelerometerDelay) {
    events.send(getAccReadings().c_str(),"accelerometer_readings",millis());
    lastTimeAcc = millis();
  }
  if ((millis() - lastTimeTemperature) > temperatureDelay) {
    events.send(getTemperature().c_str(),"temperature_reading",millis());
    lastTimeTemperature = millis();
  }

  // --- Lógica de Registro CSV Condicional ---
  // Solo registra datos si la variable 'dataRecording' es verdadera
  if (dataRecording && (millis() - lastTimeDataRecord) > DATA_RECORD_DELAY) {
    recordData();
    lastTimeDataRecord = millis();
  }
}