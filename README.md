# Proyecto Autómata IoT

Sistema IoT para el reconocimiento de gestos mediante Machine Learning utilizando un sensor MPU6050 y un ESP32, con respuesta visual en pantalla OLED, alertas sonoras e interfaz web.

## Estructura del repositorio

* `Codigo/BackEnd/`: Servidor principal (`app.py`), base de datos (`usuarios.db`) y la carpeta `Entrenamiento/` con los datasets (`.csv`), scripts y modelos entrenados (`.pkl`) para detectar gestos (*Ascendio, Descendio, Expelliarmus, Lumos/Nox, Reparo, Stupefy, Wingardium Leviosa*).
* `Codigo/FrontEnd/`: Interfaz web (`index.html`, `styles.css`, `script.js`).
* `Codigo/Hardware/`:
  * `ArduinoIDE/`: Firmware para el ESP32 y lectura del sensor MPU6050.
  * `OLED/`: Control de pantalla, animaciones (`Caras/`) y editor de expresiones (`editor.html`).
  * `Buzzer/`: Script para control de sonido (`buzzer.py`).
* `Documentacion_General/`: Documentación del proyecto, historias de usuario, sprints y diagrama de clases.

## Requisitos

* Python 3
* Arduino IDE (con soporte para ESP32)
* Hardware: ESP32, sensor MPU6050, pantalla OLED y buzzer.

## Ejecución

**1. Backend**
Instala las dependencias y levanta el servidor:
```bash
cd Codigo/BackEnd/Entrenamiento
bash install.sh
cd ..
python app.py
```

**2. Hardware**
* Carga el archivo `Codigo/Hardware/ArduinoIDE/ESP32_MPU_6050_Web_Server.ino` en el ESP32 desde Arduino IDE.
* Para ejecutar las pruebas de pantalla y sonido:
```bash
python Codigo/Hardware/OLED/integrado_test.py
```

**3. Frontend**
Abre `Codigo/FrontEnd/index.html` en tu navegador.
