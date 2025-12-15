#!/bin/bash

# Este script instala las dependencias de Python usando una combinación de apt y pip.

echo "--- 1. Actualizando lista de paquetes APT ---"
sudo apt update

# Instalación de paquetes del sistema vía APT (incluye Python y pyserial para compatibilidad con UART/USB)
echo "--- 2. Instalando paquetes del sistema (Python3 y serial) ---"
sudo apt install -y python3 python3-pip python3-serial build-essential

# Nota: Muchos paquetes de APT para Data Science (como python3-numpy o python3-pandas)
# están desactualizados en los repositorios de Raspbian/Debian, por lo que usaremos PIP
# para las versiones más recientes.

# Instalación de librerías Python vía PIP
echo "--- 3. Instalando librerías Python con PIP ---"
pip install --user \
    pandas \
    numpy \
    joblib \
    scikit-learn \
    pyserial

echo "--- 4. Instalación finalizada ---"
echo "Las librerías de Python se han instalado en el directorio de usuario (pip install --user)."
echo "Asegúrate de que el usuario 'pi' pertenece al grupo 'dialout' para usar el puerto serie:"
echo "sudo usermod -a -G dialout pi"
echo "Reinicia la Raspberry Pi después de ejecutar ese último comando."
