import time
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306, sh1106

# 1. Configuración de la Pantalla
#
# Configura el bus I2C (I2C en el pin 2,3 es el bus 1 en la RPi)
# Puedes cambiar la dirección si tu i2cdetect te dió una diferente (ej. 0x3D)
serial = i2c(port=1, address=0x3C)

# Inicializa el dispositivo OLED (Ajusta la resolución si es necesario)
# Comúnmente son 128x64 o 128x32. Usa 'ssd1306' para el chip más común.
# Puedes usar sh1106 si tu pantalla lo requiere.
device = ssd1306(serial, width=128, height=64) # <<-- AJUSTA ESTOS VALORES

# 2. Dibujar y Mostrar el Mensaje
def display_message(message_line1, message_line2):
    # 'canvas' permite dibujar en la pantalla
    with canvas(device) as draw:
        # Limpia la pantalla
        draw.rectangle(device.bounding_box, outline="black", fill="black")

        # Configura la fuente (usaremos la fuente por defecto)
        # Puedes instalar y usar otras fuentes si quieres
        
        # Dibuja la primera línea de texto (x, y)
        # 0,0 es la esquina superior izquierda
        draw.text((0, 0), message_line1, fill="white")
        
        # Dibuja la segunda línea de texto (puedes ajustar '15' para el espaciado)
        draw.text((0, 15), message_line2, fill="white")
        
# 3. Llamar a la Función y Ejecutar
try:
    print("Mostrando mensaje en la OLED...")
    
    # Llama a la función para mostrar tu mensaje
    display_message("¡Hola, Raspberry Pi!", "OLED lista y funcionando.")
    
    # Mantener el mensaje por 5 segundos
    time.sleep(5)
    
    # Opcional: Apagar la pantalla para ahorrar energía
    # device.hide()
    
    print("Programa terminado.")

except KeyboardInterrupt:
    print("\nInterrupción por el usuario. Limpiando y saliendo.")
    device.cleanup()
