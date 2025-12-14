import RPi.GPIO as GPIO
import time

# --- Configuración ---
BUZZER_PIN = 17  # Pin BCM que estás usando para controlar la base del transistor

# Configuración del modo GPIO
GPIO.setmode(GPIO.BCM)
# Configura el pin como salida
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Inicializa el objeto PWM (Pulse Width Modulation) en el pin
# La frecuencia inicial es 50 Hz (un tono muy grave, casi inaudible)
pwm = GPIO.PWM(BUZZER_PIN, 50) 
# Inicializa el ciclo de trabajo (duty cycle) a 0 (buzzer apagado)
pwm.start(0) 

def tocar_nota(frecuencia, duracion):
    """Genera un tono en el buzzer a una frecuencia y duración dadas."""
    # Cambia la frecuencia de la onda (esto cambia el tono)
    pwm.ChangeFrequency(frecuencia) 
    # Establece el ciclo de trabajo a 50% (para que suene)
    pwm.ChangeDutyCycle(50) 
    # Espera la duración del tono
    time.sleep(duracion) 
    # Detiene el sonido (Duty Cycle a 0)
    pwm.ChangeDutyCycle(0) 
    # Pequeña pausa entre notas
    time.sleep(0.05) 

try:
    print("Iniciando prueba de sonido...")
    
    # --- 1. Tono Fijo (440 Hz es la nota A4 - La central) ---
    print("Tono A4 (440 Hz) por 1 segundo")
    tocar_nota(440, 1)

    # --- 2. Pequeña Melodía (Escala ascendente) ---
    print("Tocando una pequeña melodía...")
    
    # Frecuencias de notas musicales comunes (C4, D4, E4, F4)
    melodia = [262, 294, 330, 349] 
    
    for nota in melodia:
        tocar_nota(nota, 0.25) # Cada nota dura 0.25 segundos

    print("Prueba finalizada.")

except KeyboardInterrupt:
    print("Detenido por el usuario.")

finally:
    # --- Limpieza (MUY IMPORTANTE) ---
    pwm.stop()      # Detiene el PWM
    GPIO.cleanup()  # Libera los pines GPIO para evitar errores
    print("Pines GPIO liberados.")