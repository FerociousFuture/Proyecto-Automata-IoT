import RPi.GPIO as GPIO
import time
import threading
import os
import cv2
import numpy as np
from PIL import Image
import random 

# Importaciones específicas para OLED
# Asegúrate de tener estas librerías instaladas:
# pip3 install luma.oled opencv-python pillow

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306


# =======================================================
# 1. BUZZER CONTROL (ADAPTADO DE buzzer.py)
# =======================================================

class BuzzerController:
    """Clase para controlar el buzzer pasivo mediante PWM."""
    def __init__(self, pin=17, initial_freq=50):
        self.BUZZER_PIN = pin
        self.INITIAL_FREQ = initial_freq
        
        # Configuración GPIO (solo se llama una vez)
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
            self.pwm = GPIO.PWM(self.BUZZER_PIN, self.INITIAL_FREQ) 
            self.pwm.start(0)
        except Exception as e:
             print(f"Error al inicializar GPIO/Buzzer: {e}")
             self.pwm = None # Marcar como no inicializado
        
    def tocar_nota(self, frecuencia, duracion):
        """Genera un tono en el buzzer a una frecuencia y duración dadas."""
        if self.pwm is None: return

        try:
            self.pwm.ChangeFrequency(frecuencia) 
            self.pwm.ChangeDutyCycle(50) 
            time.sleep(duracion) 
            self.pwm.ChangeDutyCycle(0) 
            time.sleep(0.05)
        except ValueError:
            # Ignorar si la frecuencia es inválida
            pass
            
    def tocar_reaccion(self, tono='check'):
        """Toca una secuencia de notas predefinida (Ej. 'check', 'error', o 'skull')."""
        if tono == 'check':
            # Sonido de confirmación (Ascendente: A4, C5, E5)
            melodia = [(440, 0.1), (550, 0.1), (660, 0.1)]
        elif tono == 'error':
            # Sonido de error (Descendente: E5, A4)
            melodia = [(660, 0.15), (440, 0.15)]
        elif tono == 'skull':
            # Riff icónico de "Smoke on the Water" - Deep Purple
            # Notas: G-Bb-C, G-Bb-Db-C, G-Bb-C, Bb-G
            # Frecuencias aproximadas:
            G3 = 196   # Sol
            Bb3 = 233  # Si bemol
            C4 = 262   # Do
            Db4 = 277  # Do sostenido/Re bemol
            
            melodia = [
                # Primera parte: G-Bb-C
                (G3, 0.25), (Bb3, 0.25), (C4, 0.35),
                (0, 0.1),  # Pausa corta
                # Segunda parte: G-Bb-Db-C
                (G3, 0.25), (Bb3, 0.25), (Db4, 0.2), (C4, 0.35),
                (0, 0.1),  # Pausa corta
                # Tercera parte: G-Bb-C
                (G3, 0.25), (Bb3, 0.25), (C4, 0.35),
                (0, 0.1),  # Pausa corta
                # Final: Bb-G (con énfasis)
                (Bb3, 0.25), (G3, 0.5)
            ]
        else:
            return

        for freq, duration in melodia:
            if freq == 0:  # Pausa
                time.sleep(duration)
            else:
                self.tocar_nota(freq, duration)
            
    def cleanup(self):
        """Detiene el PWM."""
        if self.pwm:
            self.pwm.stop()
        

# =======================================================
# 2. OLED ANIMATED CLASS (Copiado de oledTEST.py)
# =======================================================

# --- CONFIGURACIÓN ---
OLED_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64
IDLE_VIDEO_PATH = "Caras/idle.mp4"       # RUTA: Ajusta si es necesario
BLINK_VIDEO_PATH = "Caras/Parpadeo.mp4" # RUTA: Ajusta si es necesario
SKULL_VIDEO_PATH = "Caras/Skull.mp4"    # RUTA: Nuevo video de calavera
ANIMATION_FPS = 20
FRAME_DELAY = 1.0 / ANIMATION_FPS
MIN_IDLE_TIME = 5.0  
MAX_IDLE_TIME = 10.0 

class AnimatedOLED:
    def __init__(self):
        # Configurar dispositivo
        serial = i2c(port=1, address=OLED_ADDRESS)
        self.device = ssd1306(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
        
        # Estado
        self.modo = "idle" 
        self.figura_actual = None
        self.running = True
        self.frame = 0
        
        self.idle_frames = self.load_video_frames(IDLE_VIDEO_PATH)
        self.blink_frames = self.load_video_frames(BLINK_VIDEO_PATH)
        self.skull_frames = self.load_video_frames(SKULL_VIDEO_PATH)
        
        if not self.idle_frames:
            self.idle_frames = [Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)] 
            
        # Variables de control para el parpadeo
        self.is_blinking = False
        self.blink_frame_counter = 0
        self.next_blink_time = time.time() + self.get_random_idle_time()
        
        # Variables para animación de skull
        self.is_skull_playing = False
        self.skull_frame_counter = 0

        # Diccionario de figuras (Caras/Iconos ASCII)
        self.figuras = {
            "cubo": ["   +---+", "  /   /|", " +---+ |", " |   | +", " |   |/", " +---+" ],
            "flecha": ["    ^", "   |||", "   |||", "   |||", " =======" ],
            "check": ["       *", "      **", " * * * *", "  * **", "   **", "   *" ],
            "cruz": ["  * *", "   * *", "    *", "   * *", "  * *" ],
            "circulo": ["  ****", " * *", "* *", " * *", "  ****" ],
            "rayo": ["    **", "   **", "  ****", "    **", "   **", "  **" ],
            "casa": ["    /\\", "   /  \\", "  /____\\", "  |    |", "  | [] |", "  |____|" ],
            "triangulo": ["     *", "    * *", "   * *", "  * *", " *********" ],
            "feliz": ["  .---.", " /  o  \\", "(   ^   )", " \\  -  /", "  '---'" ],
            "triste": ["  .---.", " /  -  \\", "(   v   )", " \\  o  /", "  '---'" ]
        }

    def get_random_idle_time(self):
        return random.uniform(MIN_IDLE_TIME, MAX_IDLE_TIME)

    def load_video_frames(self, video_path):
        """Carga y pre-procesa frames de video en imágenes PIL monocromáticas."""
        if not os.path.exists(video_path):
            print(f"❌ ERROR: Archivo de video no encontrado: {video_path}. Usando un frame de error.")
            image_error = Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)
            draw_error = canvas(image_error)
            draw_error.text((5, 5), "NO VIDEO FILE:", fill=1)
            draw_error.text((5, 15), video_path, fill=1)
            return [image_error] * ANIMATION_FPS
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            resized_frame = cv2.resize(frame, (OLED_WIDTH, OLED_HEIGHT), interpolation=cv2.INTER_AREA)
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            _, monochrome_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
            pil_image = Image.fromarray(monochrome_frame).convert('1')
            frames.append(pil_image)
            
        cap.release()
        print(f"✅ Cargados {len(frames)} frames de '{video_path}'")
        return frames

    def dibujar_idle(self):
        """Dibuja el frame actual de la animación idle o blinking."""
        if self.is_blinking and self.blink_frames:
            current_frame_index = self.blink_frame_counter % len(self.blink_frames)
            frame_image = self.blink_frames[current_frame_index]
            self.device.display(frame_image)
            self.blink_frame_counter += 1
            
            if self.blink_frame_counter >= len(self.blink_frames):
                self.is_blinking = False
                self.blink_frame_counter = 0
                self.next_blink_time = time.time() + self.get_random_idle_time()
        else:
            current_frame_index = self.frame % len(self.idle_frames)
            frame_image = self.idle_frames[current_frame_index]
            self.device.display(frame_image)
            self.frame += 1

    def dibujar_skull(self):
        """Dibuja el frame actual de la animación de skull."""
        if self.skull_frames and self.is_skull_playing:
            current_frame_index = self.skull_frame_counter % len(self.skull_frames)
            frame_image = self.skull_frames[current_frame_index]
            self.device.display(frame_image)
            self.skull_frame_counter += 1
            
            # Si terminó la animación, volver a idle
            if self.skull_frame_counter >= len(self.skull_frames):
                self.is_skull_playing = False
                self.skull_frame_counter = 0
                self.modo = "idle"
                self.next_blink_time = time.time() + self.get_random_idle_time()

    def dibujar_figura(self, nombre_figura):
        """Dibuja una figura específica (mantenemos la lógica ASCII original)."""
        if nombre_figura not in self.figuras:
            return False
        
        figura = self.figuras[nombre_figura]
        
        with canvas(self.device) as draw:
            draw.rectangle(self.device.bounding_box, outline="black", fill="black")
            draw.text((2, 2), nombre_figura.upper(), fill="white")
            y_start = 18
            for i, linea in enumerate(figura):
                x = (OLED_WIDTH - len(linea) * 6) // 2
                draw.text((x, y_start + i * 10), linea, fill="white")
        
        return True
    
    def mostrar_figura(self, nombre):
        """Cambia al modo figura y la muestra."""
        if nombre in self.figuras:
            # Deshabilitar parpadeo mientras se muestra una figura
            self.modo = "figura"
            self.is_blinking = False 
            self.figura_actual = nombre
            self.dibujar_figura(nombre)
            return True
        else:
            return False
    
    def mostrar_skull(self):
        """Inicia la animación de skull."""
        self.modo = "skull"
        self.is_skull_playing = True
        self.is_blinking = False
        self.skull_frame_counter = 0
        return True
            
    def volver_idle(self):
        """Regresa al modo idle y reinicia el temporizador de parpadeo."""
        time.sleep(3) # Mostrar la figura por 3 segundos
        self.modo = "idle"
        self.figura_actual = None
        self.next_blink_time = time.time() + self.get_random_idle_time()
        
    
    def loop_animacion(self):
        """Loop principal de animación con lógica de temporizador aleatorio."""
        while self.running:
            start_time = time.time()
            
            if self.modo == "idle":
                current_time = time.time()
                
                if not self.is_blinking and current_time >= self.next_blink_time and self.blink_frames:
                    self.is_blinking = True
                    self.blink_frame_counter = 0
                
                self.dibujar_idle()
                
                elapsed_time = time.time() - start_time
                sleep_time = FRAME_DELAY - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            elif self.modo == "skull":
                self.dibujar_skull()
                elapsed_time = time.time() - start_time
                sleep_time = FRAME_DELAY - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # Si está en modo figura, solo esperar.
                time.sleep(0.1)

    def iniciar(self):
        """Inicia el loop de animación en un thread."""
        thread = threading.Thread(target=self.loop_animacion, daemon=True)
        thread.start()
        return thread
    
    def detener(self):
        """Detiene la animación y limpia."""
        self.running = False
        time.sleep(0.2)
        self.device.cleanup()
    
    def listar_figuras(self):
        """Lista todas las figuras disponibles."""
        return list(self.figuras.keys())

# =======================================================
# 3. LÓGICA DE INTEGRACIÓN Y PRUEBA
# =======================================================

def ejecutar_comando(oled: AnimatedOLED, buzzer: BuzzerController, comando: str):
    """Ejecuta la acción combinada (OLED + Buzzer) basada en el comando."""
    
    if comando == 'skull':
        # Comando especial: reproducir video de skull + riff de guitarra
        oled.mostrar_skull()
        threading.Thread(target=buzzer.tocar_reaccion, args=('skull',), daemon=True).start()
        print("💀 Comando 'skull' ejecutado (Video Skull.mp4 + Riff de guitarra).")
        
    elif oled.mostrar_figura(comando):
        # 1. Tocar sonido de CHECK en un thread separado para no bloquear la animación del OLED
        threading.Thread(target=buzzer.tocar_reaccion, args=('check',), daemon=True).start()
        # 2. Iniciar el temporizador para volver al modo IDLE
        threading.Thread(target=oled.volver_idle, daemon=True).start()
        print(f"✅ Comando '{comando}' ejecutado (Figura + Tono OK).")
        
    elif comando == 'error':
        # Simula una reacción de error
        oled.mostrar_figura('triste') # Usa una figura para el error
        threading.Thread(target=buzzer.tocar_reaccion, args=('error',), daemon=True).start()
        threading.Thread(target=oled.volver_idle, daemon=True).start()
        print("❌ Comando 'error' ejecutado (Figura TRISTE + Tono ERROR).")
        
    else:
        print(f"Comando desconocido: {comando}. Intenta con una figura, 'skull' o 'error'.")


# =======================================================
# 4. PROGRAMA PRINCIPAL
# =======================================================

if __name__ == "__main__":
    oled = None
    buzzer = None
    try:
        print("Inicializando componentes...")
        
        # 4.1 Inicialización del Buzzer (Pin 17)
        buzzer = BuzzerController(pin=17) 
        
        # 4.2 Inicialización del OLED
        oled = AnimatedOLED()
        oled.iniciar() # Inicia el loop de animación en un thread
        
        print("\n=== SISTEMA LISTO ===")
        print(f"Figuras disponibles: {', '.join(oled.listar_figuras())}")
        print("Comando especial: 'skull' (reproduce Skull.mp4 + riff de guitarra)")
        print("Escribe un nombre de figura para probar la reacción combinada.")
        print("Escribe 'error' para simular un fallo.")
        print("Escribe 'salir' para terminar\n")
        
        # Loop de comandos interactivo
        while True:
            comando = input("Comando: ").strip().lower()
            
            if comando == "salir":
                break
            elif comando == "lista":
                print(f"Figuras disponibles: {', '.join(oled.listar_figuras())}")
                print("Comando especial: 'skull'")
            elif comando:
                ejecutar_comando(oled, buzzer, comando)
                
    except Exception as e:
        print(f"ERROR FATAL durante la ejecución: {e}")
        
    finally:
        print("Limpiando pines GPIO...")
        if oled:
            oled.detener()
        if buzzer:
            buzzer.cleanup()
        # Limpieza final de GPIO, crucial para evitar errores en ejecuciones futuras
        GPIO.cleanup() 
        print("Programa terminado.")