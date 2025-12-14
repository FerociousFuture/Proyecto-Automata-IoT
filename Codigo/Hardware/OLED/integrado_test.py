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
    
    def get_duracion_melodia(self, tono):
        """Calcula la duración total de una melodía."""
        melodias = self._obtener_melodias()
        if tono in melodias:
            return sum(duracion for _, duracion in melodias[tono])
        return 0
    
    def _obtener_melodias(self):
        """Retorna el diccionario de todas las melodías disponibles."""
        # Notas comunes
        G4 = 392
        Bb4 = 466
        C5 = 523
        Db5 = 554
        E5 = 659
        D5 = 587
        A4 = 440
        F5 = 698
        
        # Among Us Theme - Melodía característica
        C4 = 262
        D4 = 294
        Eb4 = 311
        F4 = 349
        G4 = 392
        Ab4 = 415
        
        return {
            'check': [(440, 0.1), (550, 0.1), (660, 0.1)],
            'error': [(660, 0.15), (440, 0.15)],
            'skull': [
                (G4, 0.15), (Bb4, 0.15), (C5, 0.25), (0, 0.05),
                (G4, 0.15), (Bb4, 0.15), (Db5, 0.12), (C5, 0.25), (0, 0.05),
                (G4, 0.15), (Bb4, 0.15), (C5, 0.25), (0, 0.05),
                (Bb4, 0.15), (G4, 0.35)
            ],
            'navidad': [
                # "Jingle Bells" completo
                (E5, 0.2), (E5, 0.2), (E5, 0.4),
                (E5, 0.2), (E5, 0.2), (E5, 0.4),
                (E5, 0.2), (G4, 0.2), (C5, 0.3), (D5, 0.1), (E5, 0.6),
                (0, 0.1),
                (F5, 0.2), (F5, 0.2), (F5, 0.3), (F5, 0.1),
                (F5, 0.2), (E5, 0.2), (E5, 0.2), (E5, 0.1), (E5, 0.1),
                (E5, 0.2), (D5, 0.2), (D5, 0.2), (E5, 0.2),
                (D5, 0.4), (G4, 0.4),
            ],
            'amongus': [
                # Among Us Theme - Intro característico
                (C4, 0.15), (Eb4, 0.15), (F4, 0.15), (G4, 0.3), (0, 0.1),
                (F4, 0.15), (Eb4, 0.15), (C4, 0.3), (0, 0.1),
                (C4, 0.15), (Eb4, 0.15), (F4, 0.15), (G4, 0.3), (0, 0.1),
                (Ab4, 0.15), (G4, 0.15), (F4, 0.15), (Eb4, 0.3), (0, 0.1),
                # Repetir con variación
                (C4, 0.15), (Eb4, 0.15), (F4, 0.15), (G4, 0.3), (0, 0.1),
                (F4, 0.15), (Eb4, 0.15), (D4, 0.15), (C4, 0.3), (0, 0.1),
            ]
        }
            
    def tocar_reaccion(self, tono='check'):
        """Toca una secuencia de notas predefinida."""
        melodias = self._obtener_melodias()
        
        if tono not in melodias:
            return
        
        melodia = melodias[tono]
        
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
IDLE_VIDEO_PATH = "Caras/idle.mp4"
BLINK_VIDEO_PATH = "Caras/Parpadeo.mp4"
SKULL_VIDEO_PATH = "Caras/Skull.mp4"
NAVIDAD_IMAGE_PATH = "Caras/Navidad.png"
AMONGUS_IMAGE_PATH = "Caras/amongus.png"
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
        
        # Cargar videos
        self.idle_frames = self.load_video_frames(IDLE_VIDEO_PATH)
        self.blink_frames = self.load_video_frames(BLINK_VIDEO_PATH)
        self.skull_frames = self.load_video_frames(SKULL_VIDEO_PATH)
        
        # Cargar imágenes estáticas
        self.navidad_image = self.load_static_image(NAVIDAD_IMAGE_PATH)
        self.amongus_image = self.load_static_image(AMONGUS_IMAGE_PATH)
        
        if not self.idle_frames:
            self.idle_frames = [Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)] 
            
        # Variables de control para el parpadeo
        self.is_blinking = False
        self.blink_frame_counter = 0
        self.next_blink_time = time.time() + self.get_random_idle_time()
        
        # Variables para animación de skull (NO LOOP)
        self.is_skull_playing = False
        self.skull_frame_counter = 0
        
        # Variables para imágenes estáticas con temporizador
        self.static_image_display = None
        self.static_image_end_time = 0

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

    def load_static_image(self, image_path):
        """Carga una imagen estática PNG y la convierte a monocromática."""
        if not os.path.exists(image_path):
            print(f"❌ ERROR: Archivo de imagen no encontrado: {image_path}")
            image_error = Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)
            return image_error
        
        try:
            # Cargar imagen
            img = Image.open(image_path)
            # Redimensionar manteniendo aspecto ratio
            img.thumbnail((OLED_WIDTH, OLED_HEIGHT), Image.Resampling.LANCZOS)
            # Convertir a escala de grises y luego a monocromático
            img = img.convert('L')
            # Aplicar threshold para convertir a blanco y negro puro
            img = img.point(lambda x: 255 if x > 127 else 0, mode='1')
            
            # Centrar la imagen en un canvas del tamaño del OLED
            canvas_img = Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)
            offset = ((OLED_WIDTH - img.width) // 2, (OLED_HEIGHT - img.height) // 2)
            canvas_img.paste(img, offset)
            
            print(f"✅ Imagen cargada: '{image_path}'")
            return canvas_img
        except Exception as e:
            print(f"❌ Error al cargar imagen {image_path}: {e}")
            return Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)

    def load_video_frames(self, video_path):
        """Carga y pre-procesa frames de video en imágenes PIL monocromáticas."""
        if not os.path.exists(video_path):
            print(f"❌ ERROR: Archivo de video no encontrado: {video_path}")
            image_error = Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)
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
        """Dibuja el frame actual de la animación idle o blinking (CON LOOP)."""
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
            # IDLE con loop infinito
            current_frame_index = self.frame % len(self.idle_frames)
            frame_image = self.idle_frames[current_frame_index]
            self.device.display(frame_image)
            self.frame += 1

    def dibujar_skull(self):
        """Dibuja el frame actual de la animación de skull (SIN LOOP)."""
        if self.skull_frames and self.is_skull_playing:
            if self.skull_frame_counter < len(self.skull_frames):
                frame_image = self.skull_frames[self.skull_frame_counter]
                self.device.display(frame_image)
                self.skull_frame_counter += 1
            else:
                # Terminó la animación, volver a idle
                self.is_skull_playing = False
                self.skull_frame_counter = 0
                self.modo = "idle"
                self.next_blink_time = time.time() + self.get_random_idle_time()
    
    def dibujar_static_image(self):
        """Dibuja una imagen estática hasta que expire el temporizador."""
        if self.static_image_display and time.time() < self.static_image_end_time:
            self.device.display(self.static_image_display)
        else:
            # Temporizador expirado, volver a idle
            self.static_image_display = None
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
            self.modo = "figura"
            self.is_blinking = False 
            self.figura_actual = nombre
            self.dibujar_figura(nombre)
            return True
        else:
            return False
    
    def mostrar_skull(self):
        """Inicia la animación de skull (SIN LOOP)."""
        self.modo = "skull"
        self.is_skull_playing = True
        self.is_blinking = False
        self.skull_frame_counter = 0
        return True
    
    def mostrar_imagen_con_temporizador(self, imagen, duracion):
        """Muestra una imagen estática por una duración específica."""
        self.modo = "static_image"
        self.static_image_display = imagen
        self.static_image_end_time = time.time() + duracion
        self.is_blinking = False
        
    def mostrar_navidad(self, duracion):
        """Muestra la imagen de Navidad por la duración especificada."""
        self.mostrar_imagen_con_temporizador(self.navidad_image, duracion)
        
    def mostrar_amongus(self, duracion):
        """Muestra la imagen de Among Us por la duración especificada."""
        self.mostrar_imagen_con_temporizador(self.amongus_image, duracion)
            
    def volver_idle(self):
        """Regresa al modo idle y reinicia el temporizador de parpadeo."""
        time.sleep(3)
        self.modo = "idle"
        self.figura_actual = None
        self.next_blink_time = time.time() + self.get_random_idle_time()
        
    
    def loop_animacion(self):
        """Loop principal de animación."""
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
                    
            elif self.modo == "static_image":
                self.dibujar_static_image()
                time.sleep(0.1)
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
        # Comando especial: reproducir video de skull + riff de guitarra (SIN LOOP)
        oled.mostrar_skull()
        threading.Thread(target=buzzer.tocar_reaccion, args=('skull',), daemon=True).start()
        print("💀 Comando 'skull' ejecutado (Video Skull.mp4 + Riff de guitarra).")
    
    elif comando == 'navidad':
        # Comando navideño: imagen estática + Jingle Bells
        duracion = buzzer.get_duracion_melodia('navidad') + 0.5  # +0.5s de margen
        oled.mostrar_navidad(duracion)
        threading.Thread(target=buzzer.tocar_reaccion, args=('navidad',), daemon=True).start()
        print(f"🎄 Comando 'navidad' ejecutado (Imagen Navidad.png + Jingle Bells por {duracion:.1f}s).")
    
    elif comando == 'amongus':
        # Comando Among Us: imagen estática + tema de Among Us
        duracion = buzzer.get_duracion_melodia('amongus') + 0.5  # +0.5s de margen
        oled.mostrar_amongus(duracion)
        threading.Thread(target=buzzer.tocar_reaccion, args=('amongus',), daemon=True).start()
        print(f"🚀 Comando 'amongus' ejecutado (Imagen amongus.png + Among Us Theme por {duracion:.1f}s).")
        
    elif oled.mostrar_figura(comando):
        # Tocar sonido de CHECK
        threading.Thread(target=buzzer.tocar_reaccion, args=('check',), daemon=True).start()
        threading.Thread(target=oled.volver_idle, daemon=True).start()
        print(f"✅ Comando '{comando}' ejecutado (Figura + Tono OK).")
        
    elif comando == 'error':
        # Simula una reacción de error
        oled.mostrar_figura('triste')
        threading.Thread(target=buzzer.tocar_reaccion, args=('error',), daemon=True).start()
        threading.Thread(target=oled.volver_idle, daemon=True).start()
        print("❌ Comando 'error' ejecutado (Figura TRISTE + Tono ERROR).")
        
    else:
        print(f"Comando desconocido: {comando}. Intenta con una figura, 'skull', 'navidad', 'amongus' o 'error'.")


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
        oled.iniciar()
        
        print("\n=== SISTEMA LISTO ===")
        print(f"Figuras disponibles: {', '.join(oled.listar_figuras())}")
        print("Comandos especiales:")
        print("  - 'skull' (video Skull.mp4 + riff de guitarra) 💀")
        print("  - 'navidad' (imagen Navidad.png + Jingle Bells) 🎄")
        print("  - 'amongus' (imagen amongus.png + Among Us Theme) 🚀")
        print("\nNOTA: Solo IDLE y Parpadeo hacen loop. El resto se reproduce UNA VEZ.")
        print("Escribe 'error' para simular un fallo.")
        print("Escribe 'salir' para terminar\n")
        
        # Loop de comandos interactivo
        while True:
            comando = input("Comando: ").strip().lower()
            
            if comando == "salir":
                break
            elif comando == "lista":
                print(f"Figuras disponibles: {', '.join(oled.listar_figuras())}")
                print("Comandos especiales: 'skull', 'navidad', 'amongus'")
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
        GPIO.cleanup() 
        print("Programa terminado.")