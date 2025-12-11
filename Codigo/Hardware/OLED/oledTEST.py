import time
import threading
import os
import cv2
import numpy as np
from PIL import Image
import random  # Necesario para la aleatoriedad

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# --- CONFIGURACIÓN ---
OLED_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64

# --- RUTAS DE ARCHIVOS DE VIDEO (AJUSTADAS PARA LA CARPETA 'Caras') ---
IDLE_VIDEO_PATH = "Caras/idle.mp4"
BLINK_VIDEO_PATH = "Caras/Parpadeo.mp4"

# --- PARAMETROS DE ANIMACIÓN ---
ANIMATION_FPS = 20
FRAME_DELAY = 1.0 / ANIMATION_FPS

# --- NUEVOS PARAMETROS DE TIEMPO DE PARPADEO ---
MIN_IDLE_TIME = 5.0  # Mínimo de 5 segundos entre parpadeos
MAX_IDLE_TIME = 10.0 # Máximo de 10 segundos entre parpadeos

class AnimatedOLED:
    def __init__(self):
        # Configurar dispositivo
        serial = i2c(port=1, address=OLED_ADDRESS)
        self.device = ssd1306(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
        
        # Estado
        self.modo = "idle"  # puede ser "idle", "figura", o "blinking"
        self.figura_actual = None
        self.running = True
        self.frame = 0
        
        # --- LÓGICA DE CARGA DE FRAMES DE VIDEO ---
        self.idle_frames = self.load_video_frames(IDLE_VIDEO_PATH)
        self.blink_frames = self.load_video_frames(BLINK_VIDEO_PATH)
        
        if not self.idle_frames:
            # Fallback para evitar errores si no se carga el video idle
            self.idle_frames = [Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)] 
            
        # Variables de control para el parpadeo
        self.is_blinking = False
        self.blink_frame_counter = 0
        self.next_blink_time = time.time() + self.get_random_idle_time()

        # Diccionario de figuras disponibles (ASCII)
        self.figuras = {
            "cubo": ["   +---+", "  /   /|", " +---+ |", " |   | +", " |   |/", " +---+" ],
            "flecha": ["    ^", "   |||", "   |||", "   |||", " =======" ],
            "check": ["       *", "      **", " * * * *", "  * **", "   **", "   *" ],
            "cruz": ["  * *", "   * *", "    *", "   * *", "  * *" ],
            "circulo": ["  ****", " * *", "* *", " * *", "  ****" ],
            "rayo": ["    **", "   **", "  ****", "    **", "   **", "  **" ],
            "casa": ["    /\\", "   /  \\", "  /____\\", "  |    |", "  | [] |", "  |____|" ],
            "triangulo": ["     *", "    * *", "   * *", "  * *", " *********" ]
        }

    def get_random_idle_time(self):
        """Genera un tiempo de espera aleatorio entre MIN_IDLE_TIME y MAX_IDLE_TIME."""
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
        
        # Lógica para alternar entre animación normal y parpadeo
        if self.is_blinking and self.blink_frames:
            # 1. Mostrar frame de parpadeo
            current_frame_index = self.blink_frame_counter % len(self.blink_frames)
            frame_image = self.blink_frames[current_frame_index]
            self.device.display(frame_image)
            self.blink_frame_counter += 1
            
            # 2. Si la animación de parpadeo termina, regresar a idle
            if self.blink_frame_counter >= len(self.blink_frames):
                self.is_blinking = False
                self.blink_frame_counter = 0
                self.next_blink_time = time.time() + self.get_random_idle_time()
                print(f"Parpadeo terminado. Próximo parpadeo en {self.next_blink_time - time.time():.2f} segundos.")
                
        else:
            # Mostrar frame de idle (animación normal)
            current_frame_index = self.frame % len(self.idle_frames)
            frame_image = self.idle_frames[current_frame_index]
            self.device.display(frame_image)
            self.frame += 1

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
        """Cambia al modo figura y muestra por 3 segundos."""
        if nombre in self.figuras:
            # Deshabilitar parpadeo mientras se muestra una figura
            self.modo = "figura"
            self.is_blinking = False 
            self.figura_actual = nombre
            self.dibujar_figura(nombre)
            
            # Volver a idle después de 3 segundos
            def volver_idle():
                time.sleep(3)
                self.modo = "idle"
                self.figura_actual = None
                # Restablecer el temporizador de parpadeo al volver a idle
                self.next_blink_time = time.time() + self.get_random_idle_time()
            
            threading.Thread(target=volver_idle, daemon=True).start()
            return True
        else:
            print(f"Figura '{nombre}' no encontrada. Disponibles: {list(self.figuras.keys())}")
            return False
    
    def loop_animacion(self):
        """Loop principal de animación con lógica de temporizador aleatorio."""
        while self.running:
            start_time = time.time()
            
            if self.modo == "idle":
                current_time = time.time()
                
                # Iniciar el parpadeo si es el momento y la animación blink fue cargada
                if not self.is_blinking and current_time >= self.next_blink_time and self.blink_frames:
                    self.is_blinking = True
                    self.blink_frame_counter = 0
                
                self.dibujar_idle()
                
                # Control de velocidad (para mantener el FPS constante)
                elapsed_time = time.time() - start_time
                sleep_time = FRAME_DELAY - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # Si está en modo figura, solo esperar. El temporizador de parpadeo
                # se restablece cuando vuelve al modo 'idle'.
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

# --- PROGRAMA PRINCIPAL ---
if __name__ == "__main__":
    try:
        print("Iniciando OLED animado con temporizador aleatorio...")
        oled = AnimatedOLED()
        
        # Iniciar animación
        oled.iniciar()
        
        print("\n=== COMANDOS DISPONIBLES ===")
        print(f"Parpadeo aleatorio entre {MIN_IDLE_TIME}s y {MAX_IDLE_TIME}s.")
        print("Escribe el nombre de una figura (ASCII) para mostrarla temporalmente:")
        print(f"Figuras: {', '.join(oled.listar_figuras())}")
        print("Escribe 'salir' para terminar\n")
        
        # Loop de comandos
        while True:
            comando = input("Comando: ").strip().lower()
            
            if comando == "salir":
                print("Cerrando...")
                break
            elif comando == "lista":
                print(f"Figuras disponibles: {', '.join(oled.listar_figuras())}")
            elif comando:
                if not oled.mostrar_figura(comando):
                    print("Intenta con otra figura o escribe 'lista' para ver opciones")
        
        oled.detener()
        print("Programa terminado")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nVerifica las dependencias (opencv-python, Pillow) y las rutas de los videos.")