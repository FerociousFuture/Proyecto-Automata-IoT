import time
import threading
import os
# Importaciones necesarias para OpenCV y la manipulación de imágenes
import cv2
import numpy as np
from PIL import Image

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# --- CONFIGURACIÓN ---
OLED_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64

# --- RUTAS DE ARCHIVOS DE VIDEO (Ajustar según sea necesario) ---
# **IMPORTANTE:** Estos archivos de video deben existir en el mismo directorio
# o proporcionar la ruta completa en su sistema.
IDLE_VIDEO_PATH = "Caras/idle.mp4"
BLINK_VIDEO_PATH = "Caras/Parpadeo.mp4"

# --- FRAME RATE DE LA ANIMACIÓN ---
# Si los videos son muy largos, puede ser necesario ajustar esto
ANIMATION_FPS = 20
FRAME_DELAY = 1.0 / ANIMATION_FPS

class AnimatedOLED:
    def __init__(self):
        # Configurar dispositivo
        serial = i2c(port=1, address=OLED_ADDRESS)
        self.device = ssd1306(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
        
        # Estado
        self.modo = "idle"  # puede ser "idle" o "figura"
        self.figura_actual = None
        self.running = True
        self.frame = 0
        
        # --- NUEVA LÓGICA DE CARGA DE FRAMES DE VIDEO ---
        # Cargar frames para el modo 'idle' y 'parpadeando'
        idle_frames = self.load_video_frames(IDLE_VIDEO_PATH)
        blink_frames = self.load_video_frames(BLINK_VIDEO_PATH)
        
        # Combinar las animaciones para simular el ciclo del modo idle
        # Esto permite una transición fluida si los videos están bien diseñados
        self.idle_frames = idle_frames + blink_frames * 2
        
        # Diccionario de figuras disponibles (mantenemos el ASCII para compatibilidad)
        self.figuras = {
            "cubo": ["   +---+", "  /   /|", " +---+ |", " |   | +", " |   |/", " +---+" ],
            "flecha": ["    ^", "   |||", "   |||", "   |||", " =======" ],
            "check": ["       *", "      **", " * **", "  * **", "   **", "   *" ],
            "cruz": ["  * *", "   * *", "    *", "   * *", "  * *" ],
            "circulo": ["  ****", " * *", "* *", " * *", "  ****" ],
            "rayo": ["    **", "   **", "  ****", "    **", "   **", "  **" ],
            "casa": ["    /\\", "   /  \\", "  /____\\", "  |    |", "  | [] |", "  |____|" ],
            "triangulo": ["     *", "    * *", "   * *", "  * *", " *********" ]
        }

    def load_video_frames(self, video_path):
        """Carga y pre-procesa frames de video en imágenes PIL monocromáticas."""
        if not os.path.exists(video_path):
            print(f"❌ ERROR: Archivo de video no encontrado: {video_path}. Usando un frame de error.")
            # Crear un frame de error simple (fondo negro con "X" blanca)
            image_error = Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)
            draw_error = canvas(image_error)
            draw_error.text((20, 20), "NO VIDEO FILE", fill=1)
            draw_error.text((20, 30), video_path.split('/')[-1], fill=1)
            return [image_error] * 10 # 10 frames de error
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Redimensionar al tamaño del OLED (128x64)
            resized_frame = cv2.resize(frame, (OLED_WIDTH, OLED_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # 2. Convertir a escala de grises
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # 3. Aplicar umbral para obtener blanco y negro (monocromático, modo '1')
            _, monochrome_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
            
            # 4. Convertir a imagen PIL en modo '1'
            pil_image = Image.fromarray(monochrome_frame).convert('1')
            frames.append(pil_image)
            
        cap.release()
        print(f"✅ Cargados {len(frames)} frames de '{video_path}'")
        return frames

    def dibujar_idle(self):
        """Dibuja el frame actual de la animación idle (video)."""
        if not self.idle_frames:
            # Dibuja un mensaje de error si no hay frames (ya manejado en load_video_frames,
            # pero es un fallback seguro)
            with canvas(self.device) as draw:
                draw.text((20, 20), "ERROR FATAL FRAMES", fill="white")
            return

        # Seleccionar el frame actual de la animación de video
        current_frame_index = self.frame % len(self.idle_frames)
        frame_image = self.idle_frames[current_frame_index]
        
        # Usamos canvas para dibujar la imagen completa sobre la pantalla
        with canvas(self.device) as draw:
            # draw.paste() pega la imagen PIL monocromática sobre el canvas.
            draw.paste(frame_image, (0, 0))

    def dibujar_figura(self, nombre_figura):
        """Dibuja una figura específica (mantenemos la lógica ASCII original)."""
        # ... Lógica de dibujo ASCII original ...
        if nombre_figura not in self.figuras:
            return False
        
        figura = self.figuras[nombre_figura]
        
        with canvas(self.device) as draw:
            draw.rectangle(self.device.bounding_box, outline="black", fill="black")
            
            # Titulo simple
            draw.text((2, 2), nombre_figura.upper(), fill="white")
            
            # Centrar la figura
            y_start = 18
            for i, linea in enumerate(figura):
                x = (OLED_WIDTH - len(linea) * 6) // 2
                draw.text((x, y_start + i * 10), linea, fill="white")
        
        return True
    
    def mostrar_figura(self, nombre):
        """Cambia al modo figura y muestra por 3 segundos."""
        if nombre in self.figuras:
            self.modo = "figura"
            self.figura_actual = nombre
            self.dibujar_figura(nombre)
            
            # Volver a idle después de 3 segundos
            def volver_idle():
                time.sleep(3)
                self.modo = "idle"
                self.figura_actual = None
            
            threading.Thread(target=volver_idle, daemon=True).start()
            return True
        else:
            print(f"Figura '{nombre}' no encontrada. Disponibles: {list(self.figuras.keys())}")
            return False
    
    def loop_animacion(self):
        """Loop principal de animación (ahora controla el frame rate del video)."""
        while self.running:
            if self.modo == "idle":
                # La velocidad de la animación ahora se controla por FRAME_DELAY
                self.dibujar_idle()
                self.frame += 1
                time.sleep(FRAME_DELAY)
            else:
                time.sleep(0.1)  # En modo figura, solo esperar
    
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
        # Requerir las librerías al inicio
        print("Revisando dependencias: OpenCV (cv2) y PIL son necesarias para este código.")
        
        print("Iniciando OLED animado...")
        oled = AnimatedOLED()
        
        # Iniciar animación
        oled.iniciar()
        
        print("\n=== COMANDOS DISPONIBLES ===")
        print(f"FPS de la animación de video: {ANIMATION_FPS} ({FRAME_DELAY:.2f}s/frame)")
        print("Escribe el nombre de una figura para mostrarla:")
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
        print("\nAsegúrate de tener instaladas las librerías 'opencv-python' y 'Pillow'.")