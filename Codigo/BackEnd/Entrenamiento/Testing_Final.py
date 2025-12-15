#!/usr/bin/env python3
"""
Sistema Integrado de Detección de Gestos con Feedback Visual y Auditivo
Combina el detector de gestos con control de OLED y Buzzer
"""

import pandas as pd
import numpy as np
import joblib
import serial
import time
import sys
import os
import json
import threading
import RPi.GPIO as GPIO
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2
import random

# Importaciones OLED
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306


# =======================================================
# CONFIGURACIÓN GLOBAL DEL SISTEMA
# =======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'gesture_models')
CARAS_DIR = os.path.join(BASE_DIR, 'Caras')

# Configuración del Sensor
SENSOR_COLS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']
NUM_SENSOR_VALUES = 6

# Parámetros de Detección
TEMPLATE_LENGTH = 80
DETECTION_WINDOW = 100
STEP_SIZE = 5
DTW_THRESHOLD = 190.0
MIN_ACTIVITY = 0.08
COOLDOWN_SAMPLES = 40

# Configuración OLED
OLED_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64
ANIMATION_FPS = 20
FRAME_DELAY = 1.0 / ANIMATION_FPS
MIN_IDLE_TIME = 5.0
MAX_IDLE_TIME = 10.0

# Configuración Buzzer
BUZZER_PIN = 17

# Buffer global para detección en tiempo real
realtime_buffer = []
cooldown_counter = 0


# =======================================================
# MAPEO DE GESTOS A ACCIONES
# =======================================================

GESTURE_ACTIONS = {
    'Lumos_Nox': {
        'oled_mode': 'ascii',
        'oled_art': 'bombilla',
        'buzzer': 'lumos',
        'description': 'Luz ON/OFF'
    },
    'Wingardium_Leviosa': {
        'oled_mode': 'ascii',
        'oled_art': 'pluma',
        'buzzer': 'levitation',
        'description': 'Levitación'
    },
    'Ascendio': {
        'oled_mode': 'ascii',
        'oled_art': 'flecha_arriba',
        'buzzer': 'check',
        'description': 'Subir volumen'
    },
    'Descendio': {
        'oled_mode': 'ascii',
        'oled_art': 'flecha_abajo',
        'buzzer': 'check',
        'description': 'Bajar volumen'
    },
    'Stupefy': {
        'oled_mode': 'ascii',
        'oled_art': 'rayo',
        'buzzer': 'stupefy',
        'description': 'Aturdimiento'
    },
    'Reparo': {
        'oled_mode': 'ascii',
        'oled_art': 'herramienta',
        'buzzer': 'check',
        'description': 'Reparación'
    },
    'Expelliarmus': {
        'oled_mode': 'ascii',
        'oled_art': 'varita_rota',
        'buzzer': 'expelliarmus',
        'description': 'Desarmar'
    },
    'skull': {
        'oled_mode': 'ascii',
        'oled_art': 'calavera',
        'buzzer': 'skull',
        'description': 'Apagado del sistema'
    }
}


# =======================================================
# FUNCIONES DE PROCESAMIENTO DE GESTOS
# =======================================================

def normalize_sequence(sequence):
    scaler = StandardScaler()
    return scaler.fit_transform(sequence)

def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    return dtw_matrix[n, m]

def extract_temporal_features(data):
    data_df = pd.DataFrame(data, columns=SENSOR_COLS)
    gyro_mag = np.sqrt(data_df['Gyro_X']**2 + data_df['Gyro_Y']**2 + data_df['Gyro_Z']**2)
    acc_mag = np.sqrt(data_df['Acc_X']**2 + data_df['Acc_Y']**2 + data_df['Acc_Z']**2)
    temporal_features = np.column_stack([
        data_df[SENSOR_COLS].values,
        gyro_mag.values.reshape(-1, 1),
        acc_mag.values.reshape(-1, 1)
    ])
    return temporal_features

def load_gesture(gesture_name):
    path = os.path.join(MODELS_DIR, f"{gesture_name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_all_gestures():
    gestures = {}
    if not os.path.exists(MODELS_DIR):
        return gestures
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pkl'):
            name = f.replace('.pkl', '')
            gestures[name] = load_gesture(name)
    return gestures


# =======================================================
# CLASE: BUZZER CONTROLLER
# =======================================================

class BuzzerController:
    def __init__(self, pin=BUZZER_PIN):
        self.BUZZER_PIN = pin
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
            self.pwm = GPIO.PWM(self.BUZZER_PIN, 50)
            self.pwm.start(0)
        except Exception as e:
            print(f"⚠️  Error al inicializar Buzzer: {e}")
            self.pwm = None

    def tocar_nota(self, frecuencia, duracion):
        if self.pwm is None:
            return
        try:
            self.pwm.ChangeFrequency(frecuencia)
            self.pwm.ChangeDutyCycle(50)
            time.sleep(duracion)
            self.pwm.ChangeDutyCycle(0)
            time.sleep(0.05)
        except ValueError:
            pass

    def get_duracion_melodia(self, tono):
        melodias = self._obtener_melodias()
        if tono in melodias:
            return sum(duracion for _, duracion in melodias[tono])
        return 0

    def _obtener_melodias(self):
        return {
            'check': [(440, 0.1), (550, 0.1), (660, 0.1)],
            'error': [(660, 0.15), (440, 0.15)],
            'skull': [
                (392, 0.15), (466, 0.15), (523, 0.25), (0, 0.05),
                (392, 0.15), (466, 0.15), (554, 0.12), (523, 0.25), (0, 0.05),
                (392, 0.15), (466, 0.15), (523, 0.25), (0, 0.05),
                (466, 0.15), (392, 0.35)
            ],
            'lumos': [(523, 0.2), (659, 0.2), (784, 0.3)],
            'levitation': [(440, 0.15), (494, 0.15), (523, 0.15), (587, 0.2), (659, 0.2)],
            'stupefy': [(349, 0.2), (330, 0.2), (294, 0.3)],
            'expelliarmus': [(392, 0.15), (523, 0.15), (659, 0.2), (523, 0.15), (392, 0.2)]
        }

    def tocar_reaccion(self, tono='check'):
        melodias = self._obtener_melodias()
        if tono not in melodias:
            return
        for freq, duration in melodias[tono]:
            if freq == 0:
                time.sleep(duration)
            else:
                self.tocar_nota(freq, duration)

    def cleanup(self):
        if self.pwm:
            self.pwm.stop()


# =======================================================
# CLASE: ANIMATED OLED
# =======================================================

class AnimatedOLED:
    def __init__(self):
        serial_interface = i2c(port=1, address=OLED_ADDRESS)
        self.device = ssd1306(serial_interface, width=OLED_WIDTH, height=OLED_HEIGHT)
        
        self.modo = "idle"
        self.figura_actual = None
        self.running = True
        self.frame = 0
        
        # Cargar videos idle y parpadeo
        self.idle_frames = self.load_video_frames(os.path.join(CARAS_DIR, 'idle.mp4'))
        self.blink_frames = self.load_video_frames(os.path.join(CARAS_DIR, 'Parpadeo.mp4'))
        
        if not self.idle_frames:
            self.idle_frames = [Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)]
        
        self.is_blinking = False
        self.blink_frame_counter = 0
        self.next_blink_time = time.time() + self.get_random_idle_time()
        
        # ASCII Art expandido
        self.ascii_art = {
            "bombilla": [
                "   .-.",
                "  (   )",
                "   '-'",
                "    |",
                "  .-+-.",
                "  | O |",
                "  '---'",
                "  LUMOS"
            ],
            "pluma": [
                "      /",
                "     /",
                "    (",
                "   /",
                "  /",
                " /",
                "~",
                "LEVITATION"
            ],
            "flecha_arriba": [
                "    ^",
                "   /_\\",
                "  /   \\",
                "    |",
                "    |",
                "    |",
                "",
                "ASCENDIO"
            ],
            "flecha_abajo": [
                "    |",
                "    |",
                "    |",
                "  \\   /",
                "   \\_/",
                "    v",
                "",
                "DESCENDIO"
            ],
            "rayo": [
                "    __",
                "   |  \\",
                "   |   \\",
                "    \\   \\",
                "     \\  |",
                "      \\_|",
                "",
                "STUPEFY"
            ],
            "herramienta": [
                "  .---.",
                " /     \\",
                "|  (+)  |",
                " \\     /",
                "  '---'",
                "    |",
                "",
                "REPARO"
            ],
            "varita_rota": [
                " /\\",
                "/  \\",
                "    \\",
                "  X",
                " /",
                "/",
                "",
                "EXPELLIAR"
            ],
            "calavera": [
                "  .--.",
                " /    \\",
                "| O  O |",
                " \\  v /",
                "  '||'",
                "   ||",
                "",
                "AVADA K."
            ],
            "check": [
                "       *",
                "      **",
                " *  **",
                "  ***",
                "   *",
                "",
                "OK"
            ],
            "error": [
                " \\   /",
                "  \\ /",
                "   X",
                "  / \\",
                " /   \\",
                "",
                "ERROR"
            ]
        }

    def get_random_idle_time(self):
        return random.uniform(MIN_IDLE_TIME, MAX_IDLE_TIME)

    def load_video_frames(self, video_path):
        if not os.path.exists(video_path):
            print(f"⚠️  Video no encontrado: {video_path}")
            return [Image.new('1', (OLED_WIDTH, OLED_HEIGHT), 0)]
        
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
        print(f"✅ Cargados {len(frames)} frames de '{os.path.basename(video_path)}'")
        return frames

    def dibujar_idle(self):
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

    def dibujar_ascii(self, nombre_art):
        if nombre_art not in self.ascii_art:
            return False
        
        art = self.ascii_art[nombre_art]
        
        with canvas(self.device) as draw:
            draw.rectangle(self.device.bounding_box, outline="black", fill="black")
            
            y_start = (OLED_HEIGHT - len(art) * 8) // 2
            
            for i, linea in enumerate(art):
                # Centrar cada línea
                text_width = len(linea) * 6
                x = (OLED_WIDTH - text_width) // 2
                draw.text((x, y_start + i * 8), linea, fill="white")
        
        return True

    def mostrar_ascii(self, nombre):
        if nombre in self.ascii_art:
            self.modo = "ascii"
            self.is_blinking = False
            self.figura_actual = nombre
            self.dibujar_ascii(nombre)
            threading.Thread(target=self._volver_idle_delay, daemon=True).start()
            return True
        return False

    def _volver_idle_delay(self):
        time.sleep(3)
        if self.modo == "ascii":
            self.modo = "idle"
            self.figura_actual = None
            self.next_blink_time = time.time() + self.get_random_idle_time()

    def loop_animacion(self):
        while self.running:
            start_time = time.time()
            
            if self.modo == "idle":
                current_time = time.time()
                if not self.is_blinking and current_time >= self.next_blink_time and self.blink_frames:
                    self.is_blinking = True
                    self.blink_frame_counter = 0
                
                self.dibujar_idle()
            
            # En modo ASCII no necesitamos actualizar constantemente
            elif self.modo == "ascii":
                time.sleep(0.1)
                continue
            
            elapsed_time = time.time() - start_time
            sleep_time = FRAME_DELAY - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

    def iniciar(self):
        thread = threading.Thread(target=self.loop_animacion, daemon=True)
        thread.start()
        return thread

    def detener(self):
        self.running = False
        time.sleep(0.2)
        self.device.cleanup()


# =======================================================
# FUNCIÓN PRINCIPAL: EJECUTAR ACCIÓN DE GESTO
# =======================================================

def ejecutar_accion_gesto(gesture_name, oled, buzzer):
    """
    Ejecuta la acción correspondiente al gesto detectado.
    """
    if gesture_name not in GESTURE_ACTIONS:
        print(f"⚠️  Gesto '{gesture_name}' sin acción definida")
        return
    
    action = GESTURE_ACTIONS[gesture_name]
    
    print(f"✨ EJECUTANDO: {gesture_name} - {action['description']}")
    
    # Ejecutar OLED
    if action['oled_mode'] == 'ascii':
        oled.mostrar_ascii(action['oled_art'])
    
    # Ejecutar Buzzer en thread separado
    threading.Thread(
        target=buzzer.tocar_reaccion,
        args=(action['buzzer'],),
        daemon=True
    ).start()
    
    # Acciones especiales del sistema
    if gesture_name == 'skull':
        threading.Thread(target=apagar_sistema, daemon=True).start()

def apagar_sistema():
    """Apaga el sistema después de 3 segundos."""
    time.sleep(3)
    print("💀 Iniciando apagado del sistema...")
    os.system("sudo shutdown -h now")


# =======================================================
# DETECTOR EN TIEMPO REAL INTEGRADO
# =======================================================

def run_integrated_detector(serial_port, baud_rate, oled, buzzer):
    global realtime_buffer, cooldown_counter
    
    gestures = load_all_gestures()
    
    if not gestures:
        print("❌ No hay modelos entrenados. Usa el modo de entrenamiento primero.")
        return
    
    template_len = list(gestures.values())[0]['template_length']
    
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print(f"🟢 Detector iniciado en {serial_port}")
        print(f"📊 Gestos cargados: {', '.join(gestures.keys())}")
    except Exception as e:
        print(f"❌ Error Serial: {e}")
        return
    
    evaluation_ctr = 0
    
    try:
        while True:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('latin-1').strip()
                    parts = line.split(',')
                    
                    if len(parts) == NUM_SENSOR_VALUES:
                        vals = [float(p) for p in parts]
                        
                        realtime_buffer.append(vals)
                        if len(realtime_buffer) > DETECTION_WINDOW:
                            realtime_buffer.pop(0)
                        
                        if cooldown_counter > 0:
                            cooldown_counter -= 1
                            continue
                        
                        evaluation_ctr += 1
                        if evaluation_ctr >= STEP_SIZE and len(realtime_buffer) >= template_len:
                            evaluation_ctr = 0
                            
                            recent = pd.DataFrame(realtime_buffer[-template_len:], columns=SENSOR_COLS)
                            if recent.std().mean() < MIN_ACTIVITY:
                                continue
                            
                            feats = extract_temporal_features(realtime_buffer[-template_len:])
                            curr_seq = normalize_sequence(feats)
                            
                            best_name = None
                            min_dist = float('inf')
                            
                            for name, model in gestures.items():
                                for temp in model['templates']:
                                    d = dtw_distance(curr_seq, temp)
                                    if d < min_dist:
                                        min_dist = d
                                        best_name = name
                            
                            if min_dist <= DTW_THRESHOLD:
                                confidence = max(0, 100 - (min_dist / DTW_THRESHOLD * 100))
                                print(f"\n🎯 Gesto detectado: {best_name} (Confianza: {confidence:.1f}%)")
                                
                                ejecutar_accion_gesto(best_name, oled, buzzer)
                                
                                cooldown_counter = COOLDOWN_SAMPLES
                                realtime_buffer = []
                
                except ValueError:
                    pass
            
            time.sleep(0.002)
    
    except KeyboardInterrupt:
        print("\n🛑 Detector detenido por usuario")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()


# =======================================================
# PROGRAMA PRINCIPAL
# =======================================================

if __name__ == "__main__":
    oled = None
    buzzer = None
    
    try:
        print("=" * 50)
        print("🪄 SISTEMA INTEGRADO DE GESTOS MÁGICOS")
        print("=" * 50)
        
        # Inicializar componentes
        print("\n🔧 Inicializando hardware...")
        buzzer = BuzzerController(pin=BUZZER_PIN)
        oled = AnimatedOLED()
        oled.iniciar()
        
        print("✅ Componentes inicializados correctamente")
        print(f"📁 Directorio de modelos: {MODELS_DIR}")
        print(f"📁 Directorio de caras: {CARAS_DIR}")
        
        # Verificar gestos disponibles
        gestures = load_all_gestures()
        if gestures:
            print(f"\n📚 Gestos entrenados ({len(gestures)}):")
            for name in gestures.keys():
                if name in GESTURE_ACTIONS:
                    print(f"  ✓ {name} → {GESTURE_ACTIONS[name]['description']}")
                else:
                    print(f"  ⚠ {name} → Sin acción definida")
        else:
            print("\n⚠️  No se encontraron gestos entrenados")
            print("Por favor, entrena gestos primero usando Entrenamiento.py")
            sys.exit(1)
        
        # Configuración del puerto serial
        serial_port = '/dev/ttyUSB0'
        if len(sys.argv) > 1:
            serial_port = sys.argv[1]
        
        print(f"\n🚀 Iniciando detector en puerto: {serial_port}")
        print("⏸️  Presiona Ctrl+C para detener")
        print("-" * 50)
        
        # Iniciar detector integrado
        run_integrated_detector(serial_port, 115200, oled, buzzer)
        
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🧹 Limpiando recursos...")
        if oled:
            oled.detener()
        if buzzer:
            buzzer.cleanup()
        GPIO.cleanup()
        print("✅ Programa terminado correctamente")