import pandas as pd
import numpy as np
import joblib
import serial
import time
import sys
import os
import json
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN GLOBAL ---
# Ajusta esta ruta si es necesario para que apunte a donde están tus .pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'gesture_models') 

SENSOR_COLS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']
NUM_SENSOR_VALUES = 6
HEADER = "Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z\n"

# --- PARÁMETROS DE DETECCIÓN ---
TEMPLATE_LENGTH = 80        
DETECTION_WINDOW = 100      
STEP_SIZE = 5               
DTW_THRESHOLD = 150.0       
MIN_ACTIVITY = 0.08         
COOLDOWN_SAMPLES = 40       

realtime_buffer = []
cooldown_counter = 0

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --- FUNCIONES DE UTILIDAD (Normalización, DTW, Limpieza) ---
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
    gesture_path = os.path.join(MODELS_DIR, f"{gesture_name}.pkl")
    if not os.path.exists(gesture_path):
        return None
    return joblib.load(gesture_path)

def load_all_gestures():
    gestures = {}
    if not os.path.exists(MODELS_DIR):
        return gestures
        
    files = [f.replace('.pkl', '') for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    for name in files:
        gestures[name] = load_gesture(name)
    return gestures

# --- FASE 2: DETECCIÓN EN TIEMPO REAL (OPTIMIZADA PARA WEB) ---
def run_detector(serial_port, baud_rate, target_gestures=None, message_callback=None):
    global realtime_buffer, cooldown_counter
    
    # Función auxiliar para enviar mensajes a la web y consola
    def emit_log(text_msg, type="info", data=None):
        print(f"[{type.upper()}] {text_msg}")
        if message_callback:
            # Enviamos un objeto JSON serializado para fácil parseo en JS
            payload = {
                "text": text_msg,
                "type": type,
                "data": data,
                "timestamp": time.time()
            }
            message_callback(json.dumps(payload))

    # Cargar modelos
    if target_gestures is None:
        all_gestures = load_all_gestures()
    else:
        all_gestures = {}
        for gesture_name in target_gestures:
            gd = load_gesture(gesture_name)
            if gd: all_gestures[gesture_name] = gd
    
    if not all_gestures:
        emit_log("No se encontraron modelos de gestos (.pkl) en la carpeta.", "error")
        return

    # Usamos la longitud del template del primer gesto cargado como referencia
    template_length = list(all_gestures.values())[0]['template_length']
    
    emit_log(f"Intentando conectar al puerto {serial_port}...", "system")
    
    try:
        # Intentar abrir puerto serie
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2) # Esperar reinicio de Arduino/ESP
        ser.flushInput()
        emit_log("Detector ACTIVO. Realiza movimientos con la varita.", "success")
    except Exception as e:
        emit_log(f"No se pudo conectar al hardware: {e}", "error")
        # Modo simulación para pruebas si no hay hardware (opcional)
        return

    evaluation_counter = 0
    
    # Bucle principal de lectura
    try:
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('latin-1').strip()
                    # Esperamos 6 valores: Gx, Gy, Gz, Ax, Ay, Az
                    parts = line.split(',')
                    
                    if len(parts) == NUM_SENSOR_VALUES:
                        try:
                            vals = [float(p) for p in parts]
                            realtime_buffer.append(vals)
                            
                            # Mantener buffer limpio
                            if len(realtime_buffer) > DETECTION_WINDOW: 
                                realtime_buffer.pop(0)
                            
                            # Si estamos en enfriamiento (acabamos de detectar algo), esperar
                            if cooldown_counter > 0:
                                cooldown_counter -= 1
                                continue
                            
                            evaluation_counter += 1
                            
                            # Evaluar cada STEP_SIZE muestras si tenemos suficientes datos
                            if evaluation_counter >= STEP_SIZE and len(realtime_buffer) >= template_length:
                                evaluation_counter = 0
                                
                                # Análisis rápido de actividad (si está quieto, no procesar)
                                recent_df = pd.DataFrame(realtime_buffer[-template_length:], columns=SENSOR_COLS)
                                activity = recent_df.std().mean()
                                
                                if activity < MIN_ACTIVITY: 
                                    continue 
                                
                                # Extraer características y normalizar
                                current_features = extract_temporal_features(realtime_buffer[-template_length:])
                                current_seq_norm = normalize_sequence(current_features)
                                
                                # Comparar con todos los gestos cargados usando DTW
                                best_gesture = None
                                min_dist = float('inf')
                                
                                for name, g_data in all_gestures.items():
                                    for template in g_data['templates']:
                                        d = dtw_distance(current_seq_norm, template)
                                        if d < min_dist:
                                            min_dist = d
                                            best_gesture = name
                                
                                # Verificar umbral
                                if min_dist <= DTW_THRESHOLD:
                                    # Calcular porcentaje de confianza (aproximado)
                                    confidence = max(0, 100 - (min_dist / DTW_THRESHOLD * 100))
                                    
                                    # ¡HECHIZO DETECTADO!
                                    emit_log(
                                        f"Gesto detectado: {best_gesture} ({confidence:.1f}%)", 
                                        "gesture", 
                                        {"name": best_gesture, "score": confidence}
                                    )
                                    
                                    # Reiniciar buffer y poner enfriamiento
                                    cooldown_counter = COOLDOWN_SAMPLES
                                    realtime_buffer = []
                                    
                        except ValueError:
                            pass # Error de parseo de float, ignorar linea
            except UnicodeDecodeError:
                pass # Error de lectura serial, ignorar
                
            time.sleep(0.005) # Pequeña pausa para no saturar CPU
                
    except KeyboardInterrupt:
        emit_log("Detector detenido manualmente.", "system")
    except Exception as e:
        emit_log(f"Error crítico en detector: {e}", "error")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            emit_log("Conexión Serial cerrada.", "system")

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🪄 SISTEMA DE DETECCIÓN MULTI-GESTO CON SECUENCIA TEMPORAL")
        print("="*70)
        print("\nCOMANDOS DISPONIBLES:")
        
        print("\n1. RECOLECTAR DATOS PARA UN GESTO:")
        print("   python Entrenamiento.py collect <salida.csv> <PUERTO> <REPS>")
        print("   Ejemplo: python Entrenamiento.py collect lumos.csv /dev/ttyUSB0 10")
        
        print("\n2. ENTRENAR UN GESTO ESPECÍFICO:")
        print("   python Entrenamiento.py train <archivo.csv> [nombre_gesto]")
        print("   Ejemplo: python Entrenamiento.py train lumos.csv lumos")
        print("   (Si no especificas nombre, usa el nombre del archivo)")
        
        print("\n3. DETECTAR GESTOS:")
        print("   a) Detectar TODOS los gestos entrenados:")
        print("      python Entrenamiento.py detect <PUERTO> <BAUD>")
        print("      Ejemplo: python Entrenamiento.py detect /dev/ttyUSB0 115200")
        print("\n   b) Detectar SOLO gestos específicos:")
        print("      python Entrenamiento.py detect <PUERTO> <BAUD> <gesto1> [gesto2] ...")
        print("      Ejemplo: python Entrenamiento.py detect /dev/ttyUSB0 115200 lumos expelliarmus")
        
        print("\n4. VER INFORMACIÓN DE GESTOS:")
        print("   a) Ver todos los gestos:")
        print("      python Entrenamiento.py info")
        print("   b) Ver un gesto específico:")
        print("      python Entrenamiento.py info <nombre_gesto>")
        print("      Ejemplo: python Entrenamiento.py info lumos")
        
        print("\n5. LISTAR GESTOS ENTRENADOS:")
        print("   python Entrenamiento.py list")
        
        print("\n6. ELIMINAR UN GESTO:")
        print("   python Entrenamiento.py delete <nombre_gesto>")
        print("   Ejemplo: python Entrenamiento.py delete lumos")
        
        print("\n" + "="*70)
        print("💡 FLUJO DE TRABAJO RECOMENDADO:")
        print("   1. Recolecta datos: collect hechizo1.csv /dev/ttyUSB0 10")
        print("   2. Entrena el gesto: train hechizo1.csv hechizo1")
        print("   3. Repite para más gestos (hechizo2, hechizo3, etc.)")
        print("   4. Detecta todos: detect /dev/ttyUSB0 115200")
        print("="*70 + "\n")
        
    elif sys.argv[1] == 'collect' and len(sys.argv) >= 5:
        output = sys.argv[2]
        port = sys.argv[3]
        reps = int(sys.argv[4])
        collect_data(output, reps, port, 115200)

    elif sys.argv[1] == 'train':
        if len(sys.argv) == 3:
            # Solo archivo CSV, nombre automático
            train_model(sys.argv[2])
        elif len(sys.argv) == 4:
            # Archivo CSV + nombre personalizado
            train_model(sys.argv[2], sys.argv[3])
        else:
            print("❌ Uso: train <archivo.csv> [nombre_gesto]")
    
    elif sys.argv[1] == 'detect':
        if len(sys.argv) >= 3:
            port = sys.argv[2]
            baud = int(sys.argv[3]) if len(sys.argv) >= 4 and sys.argv[3].isdigit() else 115200
            
            # Verificar si hay gestos específicos
            gesture_start_index = 4 if len(sys.argv) >= 4 and sys.argv[3].isdigit() else 3
            target_gestures = sys.argv[gesture_start_index:] if len(sys.argv) > gesture_start_index else None
            
            run_detector(port, baud, target_gestures)
        else:
            print("❌ Uso: detect <PUERTO> <BAUD> [gesto1] [gesto2] ...")
    
    elif sys.argv[1] == 'info':
        if len(sys.argv) == 2:
            visualize_template()  # Mostrar todos
        else:
            visualize_template(sys.argv[2])  # Mostrar específico
    
    elif sys.argv[1] == 'list':
        gestures = list_trained_gestures()
        if gestures:
            print("\n📚 Gestos entrenados:")
            for i, gesture in enumerate(gestures, 1):
                print(f"   {i}. {gesture}")
            print(f"\nTotal: {len(gestures)} gesto(s)")
        else:
            print("❌ No hay gestos entrenados.")
    
    elif sys.argv[1] == 'delete' and len(sys.argv) == 3:
        delete_gesture(sys.argv[2])
    
    else:
        print("❌ Comando no reconocido. Usa sin argumentos para ver la ayuda.")