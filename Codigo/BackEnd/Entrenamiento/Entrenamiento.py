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
# Usamos rutas absolutas para que funcione bien tanto desde terminal como desde Flask
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'gesture_models') 

SENSOR_COLS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']
NUM_SENSOR_VALUES = 6
HEADER = "Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z\n"

# --- PARÁMETROS DE DETECCIÓN ---
TEMPLATE_LENGTH = 80        # Longitud de la secuencia template
DETECTION_WINDOW = 100      # Ventana de búsqueda en tiempo real
STEP_SIZE = 5               # Paso de evaluación (cada 5 muestras)
DTW_THRESHOLD = 190.0       # Umbral de similitud (menor es más estricto)
MIN_ACTIVITY = 0.08         # Filtro de ruido/reposo
COOLDOWN_SAMPLES = 40       # Tiempo de espera tras detección
SIMILARITY_MARGIN = 30.0    # Margen para diferenciar entre gestos similares

realtime_buffer = []
cooldown_counter = 0

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --- FUNCIONES DE MATEMÁTICAS Y PROCESAMIENTO ---

def normalize_sequence(sequence):
    """Normaliza los datos para que la escala no afecte la comparación."""
    scaler = StandardScaler()
    return scaler.fit_transform(sequence)

def dtw_distance(seq1, seq2):
    """Dynamic Time Warping: Compara dos secuencias temporales."""
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
    """Calcula magnitudes y combina con datos crudos."""
    data_df = pd.DataFrame(data, columns=SENSOR_COLS)
    gyro_mag = np.sqrt(data_df['Gyro_X']**2 + data_df['Gyro_Y']**2 + data_df['Gyro_Z']**2)
    acc_mag = np.sqrt(data_df['Acc_X']**2 + data_df['Acc_Y']**2 + data_df['Acc_Z']**2)
    temporal_features = np.column_stack([
        data_df[SENSOR_COLS].values,
        gyro_mag.values.reshape(-1, 1),
        acc_mag.values.reshape(-1, 1)
    ])
    return temporal_features

def clean_and_validate_csv(df):
    """Limpia filas corruptas o con valores extremos del CSV."""
    original_len = len(df)
    for col in SENSOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_clean = df.dropna(subset=SENSOR_COLS)
    
    # Filtro de valores extremos (ruido eléctrico)
    for col in SENSOR_COLS:
        if 'Gyro' in col:
            df_clean = df_clean[(df_clean[col] >= -3000) & (df_clean[col] <= 3000)]
        else: 
            df_clean = df_clean[(df_clean[col] >= -200) & (df_clean[col] <= 200)]
    
    if len(df_clean) < original_len:
        print(f"⚠️  Limpieza: Se eliminaron {original_len - len(df_clean)} filas corruptas.")
    
    return df_clean

# --- GESTIÓN DE ARCHIVOS Y MODELOS ---

def get_gesture_path(gesture_name):
    return os.path.join(MODELS_DIR, f"{gesture_name}.pkl")

def list_trained_gestures():
    if not os.path.exists(MODELS_DIR):
        return []
    return [f.replace('.pkl', '') for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]

def load_gesture(gesture_name):
    path = get_gesture_path(gesture_name)
    if not os.path.exists(path): return None
    return joblib.load(path)

def load_all_gestures():
    gestures = {}
    for name in list_trained_gestures():
        gestures[name] = load_gesture(name)
    return gestures

def delete_gesture(gesture_name):
    path = get_gesture_path(gesture_name)
    if os.path.exists(path):
        os.remove(path)
        print(f"✅ Gesto '{gesture_name}' eliminado.")
    else:
        print(f"❌ Gesto '{gesture_name}' no encontrado.")

# --- FASE 1: ENTRENAMIENTO (RESTORED) ---

def train_model(csv_file, gesture_name=None):
    if gesture_name is None:
        gesture_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ CSV cargado: {len(df)} filas.")
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")
        return

    df = clean_and_validate_csv(df)
    
    if len(df) < TEMPLATE_LENGTH:
        print(f"❌ Error: Insuficientes datos ({len(df)}). Mínimo requerido: {TEMPLATE_LENGTH}")
        return

    # Buscar el segmento con mayor actividad
    df['Activity'] = df[SENSOR_COLS].std(axis=1)
    best_start = 0
    max_activity = 0
    
    for i in range(len(df) - TEMPLATE_LENGTH):
        window = df.iloc[i:i+TEMPLATE_LENGTH]
        avg_act = window['Activity'].mean()
        if avg_act > max_activity:
            max_activity = avg_act
            best_start = i
    
    # Extraer y procesar
    segment = df.iloc[best_start:best_start+TEMPLATE_LENGTH]
    features = extract_temporal_features(segment[SENSOR_COLS])
    template_norm = normalize_sequence(features)
    
    # Guardar
    data = {
        'gesture_name': gesture_name,
        'templates': [template_norm], # Lista para soportar múltiples variaciones a futuro
        'template_length': TEMPLATE_LENGTH,
        'avg_activity': max_activity,
        'trained_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(data, get_gesture_path(gesture_name))
    print(f"🎉 Modelo guardado: {gesture_name.upper()} (Actividad: {max_activity:.2f})")

# --- FASE 2: DETECCIÓN EN TIEMPO REAL (HYBRID: CLI + WEB) ---

def run_detector(serial_port, baud_rate, target_gestures=None, message_callback=None, data_callback=None):
    global realtime_buffer, cooldown_counter
    
    # Sistema de Logs Híbrido (Consola + Web)
    def emit_log(text, type="info", data=None):
        # 1. Salida a Consola
        if type == "gesture": print(f"\n✨ HECHIZO DETECTADO: {text}")
        elif type == "error": print(f"❌ {text}")
        else: print(f"[{type}] {text}")
        
        # 2. Salida a Web (si existe callback)
        if message_callback:
            msg = {"text": text, "type": type, "data": data, "timestamp": time.time()}
            message_callback(json.dumps(msg))

    # Carga de modelos
    if target_gestures:
        gestures = {name: load_gesture(name) for name in target_gestures if load_gesture(name)}
    else:
        gestures = load_all_gestures()
        
    if not gestures:
        emit_log("No hay modelos cargados. Usa 'train' primero.", "error")
        return

    template_len = list(gestures.values())[0]['template_length']
    
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        emit_log(f"Detector iniciado en {serial_port}. Gestos: {len(gestures)}", "success")
    except Exception as e:
        emit_log(f"Error Serial: {e}", "error")
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
                        
                        # --- ENVIAR DATOS CRUDOS A LA WEB (DIBUJO) ---
                        if data_callback:
                            data_packet = {
                                "gx": vals[0], "gy": vals[1], "gz": vals[2],
                                "ax": vals[3], "ay": vals[4], "az": vals[5]
                            }
                            data_callback(data_packet)
                        # ---------------------------------------------

                        realtime_buffer.append(vals)
                        if len(realtime_buffer) > DETECTION_WINDOW: 
                            realtime_buffer.pop(0)
                        
                        if cooldown_counter > 0:
                            cooldown_counter -= 1
                            continue
                            
                        evaluation_ctr += 1
                        # Evaluar solo si tenemos datos suficientes y toca turno
                        if evaluation_ctr >= STEP_SIZE and len(realtime_buffer) >= template_len:
                            evaluation_ctr = 0
                            
                            # Análisis de Actividad (ahorra CPU si está quieto)
                            recent = pd.DataFrame(realtime_buffer[-template_len:], columns=SENSOR_COLS)
                            if recent.std().mean() < MIN_ACTIVITY: continue
                            
                            # Procesamiento
                            feats = extract_temporal_features(realtime_buffer[-template_len:])
                            curr_seq = normalize_sequence(feats)
                            
                            # Comparación DTW
                            best_name = None
                            min_dist = float('inf')
                            
                            for name, model in gestures.items():
                                for temp in model['templates']:
                                    d = dtw_distance(curr_seq, temp)
                                    if d < min_dist:
                                        min_dist = d
                                        best_name = name
                            
                            # Validación de Umbral
                            if min_dist <= DTW_THRESHOLD:
                                # Calcular confianza (100% = distancia 0)
                                confidence = max(0, 100 - (min_dist / DTW_THRESHOLD * 100))
                                
                                emit_log(best_name, "gesture", {"name": best_name, "score": confidence})
                                cooldown_counter = COOLDOWN_SAMPLES
                                realtime_buffer = []

                except ValueError: pass
            
            # Pequeño sleep para no saturar CPU, pero bajo para fluidez del dibujo
            time.sleep(0.002)

    except KeyboardInterrupt:
        emit_log("Detector detenido.", "system")
    finally:
        if 'ser' in locals() and ser.is_open: ser.close()

# --- FASE 3: RECOLECCIÓN DE DATOS (RESTORED) ---

def collect_data(output_file, repetitions, serial_port, baud_rate):
    print(f"📡 Conectando a {serial_port}...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print("🟢 Listo para grabar.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    with open(output_file, 'w') as f:
        f.write(HEADER)
        print(f"\n📝 Grabando {repetitions} repeticiones en '{output_file}'")
        
        for i in range(repetitions):
            input(f"Repetición {i+1}/{repetitions} - Presiona ENTER para empezar...")
            print("🔴 GRABANDO... (Ctrl+C para parar esta toma)")
            
            samples = 0
            try:
                while True:
                    if ser.in_waiting:
                        line = ser.readline().decode('latin-1').strip()
                        if line.count(',') == 5: # Validación básica CSV
                            f.write(line + '\n')
                            samples += 1
                            sys.stdout.write(f"\rMuestras: {samples}")
                            sys.stdout.flush()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                print(f"\n✅ Repetición {i+1} guardada.")
                pass
                
    print("\n🎉 Colección finalizada.")
    ser.close()

# --- UTILITIES CLI ---

def visualize_template(gesture_name=None):
    if gesture_name:
        data = load_gesture(gesture_name)
        if data:
            print(f"\n📊 GESTO: {data['gesture_name']}")
            print(f"   Fecha: {data['trained_date']}")
            print(f"   Samples: {data['template_length']}")
        else:
            print("❌ No encontrado.")
    else:
        gestures = list_trained_gestures()
        print(f"\n📚 Gestos disponibles ({len(gestures)}):")
        for g in gestures: print(f" - {g}")

# --- PUNTO DE ENTRADA PRINCIPAL (CLI) ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n🪄 GESTOR DE HECHIZOS IOT")
        print("1. collect <archivo.csv> <PUERTO> <REPS>")
        print("2. train   <archivo.csv> [nombre]")
        print("3. detect  <PUERTO>")
        print("4. list")
        print("5. info    [nombre]")
        print("6. delete  <nombre>\n")
        exit()

    cmd = sys.argv[1]
    
    if cmd == 'collect':
        collect_data(sys.argv[2], int(sys.argv[4]), sys.argv[3], 115200)
    elif cmd == 'train':
        name = sys.argv[3] if len(sys.argv) > 3 else None
        train_model(sys.argv[2], name)
    elif cmd == 'detect':
        port = sys.argv[2] if len(sys.argv) > 2 else '/dev/ttyUSB0'
        run_detector(port, 115200)
    elif cmd == 'list':
        visualize_template()
    elif cmd == 'info':
        visualize_template(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == 'delete':
        delete_gesture(sys.argv[2])
    else:
        print("❌ Comando desconocido.")