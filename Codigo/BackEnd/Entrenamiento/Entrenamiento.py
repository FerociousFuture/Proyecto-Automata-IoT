import pandas as pd
import numpy as np
import joblib
import serial
import time
import sys
import os
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN GLOBAL ---
MODEL_PATH = 'gesture_sequence_template.pkl' 
SCALER_PATH = 'scaler_for_prediction.pkl'
SENSOR_COLS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']
NUM_SENSOR_VALUES = 6
HEADER = "Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z\n"

# --- PARÁMETROS DE DETECCIÓN POR SECUENCIA ---
TEMPLATE_LENGTH = 80        # Longitud de la secuencia template (ajustable)
DETECTION_WINDOW = 100      # Ventana máxima de búsqueda en tiempo real
STEP_SIZE = 5               # Evaluación cada 5 muestras (250ms @ 20Hz)
DTW_THRESHOLD = 150.0       # Umbral de similitud DTW (ajustar según pruebas)
MIN_ACTIVITY = 0.08         # Filtro de actividad mínima
COOLDOWN_SAMPLES = 40       # Período de enfriamiento tras detección (2s)

realtime_buffer = []
cooldown_counter = 0

# --- FUNCIÓN DE NORMALIZACIÓN DE SECUENCIA ---
def normalize_sequence(sequence):
    """
    Normaliza una secuencia de datos de sensores usando StandardScaler.
    Preserva la forma temporal del gesto.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(sequence)

# --- DYNAMIC TIME WARPING (DTW) OPTIMIZADO ---
def dtw_distance(seq1, seq2):
    """
    Calcula la distancia DTW entre dos secuencias multidimensionales.
    Permite variaciones de velocidad en el gesto.
    """
    n, m = len(seq1), len(seq2)
    
    # Matriz de costos
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Llenar la matriz DTW
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(seq1[i-1], seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # Inserción
                dtw_matrix[i, j-1],      # Eliminación
                dtw_matrix[i-1, j-1]     # Match
            )
    
    return dtw_matrix[n, m]

# --- EXTRACCIÓN DE CARACTERÍSTICAS TEMPORALES ---
def extract_temporal_features(data):
    """
    Extrae características que preservan la información temporal del gesto.
    Incluye velocidades angulares y aceleraciones.
    """
    data_df = pd.DataFrame(data, columns=SENSOR_COLS)
    
    # Magnitudes (energía del movimiento)
    gyro_mag = np.sqrt(data_df['Gyro_X']**2 + data_df['Gyro_Y']**2 + data_df['Gyro_Z']**2)
    acc_mag = np.sqrt(data_df['Acc_X']**2 + data_df['Acc_Y']**2 + data_df['Acc_Z']**2)
    
    # Combinamos todo en una matriz temporal
    temporal_features = np.column_stack([
        data_df[SENSOR_COLS].values,
        gyro_mag.values.reshape(-1, 1),
        acc_mag.values.reshape(-1, 1)
    ])
    
    return temporal_features

# --- FASE 1: ENTRENAMIENTO - CREA TEMPLATE DE SECUENCIA ---
def train_model(csv_file):
    """
    Lee el CSV de entrenamiento y crea un TEMPLATE de secuencia.
    Este template representa el "camino ideal" del gesto.
    """
    try:
        df_gesture = pd.read_csv(csv_file)
        print(f"✅ Datos de Gesto cargados de '{csv_file}' ({len(df_gesture)} muestras).")
    except FileNotFoundError:
        print(f"❌ Error: Archivo CSV '{csv_file}' no encontrado.")
        return

    if len(df_gesture) < TEMPLATE_LENGTH:
        print(f"❌ Error: Se necesitan al menos {TEMPLATE_LENGTH} muestras. Archivo tiene {len(df_gesture)}.")
        return

    # 1. CALCULAR ACTIVIDAD Y SEGMENTAR EL GESTO PRINCIPAL
    df_gesture['Activity'] = df_gesture[SENSOR_COLS].std(axis=1)
    active_segments = df_gesture[df_gesture['Activity'] > MIN_ACTIVITY]
    
    if len(active_segments) < TEMPLATE_LENGTH:
        print("⚠️  Advertencia: Pocas muestras activas. Usando datos completos.")
        active_segments = df_gesture
    
    # 2. EXTRAER LA SECUENCIA CENTRAL MÁS REPRESENTATIVA
    # Buscamos el segmento con mayor actividad sostenida
    best_start = 0
    max_avg_activity = 0
    
    for i in range(len(df_gesture) - TEMPLATE_LENGTH):
        window = df_gesture.iloc[i:i+TEMPLATE_LENGTH]
        avg_activity = window['Activity'].mean()
        if avg_activity > max_avg_activity:
            max_avg_activity = avg_activity
            best_start = i
    
    template_segment = df_gesture.iloc[best_start:best_start+TEMPLATE_LENGTH]
    
    # 3. EXTRAER CARACTERÍSTICAS TEMPORALES
    template_features = extract_temporal_features(template_segment[SENSOR_COLS])
    
    # 4. NORMALIZAR LA SECUENCIA TEMPLATE
    template_normalized = normalize_sequence(template_features)
    
    # 5. CREAR MÚLTIPLES VARIACIONES DEL TEMPLATE (para robustez)
    templates = [template_normalized]
    
    # Variación con diferentes segmentos si hay suficientes datos
    if len(df_gesture) >= TEMPLATE_LENGTH * 3:
        mid_start = len(df_gesture) // 2 - TEMPLATE_LENGTH // 2
        mid_segment = df_gesture.iloc[mid_start:mid_start+TEMPLATE_LENGTH]
        mid_features = extract_temporal_features(mid_segment[SENSOR_COLS])
        templates.append(normalize_sequence(mid_features))
    
    # 6. GUARDAR TEMPLATES Y METADATOS
    template_data = {
        'templates': templates,
        'template_length': TEMPLATE_LENGTH,
        'avg_activity': max_avg_activity,
        'sensor_cols': SENSOR_COLS
    }
    
    joblib.dump(template_data, MODEL_PATH)
    
    print("\n" + "="*70)
    print("✅ TEMPLATE DE SECUENCIA CREADO CON ÉXITO")
    print(f"   Longitud del Template: {TEMPLATE_LENGTH} muestras")
    print(f"   Actividad Promedio: {max_avg_activity:.4f}")
    print(f"   Número de Templates: {len(templates)}")
    print(f"   Archivo guardado: '{MODEL_PATH}'")
    print(f"   Umbral DTW: {DTW_THRESHOLD} (ajusta si hay falsos positivos/negativos)")
    print("="*70)
    return True

# --- FASE 2: DETECCIÓN EN TIEMPO REAL CON DTW ---
def run_detector(serial_port, baud_rate):
    """
    Detecta el gesto comparando la secuencia en tiempo real con el template
    usando Dynamic Time Warping (DTW).
    """
    global realtime_buffer, cooldown_counter
    
    try:
        template_data = joblib.load(MODEL_PATH)
        templates = template_data['templates']
        template_length = template_data['template_length']
        print(f"✅ Templates cargados: {len(templates)} variaciones")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo del modelo. Ejecuta 'train' primero.")
        return

    print(f"📡 Conectando a {serial_port} @ {baud_rate}...")
    
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print("🟢 Detector de Secuencia ACTIVO")
        print(f"🎯 Esperando gesto... (DTW Threshold: {DTW_THRESHOLD})")
    except serial.SerialException as e:
        print(f"❌ Error al abrir '{serial_port}': {e}")
        return

    evaluation_counter = 0
    
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('latin-1').strip()
                parts = line.split(',')
                
                if len(parts) == NUM_SENSOR_VALUES:
                    try:
                        sensor_values = [float(p.strip()) for p in parts]
                        realtime_buffer.append(sensor_values)
                        
                        # Mantener ventana de detección
                        if len(realtime_buffer) > DETECTION_WINDOW:
                            realtime_buffer = realtime_buffer[-DETECTION_WINDOW:]
                        
                        # Actualizar cooldown
                        if cooldown_counter > 0:
                            cooldown_counter -= 1
                            continue
                        
                        # Evaluar cada STEP_SIZE muestras
                        evaluation_counter += 1
                        if evaluation_counter >= STEP_SIZE and len(realtime_buffer) >= template_length:
                            evaluation_counter = 0
                            
                            # 1. VERIFICAR ACTIVIDAD MÍNIMA
                            recent_data = pd.DataFrame(
                                realtime_buffer[-template_length:], 
                                columns=SENSOR_COLS
                            )
                            current_activity = recent_data.std().mean()
                            
                            if current_activity < MIN_ACTIVITY:
                                sys.stdout.write(f"\r⚪ Reposo (Actividad: {current_activity:.3f})   ")
                                sys.stdout.flush()
                                continue
                            
                            # 2. EXTRAER CARACTERÍSTICAS DE LA VENTANA ACTUAL
                            current_features = extract_temporal_features(
                                realtime_buffer[-template_length:]
                            )
                            current_normalized = normalize_sequence(current_features)
                            
                            # 3. CALCULAR DTW CON TODOS LOS TEMPLATES
                            min_distance = float('inf')
                            for template in templates:
                                distance = dtw_distance(current_normalized, template)
                                if distance < min_distance:
                                    min_distance = distance
                            
                            # 4. EVALUAR SIMILITUD
                            similarity_score = max(0, 100 - (min_distance / DTW_THRESHOLD * 100))
                            
                            if min_distance <= DTW_THRESHOLD:
                                print(f"\n\n🎉 ¡GESTO DETECTADO! ✨")
                                print(f"   Distancia DTW: {min_distance:.2f}")
                                print(f"   Similitud: {similarity_score:.1f}%")
                                print(f"   Actividad: {current_activity:.3f}")
                                
                                # Activar cooldown y limpiar buffer
                                cooldown_counter = COOLDOWN_SAMPLES
                                realtime_buffer = []
                            else:
                                sys.stdout.write(
                                    f"\r🔍 Analizando... "
                                    f"DTW: {min_distance:.1f} | "
                                    f"Similitud: {similarity_score:.1f}% | "
                                    f"Act: {current_activity:.3f}   "
                                )
                                sys.stdout.flush()
                            
                    except ValueError:
                        continue
            
            time.sleep(0.001)
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Deteniendo detector...")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            break

    ser.close()
    print("Conexión cerrada.")

# --- FASE 3: RECOLECCIÓN DE DATOS ---
def collect_data(output_file, repetitions, serial_port, baud_rate):
    """
    Recolecta datos para entrenamiento con control ENTER/Ctrl+C.
    """
    print(f"📡 Conectando a {serial_port} @ {baud_rate}...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print("🟢 Conexión establecida.")
    except serial.SerialException as e:
        print(f"❌ Error: {e}")
        return

    data_count = 0
    
    with open(output_file, 'w') as f:
        f.write(HEADER)
        print(f"\n📝 Recolección iniciada. Objetivo: {repetitions} repeticiones")
        print(f"⚠️  IMPORTANTE: Realiza el gesto COMPLETO de forma CONSISTENTE\n")
        
        for rep in range(1, repetitions + 1):
            input(f"Repetición {rep}/{repetitions} - Presiona ENTER para INICIAR...")
            print("🔴 GRABANDO... (Ctrl+C para detener)")
            
            rep_samples = 0
            
            try:
                while True:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('latin-1').strip()
                        parts = line.split(',')
                        
                        if len(parts) == NUM_SENSOR_VALUES:
                            f.write(line + '\n')
                            rep_samples += 1
                            data_count += 1
                            sys.stdout.write(
                                f"\rRep {rep}: {rep_samples} muestras | "
                                f"Total: {data_count} (Ctrl+C para finalizar)"
                            )
                            sys.stdout.flush()
                    
                    time.sleep(0.01)
                    
            except KeyboardInterrupt:
                pass
            
            print(f"\n✅ Repetición {rep} completada: {rep_samples} muestras")
            input("Presiona ENTER para continuar...\n")

    ser.close()
    print(f"\n{'='*60}")
    print(f"✅ RECOLECCIÓN FINALIZADA")
    print(f"   Total de muestras: {data_count}")
    print(f"   Archivo: '{output_file}'")
    print(f"   Recomendado: Mínimo {TEMPLATE_LENGTH} muestras por repetición")
    print(f"{'='*60}")

# --- UTILIDAD: VISUALIZAR TEMPLATE ---
def visualize_template():
    """
    Muestra información del template entrenado.
    """
    try:
        template_data = joblib.load(MODEL_PATH)
        templates = template_data['templates']
        
        print("\n" + "="*60)
        print("📊 INFORMACIÓN DEL TEMPLATE")
        print("="*60)
        print(f"Número de templates: {len(templates)}")
        print(f"Longitud: {template_data['template_length']} muestras")
        print(f"Dimensiones: {templates[0].shape[1]} características")
        print(f"Actividad promedio: {template_data['avg_activity']:.4f}")
        print(f"Umbral DTW actual: {DTW_THRESHOLD}")
        print("="*60)
        
    except FileNotFoundError:
        print("❌ No se encontró el template. Ejecuta 'train' primero.")

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🪄 SISTEMA DE DETECCIÓN DE GESTOS POR SECUENCIA TEMPORAL")
        print("="*70)
        print("\nCOMANDOS DISPONIBLES:")
        print("\n1. RECOLECTAR DATOS:")
        print("   python Entrenamiento.py collect <salida.csv> <PUERTO> <REPS>")
        print("   Ejemplo: python Entrenamiento.py collect hechizo.csv /dev/ttyUSB0 10")
        
        print("\n2. ENTRENAR TEMPLATE:")
        print("   python Entrenamiento.py train <archivo.csv>")
        print("   Ejemplo: python Entrenamiento.py train hechizo.csv")
        
        print("\n3. DETECTAR GESTO:")
        print("   python Entrenamiento.py detect <PUERTO> <BAUD>")
        print("   Ejemplo: python Entrenamiento.py detect /dev/ttyUSB0 115200")
        
        print("\n4. VER INFORMACIÓN DEL TEMPLATE:")
        print("   python Entrenamiento.py info")
        print("="*70 + "\n")
        
    elif sys.argv[1] == 'collect' and len(sys.argv) >= 5:
        output = sys.argv[2]
        port = sys.argv[3]
        reps = int(sys.argv[4])
        collect_data(output, reps, port, 115200)

    elif sys.argv[1] == 'train' and len(sys.argv) == 3:
        train_model(sys.argv[2])
    
    elif sys.argv[1] == 'detect' and len(sys.argv) >= 3:
        port = sys.argv[2]
        baud = int(sys.argv[3]) if len(sys.argv) == 4 else 115200
        run_detector(port, baud)
    
    elif sys.argv[1] == 'info':
        visualize_template()
    
    else:
        print("❌ Comando no reconocido. Usa sin argumentos para ver la ayuda.")