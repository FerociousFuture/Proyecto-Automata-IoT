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
MODELS_DIR = 'gesture_models'  # Directorio para guardar múltiples gestos
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
SIMILARITY_MARGIN = 30.0    # Margen para diferenciar entre gestos (nuevo)

realtime_buffer = []
cooldown_counter = 0

# Crear directorio si no existe
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

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

# --- FUNCIÓN DE LIMPIEZA Y VALIDACIÓN DE DATOS ---
def clean_and_validate_csv(df):
    """
    Limpia y valida el DataFrame eliminando filas con datos corruptos.
    """
    original_len = len(df)
    
    # 1. Intentar convertir todas las columnas a numérico
    for col in SENSOR_COLS:
        if col in df.columns:
            # Forzar conversión, valores inválidos se vuelven NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Eliminar filas con valores NaN
    df_clean = df.dropna(subset=SENSOR_COLS)
    
    # 3. Eliminar filas con valores extremos (outliers severos)
    for col in SENSOR_COLS:
        # Definir límites razonables para cada sensor
        if 'Gyro' in col:
            # Giroscopio: típicamente ±2000 deg/s
            df_clean = df_clean[(df_clean[col] >= -2500) & (df_clean[col] <= 2500)]
        else:  # Acelerómetro
            # Acelerómetro: típicamente ±16g (1g ≈ 9.81 m/s²)
            df_clean = df_clean[(df_clean[col] >= -200) & (df_clean[col] <= 200)]
    
    removed = original_len - len(df_clean)
    
    if removed > 0:
        print(f"⚠️  Limpieza de datos: {removed} filas corruptas/inválidas eliminadas")
        print(f"   Filas válidas restantes: {len(df_clean)}/{original_len}")
    
    return df_clean

# --- UTILIDADES PARA GESTIÓN DE MÚLTIPLES GESTOS ---
def get_gesture_path(gesture_name):
    """Retorna la ruta del archivo de un gesto específico."""
    return os.path.join(MODELS_DIR, f"{gesture_name}.pkl")

def list_trained_gestures():
    """Lista todos los gestos entrenados disponibles."""
    if not os.path.exists(MODELS_DIR):
        return []
    files = [f.replace('.pkl', '') for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    return files

def load_gesture(gesture_name):
    """Carga un gesto específico."""
    gesture_path = get_gesture_path(gesture_name)
    if not os.path.exists(gesture_path):
        return None
    return joblib.load(gesture_path)

def load_all_gestures():
    """Carga todos los gestos entrenados."""
    gestures = {}
    gesture_names = list_trained_gestures()
    for name in gesture_names:
        gestures[name] = load_gesture(name)
    return gestures
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
def train_model(csv_file, gesture_name=None):
    """
    Lee el CSV de entrenamiento y crea un TEMPLATE de secuencia.
    Este template representa el "camino ideal" del gesto.
    
    Args:
        csv_file: Archivo CSV con los datos del gesto
        gesture_name: Nombre único del gesto (ej: "lumos", "expelliarmus")
    """
    if gesture_name is None:
        gesture_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    try:
        df_gesture = pd.read_csv(csv_file)
        print(f"✅ Datos cargados de '{csv_file}' ({len(df_gesture)} muestras).")
    except FileNotFoundError:
        print(f"❌ Error: Archivo CSV '{csv_file}' no encontrado.")
        return
    except Exception as e:
        print(f"❌ Error al leer el CSV: {e}")
        return

    # LIMPIEZA Y VALIDACIÓN DE DATOS
    df_gesture = clean_and_validate_csv(df_gesture)
    
    if len(df_gesture) < TEMPLATE_LENGTH:
        print(f"❌ Error: Después de limpiar, quedan {len(df_gesture)} muestras.")
        print(f"   Se necesitan al menos {TEMPLATE_LENGTH} muestras válidas.")
        print(f"   Revisa la calidad de los datos recolectados.")
        return

    # 1. CALCULAR ACTIVIDAD Y SEGMENTAR EL GESTO PRINCIPAL
    try:
        df_gesture['Activity'] = df_gesture[SENSOR_COLS].std(axis=1)
    except Exception as e:
        print(f"❌ Error al calcular actividad: {e}")
        print("   Verifica que todas las columnas de sensores existan y sean numéricas.")
        return
    
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
    try:
        template_features = extract_temporal_features(template_segment[SENSOR_COLS])
    except Exception as e:
        print(f"❌ Error al extraer características: {e}")
        return
    
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
        'gesture_name': gesture_name,
        'templates': templates,
        'template_length': TEMPLATE_LENGTH,
        'avg_activity': max_avg_activity,
        'sensor_cols': SENSOR_COLS,
        'trained_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'samples_used': len(df_gesture)
    }
    
    gesture_path = get_gesture_path(gesture_name)
    joblib.dump(template_data, gesture_path)
    
    print("\n" + "="*70)
    print(f"✅ GESTO '{gesture_name.upper()}' ENTRENADO CON ÉXITO")
    print(f"   Muestras válidas usadas: {len(df_gesture)}")
    print(f"   Longitud del Template: {TEMPLATE_LENGTH} muestras")
    print(f"   Actividad Promedio: {max_avg_activity:.4f}")
    print(f"   Número de Templates: {len(templates)}")
    print(f"   Archivo guardado: '{gesture_path}'")
    print(f"   Umbral DTW: {DTW_THRESHOLD}")
    print("="*70)
    
    # Mostrar lista de gestos entrenados
    all_gestures = list_trained_gestures()
    print(f"\n📚 Gestos entrenados totales: {len(all_gestures)}")
    print(f"   {', '.join(all_gestures)}")
    
    return True

# --- FASE 2: DETECCIÓN EN TIEMPO REAL CON DTW Y MÚLTIPLES GESTOS ---
def run_detector(serial_port, baud_rate, target_gestures=None):
    """
    Detecta gestos comparando la secuencia en tiempo real con múltiples templates
    usando Dynamic Time Warping (DTW).
    
    Args:
        serial_port: Puerto serial del ESP32
        baud_rate: Velocidad de comunicación
        target_gestures: Lista de nombres de gestos a detectar (None = todos)
    """
    global realtime_buffer, cooldown_counter
    
    # Cargar todos los gestos o solo los especificados
    if target_gestures is None:
        all_gestures = load_all_gestures()
        if not all_gestures:
            print("❌ Error: No hay gestos entrenados. Ejecuta 'train' primero.")
            return
        print(f"✅ Cargados {len(all_gestures)} gestos: {', '.join(all_gestures.keys())}")
    else:
        all_gestures = {}
        for gesture_name in target_gestures:
            gesture_data = load_gesture(gesture_name)
            if gesture_data is None:
                print(f"⚠️  Advertencia: Gesto '{gesture_name}' no encontrado.")
            else:
                all_gestures[gesture_name] = gesture_data
        
        if not all_gestures:
            print("❌ Error: No se pudo cargar ningún gesto especificado.")
            return
        print(f"✅ Cargados gestos específicos: {', '.join(all_gestures.keys())}")
    
    template_length = list(all_gestures.values())[0]['template_length']

    print(f"📡 Conectando a {serial_port} @ {baud_rate}...")
    
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print("🟢 Detector Multi-Gesto ACTIVO")
        print(f"🎯 Esperando gestos... (DTW Threshold: {DTW_THRESHOLD})")
        print(f"🪄 Gestos activos: {', '.join(all_gestures.keys())}\n")
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
                            
                            # 3. CALCULAR DTW CON TODOS LOS GESTOS ENTRENADOS
                            gesture_distances = {}
                            
                            for gesture_name, gesture_data in all_gestures.items():
                                templates = gesture_data['templates']
                                min_distance = float('inf')
                                
                                for template in templates:
                                    distance = dtw_distance(current_normalized, template)
                                    if distance < min_distance:
                                        min_distance = distance
                                
                                gesture_distances[gesture_name] = min_distance
                            
                            # 4. ENCONTRAR EL GESTO MÁS CERCANO
                            best_gesture = min(gesture_distances, key=gesture_distances.get)
                            best_distance = gesture_distances[best_gesture]
                            
                            # 5. VERIFICAR SI SUPERA EL UMBRAL
                            if best_distance <= DTW_THRESHOLD:
                                # Verificar que sea significativamente mejor que otros gestos
                                second_best_distance = sorted(gesture_distances.values())[1] if len(gesture_distances) > 1 else float('inf')
                                
                                # Margen de discriminación entre gestos
                                is_distinctive = (second_best_distance - best_distance) > SIMILARITY_MARGIN
                                
                                if is_distinctive or len(all_gestures) == 1:
                                    similarity_score = max(0, 100 - (best_distance / DTW_THRESHOLD * 100))
                                    
                                    print(f"\n\n🎉 ¡GESTO DETECTADO: '{best_gesture.upper()}'! ✨")
                                    print(f"   Distancia DTW: {best_distance:.2f}")
                                    print(f"   Similitud: {similarity_score:.1f}%")
                                    print(f"   Actividad: {current_activity:.3f}")
                                    
                                    if len(all_gestures) > 1:
                                        print(f"   Margen sobre siguiente: {second_best_distance - best_distance:.2f}")
                                        print(f"   Otros gestos descartados:")
                                        for gname, gdist in sorted(gesture_distances.items(), key=lambda x: x[1]):
                                            if gname != best_gesture:
                                                print(f"      - {gname}: {gdist:.2f}")
                                    
                                    # Activar cooldown y limpiar buffer
                                    cooldown_counter = COOLDOWN_SAMPLES
                                    realtime_buffer = []
                                else:
                                    sys.stdout.write(
                                        f"\r⚠️  Gesto ambiguo - "
                                        f"{best_gesture}:{best_distance:.1f} vs otros:{second_best_distance:.1f}   "
                                    )
                                    sys.stdout.flush()
                            else:
                                # Mostrar progreso de análisis
                                status_line = f"🔍 Analizando... "
                                for gname in sorted(all_gestures.keys()):
                                    status_line += f"{gname}:{gesture_distances[gname]:.1f} | "
                                status_line += f"Act:{current_activity:.3f}   "
                                
                                sys.stdout.write(f"\r{status_line}")
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
    Incluye validación en tiempo real de datos.
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
    error_count = 0
    
    with open(output_file, 'w') as f:
        f.write(HEADER)
        print(f"\n📝 Recolección iniciada. Objetivo: {repetitions} repeticiones")
        print(f"⚠️  IMPORTANTE: Realiza el gesto COMPLETO de forma CONSISTENTE\n")
        
        for rep in range(1, repetitions + 1):
            input(f"Repetición {rep}/{repetitions} - Presiona ENTER para INICIAR...")
            print("🔴 GRABANDO... (Ctrl+C para detener)")
            
            rep_samples = 0
            rep_errors = 0
            
            try:
                while True:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('latin-1').strip()
                        parts = line.split(',')
                        
                        if len(parts) == NUM_SENSOR_VALUES:
                            # VALIDACIÓN EN TIEMPO REAL
                            try:
                                values = [float(p.strip()) for p in parts]
                                
                                # Verificar rangos razonables
                                valid = True
                                for i, val in enumerate(values):
                                    if i < 3:  # Giroscopio
                                        if abs(val) > 2500:
                                            valid = False
                                    else:  # Acelerómetro
                                        if abs(val) > 200:
                                            valid = False
                                
                                if valid:
                                    f.write(line + '\n')
                                    rep_samples += 1
                                    data_count += 1
                                else:
                                    rep_errors += 1
                                    error_count += 1
                                    
                            except ValueError:
                                rep_errors += 1
                                error_count += 1
                                continue
                            
                            status = f"\rRep {rep}: {rep_samples} válidas"
                            if rep_errors > 0:
                                status += f" | {rep_errors} errores"
                            status += f" | Total: {data_count} (Ctrl+C para finalizar)"
                            sys.stdout.write(status)
                            sys.stdout.flush()
                    
                    time.sleep(0.01)
                    
            except KeyboardInterrupt:
                pass
            
            quality = "✅" if rep_errors < rep_samples * 0.1 else "⚠️"
            print(f"\n{quality} Repetición {rep} completada: {rep_samples} muestras válidas")
            if rep_errors > 0:
                print(f"   {rep_errors} lecturas descartadas por valores inválidos")
            input("Presiona ENTER para continuar...\n")

    ser.close()
    
    quality_pct = (data_count / (data_count + error_count) * 100) if (data_count + error_count) > 0 else 100
    
    print(f"\n{'='*60}")
    print(f"✅ RECOLECCIÓN FINALIZADA")
    print(f"   Muestras válidas: {data_count}")
    if error_count > 0:
        print(f"   Muestras descartadas: {error_count}")
    print(f"   Calidad de datos: {quality_pct:.1f}%")
    print(f"   Archivo: '{output_file}'")
    print(f"   Recomendado: Mínimo {TEMPLATE_LENGTH} muestras por repetición")
    print(f"{'='*60}")

# --- UTILIDAD: VISUALIZAR TEMPLATE ---
def visualize_template(gesture_name=None):
    """
    Muestra información del template entrenado.
    Si no se especifica gesture_name, muestra todos.
    """
    if gesture_name:
        # Mostrar un gesto específico
        gesture_data = load_gesture(gesture_name)
        if gesture_data is None:
            print(f"❌ Gesto '{gesture_name}' no encontrado.")
            return
        
        print("\n" + "="*60)
        print(f"📊 INFORMACIÓN DEL GESTO: {gesture_name.upper()}")
        print("="*60)
        print(f"Fecha de entrenamiento: {gesture_data.get('trained_date', 'N/A')}")
        print(f"Número de templates: {len(gesture_data['templates'])}")
        print(f"Longitud: {gesture_data['template_length']} muestras")
        print(f"Dimensiones: {gesture_data['templates'][0].shape[1]} características")
        print(f"Actividad promedio: {gesture_data['avg_activity']:.4f}")
        print("="*60)
    else:
        # Mostrar todos los gestos
        all_gestures = load_all_gestures()
        if not all_gestures:
            print("❌ No hay gestos entrenados.")
            return
        
        print("\n" + "="*70)
        print(f"📚 BIBLIOTECA DE GESTOS ({len(all_gestures)} gestos entrenados)")
        print("="*70)
        
        for name, data in sorted(all_gestures.items()):
            print(f"\n🪄 {name.upper()}")
            print(f"   Fecha: {data.get('trained_date', 'N/A')}")
            print(f"   Templates: {len(data['templates'])} | Longitud: {data['template_length']} | Actividad: {data['avg_activity']:.4f}")
        
        print("\n" + "="*70)
        print(f"Umbral DTW actual: {DTW_THRESHOLD}")
        print(f"Margen de similitud: {SIMILARITY_MARGIN}")
        print("="*70)

def delete_gesture(gesture_name):
    """Elimina un gesto entrenado."""
    gesture_path = get_gesture_path(gesture_name)
    if not os.path.exists(gesture_path):
        print(f"❌ Gesto '{gesture_name}' no encontrado.")
        return False
    
    os.remove(gesture_path)
    print(f"✅ Gesto '{gesture_name}' eliminado correctamente.")
    return True

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