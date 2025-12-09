import pandas as pd
import numpy as np
import joblib
import serial
import time
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# --- CONFIGURACIÓN GLOBAL ---
MODEL_PATH = 'gesture_classifier_model.pkl' 
SCALER_PATH = 'scaler_for_prediction.pkl'
SENSOR_COLS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']
NUM_SENSOR_VALUES = 6
HEADER = "Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z\n"

# --- CLAVES PARA LA CLASIFICACIÓN DE GESTOS ---
WINDOW_SIZE = 40    # 2.0 segundos de ventana (Longitud del gesto)
STEP_SIZE = 20      # Evalúa cada 1.0 segundo (Solapamiento)
realtime_buffer = []

# --- UMBRALES DE ROBUSTEZ (AJUSTADOS PARA SER MÁS ESTRICTOS) ---
CONFIDENCE_THRESHOLD = 0.95 # Confianza mínima del modelo (Requiere alta precisión del gesto)
ACTIVITY_THRESHOLD = 0.10   # <--- Aumentado: Actividad mínima del Giroscopio para 'encender' el detector.

# --- FUNCIÓN DE INGENIERÍA DE CARACTERÍSTICAS (MEJORADA PARA INCLINACIÓN Y AMPLITUD) ---
def extract_features(data):
    """
    Calcula la media, STD, Varianza, IQR, RMS, y Ángulos (Roll/Pitch) de los ejes.
    """
    features = {}
    data_df = pd.DataFrame(data, columns=SENSOR_COLS)
    
    # 1. CARACTERÍSTICAS DERIVADAS (Amplitud y Forma del Gesto)
    
    # Vector de Aceleración Total (Energía)
    data_df['Acc_Mag'] = np.sqrt(data_df['Acc_X']**2 + data_df['Acc_Y']**2 + data_df['Acc_Z']**2)
    
    # Aproximación de Ángulo (Inclinación de la Varita - Énfasis en X e Y)
    # Roll (X): atan2(Acc_Y, Acc_Z)
    data_df['Roll'] = np.arctan2(data_df['Acc_Y'], data_df['Acc_Z']) * 180 / np.pi
    # Pitch (Y): atan2(-Acc_X, sqrt(Acc_Y^2 + Acc_Z^2))
    data_df['Pitch'] = np.arctan2(-data_df['Acc_X'], np.sqrt(data_df['Acc_Y']**2 + data_df['Acc_Z']**2)) * 180 / np.pi

    # --- Generación de Features ---
    # Incluimos los 6 sensores originales + las 3 características derivadas
    ALL_COLS = SENSOR_COLS + ['Acc_Mag', 'Roll', 'Pitch']
    
    for col in ALL_COLS:
        # Medidas Básicas
        features[f'{col}_mean'] = data_df[col].mean()
        features[f'{col}_std'] = data_df[col].std() 
        
        # Medidas de Amplitud (Clave para gestos amplios)
        features[f'{col}_var'] = data_df[col].var()
        Q1 = data_df[col].quantile(0.25)
        Q3 = data_df[col].quantile(0.75)
        features[f'{col}_iqr'] = Q3 - Q1
        
        # Medida de Energía
        features[f'{col}_rms'] = np.sqrt(np.mean(data_df[col]**2))
        
    return pd.Series(features)

# --- FASE 1: ENTRENAMIENTO DEL MODELO (train) ---
def train_model(csv_file):
    """
    Segmenta el CSV (Clase 1), genera datos de Reposo (Clase 0), 
    y entrena el Random Forest con ambas clases.
    """
    global WINDOW_SIZE, STEP_SIZE
    try:
        df_gesture = pd.read_csv(csv_file)
        print(f"✅ Datos de Gesto (Clase 1) cargados de '{csv_file}'.")
    except FileNotFoundError:
        print(f"❌ Error: Archivo CSV '{csv_file}' no encontrado.")
        return

    # --- 1. Generar Muestras de Gesto (Clase 1) ---
    X_features_gesture = []
    
    for i in range(0, len(df_gesture) - WINDOW_SIZE, STEP_SIZE):
        window = df_gesture.iloc[i: i + WINDOW_SIZE]
        if len(window) == WINDOW_SIZE:
            features = extract_features(window)
            X_features_gesture.append(features)

    X_gesture = pd.DataFrame(X_features_gesture)
    y_gesture = np.ones(len(X_gesture)) 

    if len(X_gesture) < 10:
        print("❌ Error: Muy pocas muestras de gesto para entrenamiento. Graba al menos 10s de movimiento.")
        return

    # --- 2. Generar Muestras de Reposo (Clase 0) a partir del propio CSV ---
    print("⏳ Generando datos de Reposo (Clase 0) basados en tu propio CSV...")
    
    # Calcular estadísticas medias para generar un reposo realista
    df_gesture['Activity_Metric'] = df_gesture[SENSOR_COLS].std(axis=1)
    df_rest_candidate = df_gesture.sort_values(by='Activity_Metric').head(int(len(df_gesture) * 0.1))

    if df_rest_candidate.empty:
        mean_acc_z = 9.81
    else:
        mean_acc_z = df_rest_candidate['Acc_Z'].mean() 

    # Generamos 5 veces más muestras de reposo que de gestos para robustez.
    num_rest_samples = len(X_gesture) * 5
    
    rest_data = []
    for _ in range(num_rest_samples * WINDOW_SIZE):
        # Gyro: Ruido muy bajo
        g_x = np.random.normal(0, 0.005)
        g_y = np.random.normal(0, 0.005)
        g_z = np.random.normal(0, 0.005)
        # Accel: Posición de reposo simulada
        a_x = np.random.normal(0, 0.05)
        a_y = np.random.normal(0, 0.05)
        a_z = np.random.normal(mean_acc_z, 0.05) 
        rest_data.append([g_x, g_y, g_z, a_x, a_y, a_z])

    df_rest = pd.DataFrame(rest_data, columns=SENSOR_COLS)
    
    X_features_rest = []
    for i in range(0, len(df_rest) - WINDOW_SIZE, STEP_SIZE):
        window = df_rest.iloc[i: i + WINDOW_SIZE]
        if len(window) == WINDOW_SIZE:
            features = extract_features(window)
            X_features_rest.append(features)
            
    X_rest = pd.DataFrame(X_features_rest)
    y_rest = np.zeros(len(X_rest)) 

    # --- 3. Combinar y Entrenar ---
    X_combined = pd.concat([X_gesture, X_rest], ignore_index=True)
    y_combined = np.concatenate([y_gesture, y_rest])

    print(f"Total de muestras de entrenamiento: {len(X_combined)} (Gesto: {len(X_gesture)}, Reposo: {len(X_rest)})")

    # 4. Estandarización y Entrenamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    print("⏳ Entrenando modelo Random Forest Biclase...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y_combined)
    
    # 5. Guardar
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # 6. Reporte
    accuracy = model.score(X_scaled, y_combined)
    
    print("\n" + "="*60)
    print("¡ENTRENAMIENTO COMPLETADO Y OPTIMIZADO PARA HECHIZOS!")
    print(f"Precisión General: {accuracy:.4f}")
    print(f"Archivos guardados: '{MODEL_PATH}' y '{SCALER_PATH}'")
    print(f"El umbral de detección es CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}. El modelo es ESTRICTO.")
    print("="*60)
    return True


# --- FASE 2: LÓGICA DE DETECCIÓN EN TIEMPO REAL (detect) ---
def run_detector(serial_port, baud_rate):
    """
    Conecta al puerto Serial, usa una ventana móvil, aplica umbrales de actividad/confianza y detecta el gesto.
    """
    global realtime_buffer, WINDOW_SIZE, STEP_SIZE
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("❌ Error: No se encontraron los archivos del modelo. Ejecuta el modo 'train' primero.")
        return

    print(f"📡 Intentando conectar a {serial_port} @ {baud_rate}...")
    
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print("🟢 Conexión Serial establecida. Detector activo.")
        print(f"Esperando {WINDOW_SIZE} muestras para la primera predicción...")
    except serial.SerialException as e:
        print(f"❌ Error al abrir el puerto serial '{serial_port}': {e}")
        print("Asegúrate de que el puerto sea correcto y el ESP32 esté conectado.")
        return

    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('latin-1').strip()
                parts = line.split(',')
                
                if len(parts) == NUM_SENSOR_VALUES:
                    try:
                        sensor_values = [float(p.strip()) for p in parts]
                        
                        # 1. ACTUALIZAR VENTANA MÓVIL (BUFFER)
                        realtime_buffer.append(sensor_values)
                        
                        if len(realtime_buffer) > WINDOW_SIZE:
                            realtime_buffer = realtime_buffer[-WINDOW_SIZE:] 
                        
                        # 2. EVALUAR SOLO CUANDO LA VENTANA ESTÁ LLENA
                        if len(realtime_buffer) == WINDOW_SIZE:
                            
                            window_data = pd.DataFrame(realtime_buffer, columns=SENSOR_COLS)
                            current_features = extract_features(window_data)
                            
                            # 3. APLICAR GESTIÓN DE ROBUSTEZ Y UMBRALES
                            
                            # Condición A: Verificar Actividad Mínima (INTERRUPTOR ESTRICO)
                            # Se usa la STD del Gyroscopio (X, Y, Z) para exigir rotación/movimiento en los 2 segundos de ventana
                            gyro_activity = (window_data['Gyro_X'].std() + window_data['Gyro_Y'].std() + window_data['Gyro_Z'].std()) / 3
                            is_active = gyro_activity > ACTIVITY_THRESHOLD
                            
                            # 4. PREDICCIÓN CON CONFIANZA
                            X_test = current_features.to_frame().T
                            X_test_scaled = scaler.transform(X_test)
                            
                            probabilities = model.predict_proba(X_test_scaled)[0]
                            confidence = probabilities[1] 
                            
                            # 5. DECISIÓN FINAL (GESTO VÁLIDO)
                            if is_active and confidence > CONFIDENCE_THRESHOLD:
                                print(f"\n\n🎉 GESTO DETECTADO: [HECHIZO VÁLIDO] (Confianza: {confidence:.2f})")
                                
                                # Deslizar la ventana para buscar el próximo gesto
                                realtime_buffer = realtime_buffer[STEP_SIZE:]
                            
                            else:
                                sys.stdout.write(f"\rAnalizando... Actividad Gyro: {gyro_activity:.3f} | Confianza Gesto: {confidence:.2f} (Esperando hechizo...)")
                                sys.stdout.flush()

                    except ValueError:
                        continue
                
            time.sleep(0.001) 
            
        except KeyboardInterrupt:
            print("\nDeteniendo el detector.")
            break
        except Exception as e:
            print(f"Ocurrió un error inesperado en el detector: {e}")
            break

    ser.close()
    print("Conexión serial cerrada.")

# --- FASE 3: RECOLECCIÓN DE DATOS (collect) ---
def collect_data(output_file, repetitions, samples_per_rep, serial_port, baud_rate):
    """
    Conecta al ESP32, lee los datos crudos y los guarda en un CSV por iteraciones.
    """
    print(f"📡 Intentando conectar a {serial_port} @ {baud_rate}...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print("🟢 Conexión Serial establecida.")
    except serial.SerialException as e:
        print(f"❌ Error al abrir el puerto serial '{serial_port}': {e}")
        print("Asegúrate de que el puerto sea correcto y el ESP32 esté conectado.")
        return

    data_count = 0
    
    with open(output_file, 'w') as f:
        f.write(HEADER)
        print(f"\n📢 Comienza la recolección. Objetivo: {repetitions} repeticiones de {samples_per_rep} muestras cada una.")
        
        for rep in range(1, repetitions + 1):
            input(f"\n---> PREPARADO para Repetición {rep}/{repetitions}. Presiona ENTER para INICIAR el HECHIZO (Movimiento AMPLIO)...")
            
            samples_collected_in_rep = 0
            
            # Recolección de la repetición
            while samples_collected_in_rep < samples_per_rep:
                try:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('latin-1').strip()
                        parts = line.split(',')
                        
                        if len(parts) == NUM_SENSOR_VALUES:
                            f.write(line + '\n')
                            samples_collected_in_rep += 1
                            data_count += 1
                            sys.stdout.write(f"\rRepetición {rep}: Muestras recolectadas {samples_collected_in_rep}/{samples_per_rep}")
                            sys.stdout.flush()
                    
                    time.sleep(0.01)
                    
                except KeyboardInterrupt:
                    print("\nColección detenida por el usuario.")
                    ser.close()
                    return

            print(f"\nRepetición {rep} completada. Vuelve a la posición inicial (reposo).")
            time.sleep(1) # Pequeña pausa de seguridad antes de la siguiente repetición

    ser.close()
    print(f"\n\n✅ Colección finalizada. Total de muestras: {data_count}")
    print(f"Archivo guardado: '{output_file}'.")
    print("Ahora puedes usar 'train' con este archivo.")

# --- BLOQUE PRINCIPAL DE COMANDOS ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("1. Para ADQUIRIR DATOS (Crear CSV): python Entrenamiento.py collect <salida.csv> <PUERTO> <REPETICIONES> <MUESTRAS_POR_REP>")
        print("2. Para ENTRENAR MODELO: python Entrenamiento.py train <nombre_archivo.csv>")
        print("3. Para DETECTAR MOVIMIENTO: python Entrenamiento.py detect <PUERTO> <BAUD_RATE>")
        
        print("\nEjemplo de ADQUISICIÓN: python Entrenamiento.py collect mi_hechizo.csv /dev/ttyUSB0 10 40")
        print("Ejemplo de ENTRENAMIENTO: python Entrenamiento.py train mi_hechizo.csv")
        print("Ejemplo de DETECCIÓN: python Entrenamiento.py detect /dev/ttyUSB0 115200")
        
    elif sys.argv[1] == 'collect' and len(sys.argv) >= 6:
        output = sys.argv[2]
        port = sys.argv[3]
        repetitions = int(sys.argv[4])
        samples_per_rep = int(sys.argv[5])
        collect_data(output_file=output, repetitions=repetitions, samples_per_rep=samples_per_rep, serial_port=port, baud_rate=115200)

    elif sys.argv[1] == 'train' and len(sys.argv) == 3:
        train_model(sys.argv[2])
    
    elif sys.argv[1] == 'detect' and len(sys.argv) >= 3:
        port = sys.argv[2]
        baud = int(sys.argv[3]) if len(sys.argv) == 4 else 115200
        run_detector(serial_port=port, baud_rate=baud)
    
    else:
        print("Comando no reconocido o argumentos faltantes. Revisa el formato de uso.")