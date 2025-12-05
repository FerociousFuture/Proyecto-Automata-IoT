import sqlite3
import numpy as np
from sklearn.preprocessing import LabelEncoder
from MLTrainer.Oscilloscope import Input, Mpu6050 

# --- CONFIGURACIÓN ---
DB_NAME = 'gestures_data.db'
TABLE_NAME = 'gestos_raw'

# Longitud estandarizada para las secuencias de gestos (AJUSTAR según la duración de tus gestos)
SEQUENCE_LENGTH = 50 

# --- CRUD: LECTURA DE LA BASE DE DATOS ---

def create_connection():
    """Crea una conexión a la base de datos SQLite."""
    try:
        conn = sqlite3.connect(DB_NAME)
        return conn
    except sqlite3.Error as e:
        print(f"Error al conectar a la DB: {e}")
        return None

def load_data_for_ml(conn):
    """
    Lee datos de SQLite, agrupa las muestras por la columna 'etiqueta' 
    y convierte cada secuencia en objetos Input.
    """
    print("Leyendo datos de la base de datos para el entrenamiento...")
    cursor = conn.cursor()
    
    # Selecciona solo las muestras que tienen una etiqueta definida y ordénalas
    query = f"""
    SELECT accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, etiqueta
    FROM {TABLE_NAME}
    WHERE etiqueta IS NOT NULL AND etiqueta != ''
    ORDER BY etiqueta, timestamp
    """
    cursor.execute(query)
    
    # Diccionario para agrupar secuencias: {'Lumos': [Input1, Input2, ...], 'Nox': [...]}
    gestures_data = {}
    
    for row in cursor.fetchall():
        # Desempaqueta los datos (Ax, Ay, Az, Gx, Gy, Gz, Etiqueta)
        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, label = row
        
        # Crea el objeto Input
        input_obj = Input(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        
        # Agrega la muestra a la secuencia de su etiqueta
        if label not in gestures_data:
            gestures_data[label] = []
        gestures_data[label].append(input_obj)
        
    return gestures_data

# --- PREPROCESAMIENTO PARA MACHINE LEARNING ---

def sequence_to_numpy(sequence_of_inputs):
    """Convierte una secuencia de objetos Input en un array NumPy de 2 dimensiones."""
    data_list = []
    for obj in sequence_of_inputs:
        # Vector de 6 dimensiones: [Ax, Ay, Az, Gx, Gy, Gz]
        data_list.append([obj.accX, obj.accY, obj.accZ, obj.angRX, obj.angRY, obj.angRZ])
    return np.array(data_list)

def preprocess_and_pad_sequences(gestures_data):
    """
    Aplica relleno (padding) a todas las secuencias para que tengan la misma longitud (SEQUENCE_LENGTH)
    y las transforma en el formato final para el ML (X, y).
    """
    X_sequences = []  # Entradas (gestos)
    y_labels = []     # Etiquetas (nombres de hechizos)
    
    for label, sequence_of_inputs in gestures_data.items():
        numpy_sequence = sequence_to_numpy(sequence_of_inputs)
        
        # 1. Truncar o Rellenar (Padding)
        if numpy_sequence.shape[0] > SEQUENCE_LENGTH:
            # Si es muy larga, la truncamos desde el inicio
            processed_sequence = numpy_sequence[:SEQUENCE_LENGTH, :]
        else:
            # Si es más corta, la rellenamos con ceros al final (padding)
            padding_needed = SEQUENCE_LENGTH - numpy_sequence.shape[0]
            # Creamos un array de padding de ceros (padding_needed filas x 6 columnas)
            padding_array = np.zeros((padding_needed, numpy_sequence.shape[1]))
            # Apilamos la secuencia original con el relleno
            processed_sequence = np.vstack([numpy_sequence, padding_array])
            
        X_sequences.append(processed_sequence)
        y_labels.append(label)
        
    # 2. Conversión final y codificación de etiquetas
    X = np.array(X_sequences)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels) # Codificación: 'Lumos' -> 0, 'Nox' -> 1, etc.
    
    return X, y, label_encoder

# --- FUNCIÓN PRINCIPAL DE PREPARACIÓN DE DATOS ---

def prepare_data_for_training():
    conn = create_connection()
    if conn is None:
        return None, None, None
        
    gestures_data = load_data_for_ml(conn)
    conn.close()
    
    if not gestures_data:
        print("ADVERTENCIA: No se encontraron datos etiquetados en la base de datos.")
        return None, None, None
        
    X, y, encoder = preprocess_and_pad_sequences(gestures_data)
    
    print("\n--- RESUMEN DE PREPARACIÓN DE DATOS ---")
    print(f"Número de clases únicas (Hechizos): {len(encoder.classes_)}")
    print(f"Estructura del Array X (Entradas): {X.shape} (Gestos, Longitud, Ejes)")
    
    return X, y, encoder