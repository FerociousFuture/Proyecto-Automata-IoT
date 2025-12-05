# Proyecto-Automata-IoT/Codigo/BackEnd/serial_data_collector.py

import serial
import sqlite3
import time
from datetime import datetime
import numpy as np

# Importa las clases de tu módulo
from MLTrainer.Oscilloscope import Input, Mpu6050 

# --- CONFIGURACIÓN ---
DB_NAME = 'gestures_data.db'
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200 
TABLE_NAME = 'gestos_raw'

# --- 1. LÓGICA DE BASE DE DATOS (CRUD: CREATE) ---

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        return conn
    except sqlite3.Error as e:
        print(f"Error al conectar a la DB: {e}")
        return None

def create_table(conn):
    sql_create_table = f""" CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                                id INTEGER PRIMARY KEY,
                                timestamp TEXT NOT NULL,
                                accel_x REAL,
                                accel_y REAL,
                                accel_z REAL,
                                gyro_x REAL, 
                                gyro_y REAL,
                                gyro_z REAL, 
                                magnitud REAL,
                                etiqueta TEXT DEFAULT ''
                            ); """
    try:
        conn.cursor().execute(sql_create_table)
        conn.commit()
        print(f"Tabla '{TABLE_NAME}' asegurada/creada.")
    except sqlite3.Error as e:
        print(f"Error al crear tabla: {e}")

def insert_data(conn, data):
    sql_insert = f""" INSERT INTO {TABLE_NAME}
                     (timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, magnitud)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?) """
    try:
        conn.cursor().execute(sql_insert, data)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error al insertar datos: {e}")

# --- LÓGICA DE CAPTURA SERIAL (El paso intermedio) ---

def start_serial_capture(conn):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Conexión Serial abierta en {SERIAL_PORT}. Esperando datos...")
        time.sleep(2) 
        ser.flushInput() 

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                
                # Ejemplo de línea recibida (Ax, Ay, Az, Mag)
                # El código C++ no está enviando Gyro, así que usamos 0.0 para Gx, Gy, Gz.
                if not line or line.startswith("Timestamp"):
                    continue
                
                parts = line.split(',')
                
                if len(parts) == 5: # Esperamos 5 valores del C++ (TS, Ax, Ay, Az, Mag)
                    # parts[0] = Timestamp ESP32
                    accel_x, accel_y, accel_z, magnitude = [float(p.strip()) for p in parts[1:]]

                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # Data: (timestamp, ax, ay, az, gx, gy, gz, magnitud)
                    data_to_store = (now, accel_x, accel_y, accel_z, 0.0, 0.0, 0.0, magnitude)
                    
                    insert_data(conn, data_to_store)
                    print(f"Captured and Saved: Ax={accel_x:.2f}, Mag={magnitude:.2f}")

    except serial.SerialException as e:
        print(f"FATAL ERROR: No se puede abrir el puerto Serial {SERIAL_PORT}. Error: {e}")
        print("Asegúrate de que la ESP32 esté conectada y que el baud rate (115200) sea correcto.")
    except KeyboardInterrupt:
        print("\nCaptura de datos detenida por el usuario.")
    finally:
        if 'ser' in locals():
            ser.close()

# --- 3. LÓGICA DE LECTURA Y PREPARACIÓN (CRUD: READ & NumPy) ---

def load_data_to_objects(conn, label=''):
    """Lee datos de SQLite y los convierte a una lista de objetos Input."""
    cursor = conn.cursor()
    
    # Filtra solo las filas con la etiqueta deseada para ML. 
    # (Por ahora, como no hay etiquetas, lee todo)
    query = f"SELECT accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z FROM {TABLE_NAME}"
    if label:
         query += f" WHERE etiqueta='{label}'"

    cursor.execute(query)
    
    mpu_data_set = Mpu6050()
    
    # Convierte cada fila de la DB en un objeto Input
    for row in cursor.fetchall():
        # Los 6 valores corresponden a (ax, ay, az, gx, gy, gz)
        input_obj = Input(row[0], row[1], row[2], row[3], row[4], row[5])
        mpu_data_set.insertData(input_obj)
        
    return mpu_data_set.getData()

def transform_to_numpy_array(input_objects):
    """Toma la lista de objetos Input y los transforma en un array NumPy."""
    
    # Crea una lista de listas, donde cada lista interna es un vector de 6 dimensiones (los 6 ejes)
    data_list = []
    for obj in input_objects:
        data_list.append([obj.accX, obj.accY, obj.accZ, obj.angRX, obj.angRY, obj.angRZ])
        
    # Convierte la lista de listas a un array NumPy
    return np.array(data_list)


# --- PROGRAMA PRINCIPAL ---

if __name__ == '__main__':
    conn = create_connection()
    if conn:
        create_table(conn)
        
        # Iniciar la Captura (Descomentar para usar la ESP32)
        # start_serial_capture(conn)

        # Ejemplo de Uso POST-CAPTURA (para ML):
        print("\n--- PASO 3: LECTURA DE DATOS PARA ML ---")
        
        # 1. Leer datos de la DB como objetos
        input_objects = load_data_to_objects(conn)
        print(f"Total de muestras cargadas (objetos Input): {len(input_objects)}")
        
        if input_objects:
            # 2. Transformar los objetos a array de NumPy
            numpy_array = transform_to_numpy_array(input_objects)
            
            print(f"Tipo de array para ML: {type(numpy_array)}")
            print(f"Forma (Shape) del array: {numpy_array.shape}")
            print("Primeras 5 filas del array (listas para el modelo de IA):")
            print(numpy_array[:5])
        
        conn.close()