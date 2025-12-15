import sqlite3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
import queue
import sys
import time

# Configurar path para importar Entrenamiento
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Entrenamiento import Entrenamiento

app = Flask(__name__)
CORS(app) 

DB_NAME = "usuarios.db"
PORT_SERIAL = '/dev/ttyUSB0' 
BAUD_RATE = 115200

# --- VARIABLES GLOBALES ---
detection_thread = None
message_queue = queue.Queue() # Para logs y eventos importantes (lento)
stream_queue = queue.Queue()  # Para datos de sensores en tiempo real (rápido)

def backend_log_callback(json_msg):
    """Callback para eventos (Hechizos detectados)"""
    message_queue.put(json_msg)

def backend_data_callback(data_dict):
    """Callback para flujo de datos crudos (Dibujo)"""
    # Solo guardamos si la cola no está llena para evitar latencia
    if stream_queue.qsize() < 100: 
        stream_queue.put(data_dict)

def run_ai_service():
    """Función que corre en el thread secundario"""
    try:
        Entrenamiento.run_detector(
            serial_port=PORT_SERIAL, 
            baud_rate=BAUD_RATE, 
            message_callback=backend_log_callback,
            data_callback=backend_data_callback # Nuevo callback
        )
    except Exception as e:
        print(f"Error en hilo de IA: {e}")

# --- RUTAS DE LA APP ---

@app.route('/')
def index():
    return send_from_directory('../FrontEnd', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../FrontEnd', filename)

@app.route('/api/start_practice', methods=['POST'])
def start_practice():
    global detection_thread
    if detection_thread is not None and detection_thread.is_alive():
        return jsonify({"success": True, "message": "El servicio ya está activo."})
    
    with message_queue.mutex:
        message_queue.queue.clear()
    with stream_queue.mutex:
        stream_queue.queue.clear()
    
    detection_thread = threading.Thread(target=run_ai_service, daemon=True)
    detection_thread.start()
    return jsonify({"success": True, "message": "Servicio iniciado."})

@app.route('/api/get_live_logs', methods=['GET'])
def get_live_logs():
    """Polling lento (1s) para historial"""
    messages = []
    try:
        while not message_queue.empty():
            messages.append(message_queue.get_nowait())
    except queue.Empty:
        pass
    return jsonify({"logs": messages})

@app.route('/api/sensor_stream', methods=['GET'])
def sensor_stream():
    """Polling rápido (50-100ms) para dibujo"""
    data_points = []
    # Recuperar todos los puntos acumulados desde la última llamada
    try:
        while not stream_queue.empty():
            data_points.append(stream_queue.get_nowait())
    except queue.Empty:
        pass
    return jsonify(data_points)

# --- GESTIÓN DE USUARIOS ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_completo TEXT NOT NULL,
            usuario TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/registro', methods=['POST'])
def registrar_usuario():
    data = request.json
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO usuarios (nombre_completo, usuario, password) VALUES (?, ?, ?)", 
                       (data['fullName'], data['username'], data['password']))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Usuario creado."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/login', methods=['POST'])
def login_usuario():
    data = request.json
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT nombre_completo FROM usuarios WHERE usuario=? AND password=?", 
                   (data['username'], data['password']))
    row = cursor.fetchone()
    conn.close()
    if row:
        return jsonify({"success": True, "nombre": row[0]})
    return jsonify({"success": False, "message": "Credenciales inválidas"})

if __name__ == '__main__':
    if not os.path.exists(DB_NAME):
        init_db()
    app.run(debug=True, host='0.0.0.0', port=8004, use_reloader=False)