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

# --- CONFIGURACIÓN SERIAL ---
# IMPORTANTE: Asegúrate que este puerto es correcto en tu Raspberry Pi
# En Windows suele ser 'COM3', 'COM4'. En Linux/RPi '/dev/ttyUSB0' o '/dev/ttyACM0'
PORT_SERIAL = '/dev/ttyUSB0' 
BAUD_RATE = 115200

# --- VARIABLES GLOBALES DE HILO DE IA ---
detection_thread = None
stop_event = threading.Event()
message_queue = queue.Queue()

def backend_callback(json_msg):
    """
    Esta función es llamada por Entrenamiento.py cada vez que hay un log o detección.
    Recibe un string JSON.
    """
    message_queue.put(json_msg)

def run_ai_service():
    """Función que corre en el thread secundario"""
    try:
        # Llamamos al detector pasándole nuestra función de callback
        Entrenamiento.run_detector(
            serial_port=PORT_SERIAL, 
            baud_rate=BAUD_RATE, 
            message_callback=backend_callback
        )
    except Exception as e:
        print(f"Error en hilo de IA: {e}")

# --- RUTAS DE LA APP ---

@app.route('/')
def index():
    # Sirve el index.html principal
    return send_from_directory('../FrontEnd', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # Sirve CSS, JS e imágenes
    return send_from_directory('../FrontEnd', filename)

@app.route('/api/start_practice', methods=['POST'])
def start_practice():
    global detection_thread
    
    if detection_thread is not None and detection_thread.is_alive():
        return jsonify({"success": True, "message": "El servicio de detección ya está activo."})
    
    # Limpiar cola vieja
    with message_queue.mutex:
        message_queue.queue.clear()
    
    detection_thread = threading.Thread(target=run_ai_service, daemon=True)
    detection_thread.start()
    
    return jsonify({"success": True, "message": "Servicio de detección iniciado."})

@app.route('/api/get_live_logs', methods=['GET'])
def get_live_logs():
    """
    El frontend consulta esto periódicamente para obtener nuevos eventos.
    """
    messages = []
    # Sacar todos los mensajes pendientes de la cola
    try:
        while not message_queue.empty():
            # Obtener mensaje sin bloquear
            msg = message_queue.get_nowait()
            messages.append(msg)
    except queue.Empty:
        pass
    
    return jsonify({"logs": messages})

# --- GESTIÓN DE USUARIOS (SQLite) ---

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
    # Ejecutar servidor accesible desde la red local
    app.run(debug=True, host='0.0.0.0', port=8001, use_reloader=False)