import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) # Permite que tu HTML se conecte con este Python

# Nombre del archivo de base de datos
DB_NAME = "usuarios.db"

# --- 1. CONFIGURACIÓN DE LA BASE DE DATOS ---
def init_db():
    """Crea la tabla de usuarios si no existe"""
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
    print("Base de datos inicializada correctamente.")

# --- 2. RUTAS (EL PUENTE CON EL HTML) ---

@app.route('/registro', methods=['POST'])
def registrar_usuario():
    data = request.json
    nombre = data.get('fullName')
    user = data.get('username')
    pwd = data.get('password')

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Intentamos guardar el usuario
        cursor.execute("INSERT INTO usuarios (nombre_completo, usuario, password) VALUES (?, ?, ?)", 
                       (nombre, user, pwd))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Usuario registrado exitosamente"})
    except sqlite3.IntegrityError:
        # Esto pasa si el usuario ya existe (porque pusimos UNIQUE)
        return jsonify({"success": False, "message": "El nombre de usuario ya existe"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/login', methods=['POST'])
def login_usuario():
    data = request.json
    user = data.get('username')
    pwd = data.get('password')

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Buscamos si existe alguien con ese usuario Y contraseña
    cursor.execute("SELECT nombre_completo FROM usuarios WHERE usuario=? AND password=?", (user, pwd))
    resultado = cursor.fetchone()
    conn.close()

    if resultado:
        # resultado[0] es el nombre completo
        return jsonify({"success": True, "nombre": resultado[0]})
    else:
        return jsonify({"success": False, "message": "Usuario o contraseña incorrectos"})

# --- 3. ARRANCAR EL SERVIDOR ---
if __name__ == '__main__':
    # Verificar que la DB existe antes de arrancar
    if not os.path.exists(DB_NAME):
        init_db()
    
    # Arrancar en el puerto 5000
    print("Servidor corriendo... Esperando conexiones del Frontend.")
    app.run(debug=True, host='0.0.0.0', port=5000)