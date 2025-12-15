import serial
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import numpy as np

# --- CONFIGURACIÓN ---
SENSOR_COLS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']
NUM_SENSOR_VALUES = 6
TRAIL_LENGTH = 200  # Número de puntos a mostrar en la trayectoria
CANVAS_SIZE = 10    # Tamaño del lienzo (-10 a +10)

# --- Variables globales para la trayectoria ---
trajectory_x = deque(maxlen=TRAIL_LENGTH)
trajectory_y = deque(maxlen=TRAIL_LENGTH)

# Posición actual del "pincel" (simulada por integración de velocidades)
current_x = 0.0
current_y = 0.0

# Puerto serial
ser = None

# Escala de integración ajustable (AUMENTADA SIGNIFICATIVAMENTE)
GYRO_SCALE = 0.5  # Factor de escala para velocidades angulares (x50 más grande)
MOVEMENT_THRESHOLD = 0.1  # Umbral mínimo de movimiento para considerar
RESET_SPEED = 0.98  # Velocidad de retorno al centro cuando no hay movimiento

# --- CONFIGURACIÓN DE LA VISUALIZACIÓN ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-CANVAS_SIZE, CANVAS_SIZE)
ax.set_ylim(-CANVAS_SIZE, CANVAS_SIZE)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X (Horizontal)', fontsize=12)
ax.set_ylabel('Y (Vertical)', fontsize=12)
ax.set_title('🪄 Trayectoria del Hechizo en Tiempo Real', fontsize=14, fontweight='bold')

# Línea de trayectoria
line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
trail, = ax.plot([], [], 'cyan', linewidth=1, alpha=0.4)  # Estela difusa
current_point, = ax.plot([], [], 'ro', markersize=10)  # Punto actual

# Texto de información
info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Texto de valores en vivo (DEBUG)
debug_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                     verticalalignment='bottom', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# --- FUNCIÓN DE INICIALIZACIÓN ---
def init():
    """Inicializa la animación."""
    line.set_data([], [])
    trail.set_data([], [])
    current_point.set_data([], [])
    return line, trail, current_point, info_text, debug_text

# --- FUNCIÓN DE ACTUALIZACIÓN ---
def update(frame):
    """Actualiza la visualización con nuevos datos del sensor."""
    global current_x, current_y, ser
    
    if ser is None or not ser.is_open:
        return line, trail, current_point, info_text, debug_text
    
    try:
        if ser.in_waiting > 0:
            line_data = ser.readline().decode('latin-1').strip()
            parts = line_data.split(',')
            
            if len(parts) == NUM_SENSOR_VALUES:
                try:
                    # Parsear valores del sensor
                    gyro_x = float(parts[0])
                    gyro_y = float(parts[1])
                    gyro_z = float(parts[2])
                    acc_x = float(parts[3])
                    acc_y = float(parts[4])
                    acc_z = float(parts[5])
                    
                    # --- ESTRATEGIA MEJORADA DE MAPEO 2D ---
                    # Usar Gyro_X y Gyro_Y directamente como velocidades
                    # INVERSIÓN CORREGIDA: negativo para que el movimiento sea natural
                    
                    # Calcular magnitud del movimiento
                    movement_magnitude = np.sqrt(gyro_x**2 + gyro_y**2)
                    
                    # Solo actualizar si hay movimiento significativo
                    if movement_magnitude > MOVEMENT_THRESHOLD:
                        # Integración directa con inversión correcta
                        delta_x = -gyro_y * GYRO_SCALE  # Gyro_Y controla X (horizontal, invertido)
                        delta_y = gyro_x * GYRO_SCALE   # Gyro_X controla Y (vertical, SIN invertir)
                        
                        current_x += delta_x
                        current_y += delta_y
                    else:
                        # Retorno suave al centro cuando no hay movimiento
                        current_x *= RESET_SPEED
                        current_y *= RESET_SPEED
                    
                    # Limitar al área del canvas
                    current_x = np.clip(current_x, -CANVAS_SIZE, CANVAS_SIZE)
                    current_y = np.clip(current_y, -CANVAS_SIZE, CANVAS_SIZE)
                    
                    # Agregar a la trayectoria SIEMPRE (incluso si no se mueve mucho)
                    trajectory_x.append(current_x)
                    trajectory_y.append(current_y)
                    
                    # Actualizar gráficos
                    if len(trajectory_x) > 1:
                        line.set_data(list(trajectory_x), list(trajectory_y))
                        
                        # Estela con gradiente (últimos 50 puntos)
                        trail_len = min(50, len(trajectory_x))
                        trail.set_data(
                            list(trajectory_x)[-trail_len:], 
                            list(trajectory_y)[-trail_len:]
                        )
                    
                    current_point.set_data([current_x], [current_y])
                    
                    # Actualizar texto de información
                    info_text.set_text(
                        f'Posición: ({current_x:.2f}, {current_y:.2f})\n'
                        f'Movimiento: {movement_magnitude:.2f}°/s\n'
                        f'Puntos: {len(trajectory_x)}/{TRAIL_LENGTH}'
                    )
                    
                    # Texto de DEBUG con valores crudos
                    debug_text.set_text(
                        f'DEBUG:\n'
                        f'Gyro_X: {gyro_x:+.2f} | Gyro_Y: {gyro_y:+.2f}\n'
                        f'Delta_X: {delta_x if movement_magnitude > MOVEMENT_THRESHOLD else 0:+.3f} | '
                        f'Delta_Y: {delta_y if movement_magnitude > MOVEMENT_THRESHOLD else 0:+.3f}'
                    )
                    
                except ValueError as e:
                    debug_text.set_text(f'Error parsing: {e}')
    
    except Exception as e:
        print(f"Error en update: {e}")
    
    return line, trail, current_point, info_text, debug_text

# --- FUNCIÓN DE RESETEO (tecla 'r') ---
def on_key(event):
    """Maneja eventos de teclado."""
    global current_x, current_y, trajectory_x, trajectory_y, GYRO_SCALE
    
    if event.key == 'r':
        # Resetear trayectoria
        trajectory_x.clear()
        trajectory_y.clear()
        current_x = 0.0
        current_y = 0.0
        print("🔄 Trayectoria reseteada")
    
    elif event.key == 'c':
        # Centrar vista
        ax.set_xlim(-CANVAS_SIZE, CANVAS_SIZE)
        ax.set_ylim(-CANVAS_SIZE, CANVAS_SIZE)
        print("🎯 Vista centrada")
    
    elif event.key == '+' or event.key == '=':
        # Aumentar sensibilidad
        GYRO_SCALE *= 1.5
        print(f"⬆️  Sensibilidad aumentada: {GYRO_SCALE:.2f}")
    
    elif event.key == '-' or event.key == '_':
        # Disminuir sensibilidad
        GYRO_SCALE /= 1.5
        print(f"⬇️  Sensibilidad disminuida: {GYRO_SCALE:.2f}")

# --- FUNCIÓN PRINCIPAL ---
def run_visualizer(serial_port, baud_rate):
    """
    Inicia el visualizador de trayectoria en tiempo real.
    
    Args:
        serial_port: Puerto serial del ESP32
        baud_rate: Velocidad de comunicación
    """
    global ser
    
    print("="*60)
    print("🪄 VISUALIZADOR DE TRAYECTORIA DE GESTOS")
    print("="*60)
    print(f"\n📡 Conectando a {serial_port} @ {baud_rate}...")
    
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print("✅ Conexión establecida")
    except serial.SerialException as e:
        print(f"❌ Error al conectar: {e}")
        return
    
    print("\n📋 CONTROLES:")
    print("   - Mueve la varita para dibujar")
    print("   - Presiona 'R' para resetear el dibujo")
    print("   - Presiona 'C' para centrar la vista")
    print("   - Presiona '+' para aumentar sensibilidad")
    print("   - Presiona '-' para disminuir sensibilidad")
    print("   - Cierra la ventana para salir")
    print(f"\n🎨 Sensibilidad inicial: {GYRO_SCALE}")
    print("   (Ajusta con +/- si el movimiento es muy lento/rápido)\n")
    
    # Conectar eventos de teclado
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Iniciar animación
    anim = FuncAnimation(
        fig, 
        update, 
        init_func=init,
        interval=50,  # Actualizar cada 50ms (20 FPS)
        blit=True,
        cache_frame_data=False
    )
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n⏹️  Visualizador detenido")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Conexión serial cerrada")

# --- MODO ALTERNATIVO: COMPARAR GESTO GRABADO ---
def visualize_from_csv(csv_file):
    """
    Visualiza un gesto desde un archivo CSV (para comparar con templates).
    
    Args:
        csv_file: Archivo CSV con datos del gesto
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Cargado '{csv_file}' ({len(df)} muestras)")
    except FileNotFoundError:
        print(f"❌ Archivo '{csv_file}' no encontrado")
        return
    
    # Integrar la trayectoria desde el CSV
    traj_x = [0.0]
    traj_y = [0.0]
    
    for i in range(len(df)):
        gyro_x = df.loc[i, 'Gyro_X']
        gyro_y = df.loc[i, 'Gyro_Y']
        
        # Usar la misma lógica que en tiempo real (CORREGIDA)
        delta_x = -gyro_y * GYRO_SCALE
        delta_y = gyro_x * GYRO_SCALE
        
        new_x = traj_x[-1] + delta_x
        new_y = traj_y[-1] + delta_y
        
        traj_x.append(new_x)
        traj_y.append(new_y)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(traj_x, traj_y, 'b-', linewidth=2, label='Trayectoria')
    ax.plot(traj_x[0], traj_y[0], 'go', markersize=12, label='Inicio')
    ax.plot(traj_x[-1], traj_y[-1], 'ro', markersize=12, label='Fin')
    
    # Agregar flechas para mostrar dirección
    skip = max(1, len(traj_x) // 10)  # Mostrar ~10 flechas
    for i in range(0, len(traj_x)-skip, skip):
        dx = traj_x[i+skip] - traj_x[i]
        dy = traj_y[i+skip] - traj_y[i]
        ax.arrow(traj_x[i], traj_y[i], dx, dy, 
                head_width=0.3, head_length=0.3, fc='blue', ec='blue', alpha=0.3)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (Horizontal)')
    ax.set_ylabel('Y (Vertical)')
    ax.set_title(f'Trayectoria del Gesto: {csv_file}', fontweight='bold')
    ax.legend()
    
    print(f"\n📊 Estadísticas:")
    print(f"   Rango X: [{min(traj_x):.2f}, {max(traj_x):.2f}]")
    print(f"   Rango Y: [{min(traj_y):.2f}, {max(traj_y):.2f}]")
    print(f"   Longitud total: {len(traj_x)} puntos")
    print(f"   Desplazamiento total: {np.sqrt((traj_x[-1]-traj_x[0])**2 + (traj_y[-1]-traj_y[0])**2):.2f}")
    
    plt.show()

# --- MODO: COMPARAR MÚLTIPLES GESTOS ---
def compare_gestures(*csv_files):
    """
    Compara visualmente las trayectorias de múltiples gestos.
    
    Args:
        *csv_files: Lista de archivos CSV a comparar
    """
    import pandas as pd
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta']
    
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            
            # Integrar trayectoria
            traj_x = [0.0]
            traj_y = [0.0]
            
            for i in range(len(df)):
                gyro_x = df.loc[i, 'Gyro_X']
                gyro_y = df.loc[i, 'Gyro_Y']
                
                delta_x = -gyro_y * GYRO_SCALE
                delta_y = gyro_x * GYRO_SCALE
                
                new_x = traj_x[-1] + delta_x
                new_y = traj_y[-1] + delta_y
                
                traj_x.append(new_x)
                traj_y.append(new_y)
            
            # Graficar
            color = colors[idx % len(colors)]
            gesture_name = csv_file.split('/')[-1].replace('.csv', '')
            
            ax.plot(traj_x, traj_y, color=color, linewidth=2, label=gesture_name, alpha=0.7)
            ax.plot(traj_x[0], traj_y[0], 'o', color=color, markersize=10)
            ax.plot(traj_x[-1], traj_y[-1], 's', color=color, markersize=10)
            
            print(f"✅ {gesture_name}: {len(df)} muestras, desplazamiento: {np.sqrt((traj_x[-1]-traj_x[0])**2 + (traj_y[-1]-traj_y[0])**2):.2f}")
            
        except FileNotFoundError:
            print(f"⚠️  '{csv_file}' no encontrado, omitiendo...")
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (Horizontal)')
    ax.set_ylabel('Y (Vertical)')
    ax.set_title('Comparación de Trayectorias de Gestos', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    
    plt.show()

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🪄 VISUALIZADOR DE TRAYECTORIAS DE GESTOS")
        print("="*70)
        print("\nMODOS DE USO:")
        
        print("\n1. VISUALIZAR EN TIEMPO REAL (Modo Pizarrón):")
        print("   python visualizer.py realtime <PUERTO> <BAUD>")
        print("   Ejemplo: python visualizer.py realtime /dev/ttyUSB0 115200")
        
        print("\n2. VISUALIZAR DESDE CSV (Modo Replay):")
        print("   python visualizer.py csv <archivo.csv>")
        print("   Ejemplo: python visualizer.py csv lumos.csv")
        
        print("\n3. COMPARAR MÚLTIPLES GESTOS:")
        print("   python visualizer.py compare <archivo1.csv> <archivo2.csv> ...")
        print("   Ejemplo: python visualizer.py compare lumos.csv expelliarmus.csv")
        
        print("\n" + "="*70)
        print("💡 TIPS:")
        print("   - GYRO_SCALE controla la sensibilidad (actual: 0.5)")
        print("   - En modo realtime, usa +/- para ajustar en tiempo real")
        print("   - Los valores se muestran en el panel DEBUG (abajo izquierda)")
        print("   - Si no ves movimiento, aumenta GYRO_SCALE en el código")
        print("="*70 + "\n")
    
    elif sys.argv[1] == 'realtime' and len(sys.argv) >= 3:
        port = sys.argv[2]
        baud = int(sys.argv[3]) if len(sys.argv) >= 4 else 115200
        run_visualizer(port, baud)
    
    elif sys.argv[1] == 'csv' and len(sys.argv) == 3:
        visualize_from_csv(sys.argv[2])
    
    elif sys.argv[1] == 'compare' and len(sys.argv) >= 3:
        compare_gestures(*sys.argv[2:])
    
    else:
        print("❌ Comando no reconocido. Usa sin argumentos para ver la ayuda.")