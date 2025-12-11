import time
import threading
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# --- CONFIGURACIÓN ---
OLED_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64

class AnimatedOLED:
    def __init__(self):
        # Configurar dispositivo
        serial = i2c(port=1, address=OLED_ADDRESS)
        self.device = ssd1306(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
        
        # Estado
        self.modo = "idle"  # puede ser "idle" o "figura"
        self.figura_actual = None
        self.running = True
        self.frame = 0
        
        # Caras para animación idle
        self.caras_idle = [
            # Cara normal parpadeando
            [
                "    ╭─────╮",
                "   │ ^  ^ │",
                "   │   ω  │",
                "    ╰─────╯"
            ],
            # Ojos cerrados
            [
                "    ╭─────╮",
                "   │ -  - │",
                "   │   ω  │",
                "    ╰─────╯"
            ],
            # Sonrisa
            [
                "    ╭─────╮",
                "   │ ^  ^ │",
                "   │  \_/  │",
                "    ╰─────╯"
            ]
        ]
        
        # Diccionario de figuras disponibles
        self.figuras = {
            "gato": [
                "  /\\_/\\",
                " ( o.o )",
                "  > ^ <",
                " /|   |\\",
                "(_|   |_)"
            ],
            "perro": [
                "  /^ ^\\",
                " / o o \\",
                "(   >  )",
                " | --- |",
                " U   U"
            ],
            "corazon": [
                "  ♥♥  ♥♥",
                " ♥  ♥♥  ♥",
                "♥   ♥    ♥",
                " ♥      ♥",
                "  ♥    ♥",
                "    ♥♥"
            ],
            "estrella": [
                "    *",
                "   ***",
                "  *****",
                " *******",
                "  *****",
                "   ***",
                "    *"
            ],
            "feliz": [
                "  \\(^o^)/",
                "    | |",
                "   / \\"
            ],
            "triste": [
                "   (T_T)",
                "    | |",
                "   / \\"
            ],
            "pulgar": [
                "    ___",
                "   /   \\",
                "  |  👍  |",
                "   \\___/",
                "     |"
            ]
        }
    
    def dibujar_idle(self):
        """Dibuja la animación idle con carita"""
        cara = self.caras_idle[self.frame % len(self.caras_idle)]
        
        with canvas(self.device) as draw:
            draw.rectangle(self.device.bounding_box, outline="black", fill="black")
            
            # Centrar la cara
            y_start = 15
            for i, linea in enumerate(cara):
                x = (OLED_WIDTH - len(linea) * 6) // 2
                draw.text((x, y_start + i * 12), linea, fill="white")
    
    def dibujar_figura(self, nombre_figura):
        """Dibuja una figura específica"""
        if nombre_figura not in self.figuras:
            return False
        
        figura = self.figuras[nombre_figura]
        
        with canvas(self.device) as draw:
            draw.rectangle(self.device.bounding_box, outline="black", fill="black")
            
            # Título
            titulo = f"< {nombre_figura.upper()} >"
            x_titulo = (OLED_WIDTH - len(titulo) * 6) // 2
            draw.text((x_titulo, 0), titulo, fill="white")
            
            # Centrar la figura
            y_start = 20
            for i, linea in enumerate(figura):
                x = (OLED_WIDTH - len(linea) * 6) // 2
                draw.text((x, y_start + i * 8), linea, fill="white")
        
        return True
    
    def mostrar_figura(self, nombre):
        """Cambia al modo figura y muestra por 3 segundos"""
        if nombre in self.figuras:
            self.modo = "figura"
            self.figura_actual = nombre
            self.dibujar_figura(nombre)
            
            # Volver a idle después de 3 segundos
            def volver_idle():
                time.sleep(3)
                self.modo = "idle"
                self.figura_actual = None
            
            threading.Thread(target=volver_idle, daemon=True).start()
            return True
        else:
            print(f"Figura '{nombre}' no encontrada. Disponibles: {list(self.figuras.keys())}")
            return False
    
    def loop_animacion(self):
        """Loop principal de animación"""
        while self.running:
            if self.modo == "idle":
                self.dibujar_idle()
                self.frame += 1
                time.sleep(0.5)  # Velocidad de parpadeo
            else:
                time.sleep(0.1)  # En modo figura, solo esperar
    
    def iniciar(self):
        """Inicia el loop de animación en un thread"""
        thread = threading.Thread(target=self.loop_animacion, daemon=True)
        thread.start()
        return thread
    
    def detener(self):
        """Detiene la animación y limpia"""
        self.running = False
        time.sleep(0.2)
        self.device.cleanup()
    
    def listar_figuras(self):
        """Lista todas las figuras disponibles"""
        return list(self.figuras.keys())

# --- PROGRAMA PRINCIPAL ---
if __name__ == "__main__":
    try:
        print("Iniciando OLED animado...")
        oled = AnimatedOLED()
        
        # Iniciar animación
        oled.iniciar()
        
        print("\n=== COMANDOS DISPONIBLES ===")
        print("Escribe el nombre de una figura para mostrarla:")
        print(f"Figuras: {', '.join(oled.listar_figuras())}")
        print("Escribe 'salir' para terminar\n")
        
        # Loop de comandos
        while True:
            comando = input("Comando: ").strip().lower()
            
            if comando == "salir":
                print("Cerrando...")
                break
            elif comando == "lista":
                print(f"Figuras disponibles: {', '.join(oled.listar_figuras())}")
            elif comando:
                if not oled.mostrar_figura(comando):
                    print("Intenta con otra figura o escribe 'lista' para ver opciones")
        
        oled.detener()
        print("Programa terminado")
        
    except Exception as e:
        print(f"ERROR: {e}")