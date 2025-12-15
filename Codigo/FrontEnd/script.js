const API_URL = window.location.origin;

// --- DICCIONARIO DE HECHIZOS (Sincronizado con tus modelos) ---
const SPELL_DEFINITIONS = {
    "Lumos": { icon: "light_mode", desc: "Aumenta la luz del autómata", color: "#ffeb3b" },
    "Nox": { icon: "dark_mode", desc: "Disminuye la luz del autómata", color: "#757575" },
    "Lumos_Nox": { icon: "toggle_on", desc: "Control de luz (Interruptor)", color: "#fff176" },
    "Ascendio": { icon: "volume_up", desc: "Aumenta el volumen", color: "#4caf50" },
    "Descendo": { icon: "volume_down", desc: "Disminuye el volumen", color: "#ff9800" },
    "Stupefy": { icon: "flash_on", desc: "Hechizo de aturdimiento", color: "#f44336" },
    "Expelliarmus": { icon: "pan_tool", desc: "Hechizo de desarme", color: "#2196f3" },
    "Wingardium": { icon: "flight", desc: "Levitación (Servo motor)", color: "#ce93d8" },
    "Avada": { icon: "power_settings_new", desc: "Apagado de emergencia", color: "#d50000" }
};

// --- VARIABLES GLOBALES PARA DIBUJO ---
let isDrawingActive = false;
let drawInterval = null;
let currentX = 400; // Centro X inicial (ajustar al canvas.width / 2)
let currentY = 175; // Centro Y inicial (ajustar al canvas.height / 2)
const SENSITIVITY = 3.5; // Multiplicador de movimiento

// --- NAVEGACIÓN Y QOL ---
function nav(sectionId) {
    // 1. Ocultar todas las secciones
    document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
    
    // 2. Desactivar todos los botones de navegación
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    
    // 3. Activar sección seleccionada
    const targetSection = document.getElementById(sectionId);
    if (targetSection) targetSection.classList.add('active');

    // 4. Iluminar botón correspondiente
    const targetBtn = document.getElementById('nav-' + sectionId);
    if (targetBtn) targetBtn.classList.add('active');

    // 5. Gestión inteligente de recursos
    if (sectionId === 'trace') {
        initCanvas(); // Asegurar tamaño correcto al mostrar
        startDrawingLoop();
    } else {
        stopDrawingLoop();
    }
}

// --- LÓGICA DEL CANVAS (VISUALIZER) ---
function initCanvas() {
    const canvas = document.getElementById('wandCanvas');
    const container = document.getElementById('drawing-area');
    // Ajustar canvas al contenedor
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    // Resetear posición al centro si es necesario
    if (currentX === 400) { 
        currentX = canvas.width / 2;
        currentY = canvas.height / 2;
    }
}

function startDrawingLoop() {
    if (isDrawingActive) return;
    isDrawingActive = true;
    document.getElementById('btn-draw-text').innerText = "Pausar Dibujo";
    
    const canvas = document.getElementById('wandCanvas');
    const ctx = canvas.getContext('2d');
    
    // Estilo del trazo "mágico"
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--secondary-color').trim() || '#00e5ff';
    ctx.shadowBlur = 10;
    ctx.shadowColor = ctx.strokeStyle;

    // Efecto de desvanecimiento (trail)
    // En lugar de borrar todo, dibujamos un rectángulo negro semitransparente
    // para que los trazos viejos se desvanezcan.
    
    drawInterval = setInterval(async () => {
        try {
            // Fetch de datos rápidos
            const res = await fetch(`${API_URL}/api/sensor_stream`);
            if (!res.ok) return;
            const points = await res.json();
            
            if (points && points.length > 0) {
                
                // Procesar cada punto recibido
                points.forEach(p => {
                    // Integración: Posición = Posición + Velocidad (Giroscopio)
                    // Nota: Invertimos signos según orientación típica de la varita
                    const deltaX = -p.gz * SENSITIVITY; // Yaw controla X
                    const deltaY = -p.gx * SENSITIVITY;  // Pitch controla Y
                    
                    const nextX = currentX + deltaX;
                    const nextY = currentY + deltaY;
                    
                    // Dibujar línea
                    ctx.beginPath();
                    ctx.moveTo(currentX, currentY);
                    ctx.lineTo(nextX, nextY);
                    ctx.stroke();
                    
                    // Actualizar posición (con límites de rebote suave)
                    currentX = Math.max(0, Math.min(canvas.width, nextX));
                    currentY = Math.max(0, Math.min(canvas.height, nextY));
                });

                // Actualizar UI de coordenadas
                const coordsEl = document.getElementById('coords-display');
                if(coordsEl) coordsEl.innerText = `X: ${Math.round(currentX)} | Y: ${Math.round(currentY)}`;
            }
            
            // Efecto fade: Pinta negro con alpha 0.05 cada frame
            // Esto es opcional, si prefieres que el dibujo se quede, comenta estas líneas
            ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Dibujar cursor brillante en la punta actual
            // (Lo dibujamos y lo borramos en el siguiente fillRect, creando el efecto de que se mueve)
            ctx.save();
            ctx.fillStyle = '#fff';
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#fff';
            ctx.beginPath();
            ctx.arc(currentX, currentY, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();

        } catch (e) {
            console.error("Error drawing stream:", e);
        }
    }, 50); // 20 FPS aprox
}

function stopDrawingLoop() {
    isDrawingActive = false;
    const btn = document.getElementById('btn-draw-text');
    if(btn) btn.innerText = "Reanudar Dibujo";
    
    if (drawInterval) {
        clearInterval(drawInterval);
        drawInterval = null;
    }
}

function toggleDrawing() {
    if (isDrawingActive) stopDrawingLoop();
    else startDrawingLoop();
}

function clearCanvas() {
    const canvas = document.getElementById('wandCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Resetear al centro
    currentX = canvas.width / 2;
    currentY = canvas.height / 2;
}

// --- SISTEMA DE AUTENTICACIÓN ---

function toggleAuth() {
    const login = document.getElementById('login-form');
    const reg = document.getElementById('reg-form');
    
    if (login.style.display === 'none') {
        login.style.display = 'block'; 
        reg.style.display = 'none';
    } else {
        login.style.display = 'none'; 
        reg.style.display = 'block';
    }
    document.querySelectorAll('.error-msg').forEach(e => e.innerText = '');
}

async function login() {
    const user = document.getElementById('log-user').value;
    const pass = document.getElementById('log-pass').value;
    
    if(!user || !pass) {
        document.getElementById('log-error').innerText = "Campos vacíos";
        return;
    }

    try {
        const res = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: user, password: pass})
        });
        const data = await res.json();
        
        if (data.success) {
            document.getElementById('auth-overlay').style.display = 'none';
            document.getElementById('user-display').innerText = data.nombre;
            sessionStorage.setItem('activeUser', data.nombre);
            startAIListener(); // Activar el backend
        } else {
            document.getElementById('log-error').innerText = data.message;
        }
    } catch (e) {
        console.error(e);
        document.getElementById('log-error').innerText = "Error de conexión";
    }
}

async function register() {
    const name = document.getElementById('reg-name').value;
    const user = document.getElementById('reg-user').value;
    const pass = document.getElementById('reg-pass').value;

    try {
        const res = await fetch(`${API_URL}/registro`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({fullName: name, username: user, password: pass})
        });
        const data = await res.json();
        if (data.success) {
            alert("Registro exitoso. Por favor inicia sesión.");
            toggleAuth();
        } else {
            document.getElementById('reg-error').innerText = data.message;
        }
    } catch (e) { console.error(e); }
}

function logout() {
    sessionStorage.removeItem('activeUser');
    location.reload();
}

// --- COMUNICACIÓN CON BACKEND (AI) ---

let pollingInterval = null;

function startAIListener() {
    // 1. Pedir al backend que arranque el hilo de detección
    fetch(`${API_URL}/api/start_practice`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log("Sistema IA:", data.message);
            // 2. Iniciar polling de logs
            if (!pollingInterval) {
                pollingInterval = setInterval(fetchLogs, 1000); 
            }
        })
        .catch(err => console.error("Error inicio IA:", err));
}

function fetchLogs() {
    fetch(`${API_URL}/api/get_live_logs`)
        .then(res => res.json())
        .then(data => {
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(jsonStr => {
                    try {
                        const msg = JSON.parse(jsonStr);
                        // Solo nos interesan los mensajes tipo 'gesture' para la tabla
                        if (msg.type === 'gesture' && msg.data) {
                            addSpellToTable(msg.data.name, msg.data.score);
                        }
                    } catch (e) { 
                        // Ignorar logs que no sean JSON válido
                    }
                });
            }
        })
        .catch(console.error);
}

function addSpellToTable(spellName, score) {
    const tableBody = document.getElementById('command-log');
    
    // Limpiar mensaje de "Esperando..." si existe
    if(tableBody.firstElementChild && tableBody.firstElementChild.innerText.includes("Esperando")) {
        tableBody.innerHTML = "";
    }

    // Buscar definición del hechizo
    let def = SPELL_DEFINITIONS[spellName];
    
    // Si no es exacto, buscar coincidencia parcial (ej. Lumos_Nox -> Lumos)
    if (!def) {
         const foundKey = Object.keys(SPELL_DEFINITIONS).find(key => spellName.includes(key));
         def = foundKey ? SPELL_DEFINITIONS[foundKey] : { icon: "help_outline", desc: "Hechizo desconocido", color: "#ccc" };
    }

    const row = document.createElement('tr');
    // Animación de entrada definida en CSS
    row.style.animation = "fadeEffect 0.5s";
    
    row.innerHTML = `
        <td style="color:#888;">${new Date().toLocaleTimeString()}</td>
        <td>
            <div style="display:flex; align-items:center; gap:10px;">
                <span class="material-icons" style="color:${def.color};">${def.icon}</span>
                <strong>${spellName}</strong>
            </div>
        </td>
        <td>${def.desc} <span style="font-size:0.8em;color:#666;">(${Math.round(score)}%)</span></td>
    `;
    
    // Insertar al principio
    tableBody.insertBefore(row, tableBody.firstChild);
    
    // Limitar filas a 8 para no saturar
    if (tableBody.children.length > 8) {
        tableBody.removeChild(tableBody.lastChild);
    }
}

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    // Verificar si hay sesión activa
    if(sessionStorage.getItem('activeUser')) {
        document.getElementById('auth-overlay').style.display = 'none';
        document.getElementById('user-display').innerText = sessionStorage.getItem('activeUser');
        startAIListener();
    }
    
    // Configurar canvas inicial
    initCanvas();
    window.addEventListener('resize', initCanvas);
});