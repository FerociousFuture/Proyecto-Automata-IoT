// DIRECCIÓN DEL SERVIDOR
const API_URL = window.location.origin; // Asume que se sirve desde el mismo lugar

// --- DEFINICIÓN DE HECHIZOS (Basado en tu PDF) ---
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

// --- NAVEGACIÓN ---
function nav(sectionId) {
    document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');
    
    // Si navegamos al dashboard y estamos logueados, asegurar que el listener esté activo
    if (sectionId === 'dashboard' && sessionStorage.getItem('activeUser')) {
        startAIListener();
    }
}

// --- AUTENTICACIÓN ---
function toggleAuth() {
    const login = document.getElementById('login-form');
    const reg = document.getElementById('reg-form');
    if (login.style.display === 'none') {
        login.style.display = 'block'; reg.style.display = 'none';
    } else {
        login.style.display = 'none'; reg.style.display = 'block';
    }
    document.querySelectorAll('.error-msg').forEach(e => e.innerText = '');
}

async function login() {
    const user = document.getElementById('log-user').value;
    const pass = document.getElementById('log-pass').value;
    
    if(!user || !pass) return;

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
            startAIListener(); // Iniciar detección al loguearse
        } else {
            document.getElementById('log-error').innerText = data.message;
        }
    } catch (e) {
        console.error(e);
        document.getElementById('log-error').innerText = "Error de conexión con el servidor";
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
            alert("Registro exitoso, inicia sesión.");
            toggleAuth();
        } else {
            document.getElementById('reg-error').innerText = data.message;
        }
    } catch (e) {
        console.error(e);
    }
}

function logout() {
    sessionStorage.removeItem('activeUser');
    location.reload();
}

// --- LÓGICA DEL HISTORIAL DE MOVIMIENTOS ---

let pollingInterval = null;

function startAIListener() {
    // 1. Pedir al backend que inicie el hilo de escucha serial
    fetch(`${API_URL}/api/start_practice`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log("Sistema IA:", data.message);
            // 2. Iniciar polling
            if (!pollingInterval) {
                pollingInterval = setInterval(fetchLogs, 1000); // Revisar cada segundo
            }
        })
        .catch(err => console.error("No se pudo iniciar el servicio de IA:", err));
}

function fetchLogs() {
    fetch(`${API_URL}/api/get_live_logs`)
        .then(res => res.json())
        .then(data => {
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(jsonStr => {
                    try {
                        // Parsear el mensaje JSON que viene de Python
                        const msg = JSON.parse(jsonStr);
                        
                        // Si es un tipo 'gesture', lo agregamos a la tabla
                        if (msg.type === 'gesture' && msg.data) {
                            addSpellToTable(msg.data.name, msg.data.score);
                            
                            // Efecto visual rápido en el dashboard
                            flashDashboard(msg.data.name);
                        } else if (msg.type === 'error') {
                            console.error("Error Hardware:", msg.text);
                        }
                    } catch (e) {
                        console.warn("Mensaje no JSON recibido:", jsonStr);
                    }
                });
            }
        })
        .catch(err => console.error("Error polling logs:", err));
}

function addSpellToTable(spellName, score) {
    const tableBody = document.getElementById('command-log');
    if (!tableBody) return;

    // Buscar definición bonita del hechizo
    // Busca coincidencias parciales (ej. "Lumos_Nox" coincide con "Lumos")
    let def = SPELL_DEFINITIONS["default"];
    
    // Búsqueda exacta primero, luego parcial
    if (SPELL_DEFINITIONS[spellName]) {
        def = SPELL_DEFINITIONS[spellName];
    } else {
        // Buscar si alguna clave está contenida en el nombre recibido
        const foundKey = Object.keys(SPELL_DEFINITIONS).find(key => spellName.includes(key));
        if (foundKey) def = SPELL_DEFINITIONS[foundKey];
        else def = { icon: "help_outline", desc: "Hechizo desconocido", color: "#ccc" };
    }

    const row = document.createElement('tr');
    row.style.animation = "fadeEffect 0.5s"; // Animación de entrada
    
    const time = new Date().toLocaleTimeString();
    
    row.innerHTML = `
        <td style="color: #888;">${time}</td>
        <td>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span class="material-icons" style="color: ${def.color};">${def.icon}</span>
                <strong>${spellName}</strong>
            </div>
        </td>
        <td>${def.desc} <span style="font-size:0.8em; color:#666;">(${Math.round(score)}%)</span></td>
    `;

    // Insertar al inicio
    tableBody.insertBefore(row, tableBody.firstChild);

    // Mantener solo 10 filas
    if (tableBody.children.length > 10) {
        tableBody.removeChild(tableBody.lastChild);
    }
}

function flashDashboard(spellName) {
    // Un pequeño efecto visual en el borde del panel cuando llega un hechizo
    const dash = document.getElementById('dashboard');
    dash.style.transition = "box-shadow 0.2s";
    dash.style.boxShadow = "inset 0 0 20px var(--secondary-color)";
    setTimeout(() => {
        dash.style.boxShadow = "none";
    }, 300);
}

// Inicializar si ya hay sesión
document.addEventListener('DOMContentLoaded', () => {
    if(sessionStorage.getItem('activeUser')) {
        document.getElementById('auth-overlay').style.display = 'none';
        document.getElementById('user-display').innerText = sessionStorage.getItem('activeUser');
        startAIListener();
    }
});