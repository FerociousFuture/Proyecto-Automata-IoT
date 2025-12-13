// DIRECCIÓN DEL SERVIDOR PYTHON (BACKEND)
// Si estás en la misma PC, usa localhost. Si es la Raspberry, usa su IP.
const API_URL = "http://localhost:5000";

// --- NAVEGACIÓN (Igual que antes) ---
function nav(sectionId) {
    document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');
    if (event && event.currentTarget) event.currentTarget.classList.add('active');
}

// --- AUTENTICACIÓN CON PYTHON Y SQLITE ---

function toggleAuth() {
    const login = document.getElementById('login-form');
    const reg = document.getElementById('reg-form');
    
    if (login.style.display === 'none') {
        login.style.display = 'block'; reg.style.display = 'none';
    } else {
        login.style.display = 'none'; reg.style.display = 'block';
    }
    clearErrors();
}

function clearErrors() {
    document.getElementById('log-error').innerText = '';
    document.getElementById('reg-error').innerText = '';
}

// FUNCIÓN PARA REGISTRAR (CONECTADA A PYTHON)
async function register() {
    const name = document.getElementById('reg-name').value;
    const user = document.getElementById('reg-user').value;
    const pass = document.getElementById('reg-pass').value;

    if(!name || !user || !pass) {
        document.getElementById('reg-error').innerText = "⚠ Llena todos los campos";
        return;
    }

    try {
        // Enviamos los datos a Python
        const response = await fetch(`${API_URL}/registro`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fullName: name, username: user, password: pass })
        });

        const data = await response.json();

        if (data.success) {
            alert("✅ " + data.message);
            toggleAuth();
        } else {
            document.getElementById('reg-error').innerText = "❌ " + data.message;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('reg-error').innerText = "⚠ Error de conexión con el servidor (¿Está corriendo app.py?)";
    }
}

// FUNCIÓN PARA LOGIN (CONECTADA A PYTHON)
async function login() {
    const user = document.getElementById('log-user').value;
    const pass = document.getElementById('log-pass').value;

    if(!user || !pass) {
        document.getElementById('log-error').innerText = "⚠ Faltan datos";
        return;
    }

    try {
        // Preguntamos a Python si el usuario existe
        const response = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass })
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('auth-overlay').style.display = 'none';
            document.getElementById('user-display').innerText = data.nombre;
            // Guardamos sesión temporalmente por si recarga la página
            sessionStorage.setItem('activeUser', data.nombre);
        } else {
            document.getElementById('log-error').innerText = "❌ " + data.message;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('log-error').innerText = "⚠ Error de conexión. Asegúrate de ejecutar 'app.py'";
    }
}

function logout() {
    sessionStorage.removeItem('activeUser');
    location.reload(); // Recarga la página para reiniciar todo
}