// --- JAVASCRIPT: Lógica de Interfaz ---

// 1. Navegación
function nav(sectionId) {
    // Oculta todas las secciones
    document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
    // Desactiva todos los botones del menú
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    
    // Muestra la sección deseada
    document.getElementById(sectionId).classList.add('active');
    // Activa el botón que fue presionado
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
}

// 2. Autenticación (Simulada con LocalStorage)
function toggleAuth() {
    const login = document.getElementById('login-form');
    const reg = document.getElementById('reg-form');
    
    // Alternar visibilidad
    if (login.style.display === 'none') {
        login.style.display = 'block';
        reg.style.display = 'none';
    } else {
        login.style.display = 'none';
        reg.style.display = 'block';
    }
    clearErrors();
}

function clearErrors() {
    document.getElementById('log-error').innerText = '';
    document.getElementById('reg-error').innerText = '';
}

function register() {
    const name = document.getElementById('reg-name').value;
    const user = document.getElementById('reg-user').value;
    const pass = document.getElementById('reg-pass').value;

    // Validación de campos vacíos
    if(!name || !user || !pass) {
        document.getElementById('reg-error').innerText = "⚠ Llena todos los campos";
        return;
    }
    
    // Verificar si existe (Simulación)
    if(localStorage.getItem('user_' + user)) {
        document.getElementById('reg-error').innerText = "⚠ Usuario ya existe";
        return;
    }

    // Guardar usuario
    localStorage.setItem('user_' + user, JSON.stringify({name, pass}));
    alert("✅ Registro exitoso. Inicia sesión.");
    toggleAuth();
}

function login() {
    const user = document.getElementById('log-user').value;
    const pass = document.getElementById('log-pass').value;

    if(!user || !pass) {
        document.getElementById('log-error').innerText = "⚠ Faltan datos";
        return;
    }

    const stored = localStorage.getItem('user_' + user);
    if(!stored) {
        document.getElementById('log-error').innerText = "❌ Usuario no encontrado";
        return;
    }

    const data = JSON.parse(stored);
    if(data.pass !== pass) {
        document.getElementById('log-error').innerText = "❌ Contraseña incorrecta";
        return;
    }

    // Éxito: Ocultar login y mostrar nombre
    document.getElementById('auth-overlay').style.display = 'none';
    document.getElementById('user-display').innerText = data.name;
}

function logout() {
    // Limpiar campos y volver a mostrar overlay
    document.getElementById('log-user').value = '';
    document.getElementById('log-pass').value = '';
    document.getElementById('auth-overlay').style.display = 'flex';
}