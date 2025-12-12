// Simulación de los gestos que se obtendrían de tu Backend (Flask/Python)
// Estos datos vendrían de un endpoint como /api/gestures_status
const GESTURE_DATA = [
    // Se utilizan los hechizos del documento de proyecto
    { name: "Lumos", pattern: "Línea arriba", effect: "Luz +", trained: true },
    { name: "Nox", pattern: "Línea abajo", effect: "Luz -", trained: true },
    { name: "Ascendio", pattern: "Arco ascendente", effect: "Volumen +", trained: true },
    { name: "Descendo", pattern: "Arco descendente", effect: "Volumen -", trained: true },
    { name: "Petrificus Totalus", pattern: "Cuadrado", effect: "Inmoviliza Robot", trained: false },
    { name: "Wingardium Leviosa", pattern: "Bucle ascendente", effect: "Simula 'Flotar'", trained: true },
    { name: "Revelio", pattern: "Trazo en R", effect: "Diagnóstico/Batería", trained: false },
];

// Función para cargar la tabla de gestos
function loadGesturesTable() {
    const tbody = document.querySelector('#gesture_table tbody');
    tbody.innerHTML = ''; // Limpiar tabla
    let trainedCount = 0;

    GESTURE_DATA.forEach(gesture => {
        const isTrained = gesture.trained;
        const statusClass = isTrained ? 'status-trained' : 'status-untrained';
        if (isTrained) trainedCount++;

        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${gesture.name}</td>
            <td>${gesture.pattern}</td>
            <td>${gesture.effect}</td>
            <td><span class="${statusClass}">${isTrained ? 'Dominado' : 'Pendiente'}</span></td>
            <td>
                <button class="practice-btn" onclick="startPractice('${gesture.name}', '${gesture.pattern}')">PRACTICAR</button>
            </td>
        `;
    });
    
    document.getElementById('trained_count').innerHTML = trainedCount;
}

// Función que se activa al presionar el botón PRACTICAR
function startPractice(gestureName, pattern) {
    const feedbackTitle = document.getElementById('feedback_title');
    const feedbackDesc = document.getElementById('feedback_description');
    const visualModal = document.getElementById('visual_feedback');
    const practiceLog = document.getElementById('practice_log');
    
    // 1. Mostrar el modal de práctica
    feedbackTitle.innerHTML = `Patrón: ${gestureName.toUpperCase()}`;
    feedbackDesc.innerHTML = `El movimiento clave es: <strong>${pattern}</strong>.`;
    
    // Aquí se actualizaría la imagen del patrón (si tuvieras las imágenes de los trazos)
    document.getElementById('pattern_img').src = 'pattern_' + gestureName.toLowerCase().replace(/\s/g, '_') + '.png'; // EJEMPLO
    document.getElementById('pattern_img').alt = 'Patrón de ' + gestureName;

    // Abrir el modal
    visualModal.style.display = 'block';

    // 2. Simular la conexión y el inicio de la práctica
    practiceLog.innerHTML = `[${new Date().toLocaleTimeString()}] 🪄 Hechizo '${gestureName}' seleccionado.\n`;
    practiceLog.innerHTML += `[${new Date().toLocaleTimeString()}] 📡 Esperando movimiento de varita en tiempo real...`;

    // 3. Lógica para empezar a escuchar en el Backend de Python
    // (En un entorno real, enviarías una solicitud AJAX a tu Flask Server)
    // fetch('/api/monitor_gesture?name=' + gestureName) 
    // .then(response => console.log('Monitoreo iniciado'));
}

// Función para cerrar el modal
function closeModal() {
    document.getElementById('visual_feedback').style.display = 'none';
}

// Cerrar el modal con la tecla ESC
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});

// Inicializar al cargar la página
window.onload = function() {
    loadGesturesTable();
};