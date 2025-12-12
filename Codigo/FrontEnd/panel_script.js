// Simulación de los modelos que se cargarían desde tu backend de Python
// En la implementación real con Flask, esto se obtendría de un endpoint /api/models
const MOCKED_GESTURES = [
    { name: "Lumos", modelStatus: "ENTRENADO", description: "Línea vertical recta (Luz +)" },
    { name: "Nox", modelStatus: "ENTRENADO", description: "Línea vertical recta (Luz -)" },
    { name: "Ascendio", modelStatus: "ENTRENADO", description: "Arco hacia arriba (Volumen +)" },
    { name: "Petrificus Totalus", modelStatus: "NO ENTRENADO", description: "Cuadrado (Sueño del robot)" },
    { name: "Wingardium Leviosa", modelStatus: "NO ENTRENADO", description: "Bucle ascendente (Flotar)" }
];

function loadGesturesTable() {
    const container = document.getElementById('gesture_list_container');
    let html = '<table class="gesture-table">';
    html += '<thead><tr><th>Hechizo</th><th>Descripción</th><th>Estado ML</th><th>Acción</th></tr></thead><tbody>';

    MOCKED_GESTURES.forEach(gesture => {
        const isTrained = gesture.modelStatus === "ENTRENADO";
        html += `
            <tr>
                <td>${gesture.name}</td>
                <td>${gesture.description}</td>
                <td><span style="color: ${isTrained ? 'green' : 'red'};">${gesture.modelStatus}</span></td>
                <td>
                    <button class="practice-btn" onclick="startPractice('${gesture.name}')">PRACTICAR</button>
                </td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function startPractice(gestureName) {
    const feedback = document.getElementById('visual_feedback');
    const status = document.getElementById('feedback_display');
    
    // 1. Mostrar retroalimentación al usuario
    status.innerHTML = `<h2>Practicando: ${gestureName}</h2>`;
    feedback.innerHTML = `
        <p>Prepara tu varita. Intenta dibujar el patrón para ${gestureName}.</p>
        <p style="font-weight: bold;">(Tu Python Backend enviaría aquí el comando de grabación/comparación)</p>
    `;

    // 2. Aquí iría la lógica AJAX que llama al Backend de Python
    // EJEMPLO: 
    // fetch('/api/start_practice?gesture=' + gestureName)
    // .then(response => response.json())
    // .then(data => console.log('Backend iniciado'));
}

function triggerTraining() {
    alert("Activando script de entrenamiento en Raspberry Pi. Esto tomará un tiempo...");
    // Aquí iría la lógica AJAX que llama al endpoint en Flask/Django
    // fetch('/api/train_all', { method: 'POST' });
}

// Inicializar al cargar la página
window.onload = function() {
    // Inicializar el estado de los modelos
    document.getElementById('model_status').innerHTML = 'Listo';
    document.getElementById('last_model').innerHTML = MOCKED_GESTURES.filter(g => g.modelStatus === "ENTRENADO").map(g => g.name).pop() || 'Ninguno';
    
    loadGesturesTable();
};