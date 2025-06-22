// Variables globales
let departments = [];
let isLoading = false;

// Configuración de la API
const API_BASE_URL = window.location.origin;

// Inicialización cuando se carga la página
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    console.log('🚀 Inicializando aplicación...');
    await loadDepartments();
    setupEventListeners();
    await checkServerHealth();
    console.log('✅ Aplicación inicializada');
}

async function loadDepartments() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/departments`);
        const data = await response.json();
        if (data.success) {
            departments = data.departments;
            console.log(`📍 Cargados ${departments.length} departamentos`);
        } else {
            throw new Error('Error cargando departamentos');
        }
    } catch (error) {
        console.error('❌ Error cargando departamentos:', error);
        showError('Error cargando la lista de departamentos');
    }
}

function setupEventListeners() {
    // Cambio de provincia
    const provinceSelect = document.getElementById('province');
    provinceSelect.addEventListener('change', handleProvinceChange);

    // Envío del formulario
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handleFormSubmit);

    // Validación en tiempo real
    setupRealTimeValidation();
}

function handleProvinceChange(event) {
    const selectedProvince = event.target.value;
    const departmentSelect = document.getElementById('department');
    departmentSelect.innerHTML = '<option value="">Seleccionar departamento</option>';
    if (selectedProvince) {
        const filteredDepartments = departments.filter(dept => dept.province === selectedProvince);
        filteredDepartments.forEach(dept => {
            const option = document.createElement('option');
            option.value = dept.id;
            option.textContent = dept.name;
            option.dataset.population = dept.population;
            option.dataset.area = dept.area;
            departmentSelect.appendChild(option);
        });
        departmentSelect.disabled = false;
        console.log(`📍 Cargados ${filteredDepartments.length} departamentos para ${selectedProvince}`);
    } else {
        departmentSelect.disabled = true;
    }
}

function setupRealTimeValidation() {
    // Validación de semana epidemiológica
    const weekInput = document.getElementById('week');
    weekInput.addEventListener('input', function() {
        const value = parseInt(this.value);
        if (value < 1 || value > 52) {
            this.setCustomValidity('La semana debe estar entre 1 y 52');
        } else {
            this.setCustomValidity('');
        }
    });

    // Validación de temperatura
    const tempInput = document.getElementById('temperature');
    tempInput.addEventListener('input', function() {
        const value = parseFloat(this.value);
        if (value < -10 || value > 50) {
            this.setCustomValidity('Temperatura debe estar entre -10°C y 50°C');
        } else {
            this.setCustomValidity('');
        }
    });

    // Validación de humedad
    const humidityInput = document.getElementById('humidity');
    humidityInput.addEventListener('input', function() {
        const value = parseFloat(this.value);
        if (value < 0 || value > 100) {
            this.setCustomValidity('Humedad debe estar entre 0% y 100%');
        } else {
            this.setCustomValidity('');
        }
    });

    // Validación de precipitación
    const precipInput = document.getElementById('precipitation');
    precipInput.addEventListener('input', function() {
        const value = parseFloat(this.value);
        if (value < 0 || value > 1000) {
            this.setCustomValidity('Precipitación debe estar entre 0mm y 1000mm');
        } else {
            this.setCustomValidity('');
        }
    });

    // Validación de Google Trends
    const trendsInput = document.getElementById('trends_dengue');
    trendsInput.addEventListener('input', function() {
        const value = parseInt(this.value);
        if (value < 0 || value > 100) {
            this.setCustomValidity('Google Trends debe estar entre 0 y 100');
        } else {
            this.setCustomValidity('');
        }
    });
}

async function handleFormSubmit(event) {
    event.preventDefault();
    if (isLoading) return;
    if (!validateForm()) return;
    const formData = collectFormData();
    await makePrediction(formData);
}

function validateForm() {
    const form = document.getElementById('predictionForm');
    if (!form.checkValidity()) {
        form.reportValidity();
        return false;
    }
    const department = document.getElementById('department').value;
    if (!department) {
        showError('Por favor selecciona un departamento');
        return false;
    }
    return true;
}

function collectFormData() {
    return {
        department_id: parseInt(document.getElementById('department').value),
        year: parseInt(document.getElementById('year').value),
        week: parseInt(document.getElementById('week').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        precipitation: parseFloat(document.getElementById('precipitation').value),
        trends_dengue: parseInt(document.getElementById('trends_dengue').value),
        cases_lag1: parseInt(document.getElementById('cases_lag1').value) || 0,
        cases_lag2: parseInt(document.getElementById('cases_lag2').value) || 0,
        cases_lag3: parseInt(document.getElementById('cases_lag3').value) || 0,
        cases_lag4: parseInt(document.getElementById('cases_lag4').value) || 0
    };
}

async function makePrediction(data) {
    try {
        showLoading(true);
        isLoading = true;
        console.log('📊 Enviando datos para predicción:', data);
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (result.success) {
            console.log('✅ Predicción exitosa:', result.prediction);
            displayPredictionResult(result.prediction, data);
        } else {
            throw new Error(result.error || 'Error en la predicción');
        }
    } catch (error) {
        console.error('❌ Error en predicción:', error);
        showError(`Error realizando predicción: ${error.message}`);
    } finally {
        showLoading(false);
        isLoading = false;
    }
}

function displayPredictionResult(prediction, inputData) {
    const resultsCard = document.getElementById('resultsCard');
    const resultContainer = document.getElementById('predictionResult');
    const department = departments.find(d => d.id === inputData.department_id);
    const resultHTML = `
        <div class="result-main">
            <div class="result-cases">${prediction.cases}</div>
            <div class="result-risk" style="background-color: ${prediction.risk_color}">
                Riesgo ${prediction.risk_level}
            </div>
            <div class="result-location">
                ${prediction.department}, ${prediction.province}
            </div>
            <div class="result-period">
                Semana ${prediction.week}, ${prediction.year}
            </div>
        </div>
        <div class="result-details">
            <div class="result-detail">
                <div class="result-detail-title">Población</div>
                <div class="result-detail-value">${department.population.toLocaleString()}</div>
            </div>
            <div class="result-detail">
                <div class="result-detail-title">Área</div>
                <div class="result-detail-value">${department.area} km²</div>
            </div>
            <div class="result-detail">
                <div class="result-detail-title">Densidad</div>
                <div class="result-detail-value">${Math.round(department.population / department.area)} hab/km²</div>
            </div>
            <div class="result-detail">
                <div class="result-detail-title">Temperatura</div>
                <div class="result-detail-value">${inputData.temperature}°C</div>
            </div>
            <div class="result-detail">
                <div class="result-detail-title">Humedad</div>
                <div class="result-detail-value">${inputData.humidity}%</div>
            </div>
            <div class="result-detail">
                <div class="result-detail-title">Precipitación</div>
                <div class="result-detail-value">${inputData.precipitation}mm</div>
            </div>
        </div>
    `;
    resultContainer.innerHTML = resultHTML;
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    const predictBtn = document.getElementById('predictBtn');
    if (show) {
        loading.style.display = 'flex';
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';
    } else {
        loading.style.display = 'none';
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-chart-line"></i> Predecir Casos de Dengue';
    }
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
    console.error('❌ Error mostrado al usuario:', message);
}

async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        if (data.success) {
            console.log('✅ Servidor saludable, modelo cargado:', data.model_loaded);
            if (!data.model_loaded) {
                showError('Advertencia: El modelo no está cargado en el servidor');
            }
        } else {
            throw new Error('Servidor no responde correctamente');
        }
    } catch (error) {
        console.error('❌ Error verificando servidor:', error);
        showError('Error conectando con el servidor');
    }
}

// Utilidades
function formatNumber(num) {
    return new Intl.NumberFormat('es-AR').format(num);
}

function getCurrentWeek() {
    const now = new Date();
    const start = new Date(now.getFullYear(), 0, 1);
    const diff = now - start;
    const oneWeek = 1000 * 60 * 60 * 24 * 7;
    return Math.ceil(diff / oneWeek);
}

// Establecer semana actual por defecto
document.addEventListener('DOMContentLoaded', function() {
    const weekInput = document.getElementById('week');
    if (weekInput) {
        weekInput.value = getCurrentWeek();
    }
});

// Manejo de errores globales
window.addEventListener('error', function(event) {
    console.error('❌ Error global:', event.error);
    showError('Ha ocurrido un error inesperado');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('❌ Promesa rechazada:', event.reason);
    showError('Error de conexión o procesamiento');
    event.preventDefault();
});

console.log('📱 Script cargado correctamente');