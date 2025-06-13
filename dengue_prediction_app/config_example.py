# Configuración de ejemplo para la aplicación de predicción de dengue

# Configuración del servidor Flask
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000
}

# Rutas de archivos
MODEL_PATH = 'dengue_model_optimized.joblib'
BACKUP_MODEL_PATH = '../dengue_model_optimized.joblib'

# Configuración de la API
API_CONFIG = {
    'MAX_REQUESTS_PER_MINUTE': 60,
    'ENABLE_CORS': True,
    'CORS_ORIGINS': ['http://localhost:3000', 'http://127.0.0.1:3000']
}

# Umbrales de riesgo (casos de dengue)
RISK_THRESHOLDS = {
    'LOW': 2,        # 0-2 casos: Riesgo Bajo
    'MODERATE': 5,   # 3-5 casos: Riesgo Moderado  
    'HIGH': 10,      # 6-10 casos: Riesgo Alto
    'VERY_HIGH': 11  # 11+ casos: Riesgo Muy Alto
}

# Colores para niveles de riesgo
RISK_COLORS = {
    'Bajo': '#28a745',      # Verde
    'Moderado': '#ffc107',  # Amarillo
    'Alto': '#fd7e14',      # Naranja
    'Muy Alto': '#dc3545'   # Rojo
}

# Validación de entrada
VALIDATION_RANGES = {
    'year': {'min': 2024, 'max': 2030},
    'week': {'min': 1, 'max': 52},
    'temperature': {'min': -10, 'max': 50},
    'humidity': {'min': 0, 'max': 100},
    'precipitation': {'min': 0, 'max': 1000},
    'trends_dengue': {'min': 0, 'max': 100}
}

# Umbrales para eventos extremos (percentil 90 aproximado)
EXTREME_THRESHOLDS = {
    'extreme_rain': 50,      # mm
    'heat_wave': 28,         # °C
    'extreme_humidity': 80,  # %
    'dengue_favorable': {
        'min_temp': 25,      # °C
        'min_humidity': 60,  # %
        'min_precipitation': 20  # mm
    }
}

# Configuración de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'dengue_app.log'
}

# Departamentos top 10 con más casos históricos
TOP_DEPARTMENTS = [1, 2, 3, 4, 5, 35, 36, 37, 38, 47]

# Mensajes de la aplicación
MESSAGES = {
    'model_not_loaded': 'El modelo de predicción no está disponible',
    'invalid_department': 'Departamento no válido',
    'prediction_error': 'Error realizando la predicción',
    'server_error': 'Error interno del servidor',
    'validation_error': 'Datos de entrada no válidos'
}

# Configuración de cache (para futuras mejoras)
CACHE_CONFIG = {
    'enabled': False,
    'timeout': 300,  # 5 minutos
    'max_size': 100
} 