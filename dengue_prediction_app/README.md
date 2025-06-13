# 🦟 Predictor de Dengue - CABA y AMBA

Aplicación web para predecir casos de dengue en la Ciudad Autónoma de Buenos Aires (CABA) y el Área Metropolitana de Buenos Aires (AMBA) utilizando machine learning.

## 📋 Características

- **Predicción en tiempo real** de casos de dengue por departamento/comuna
- **Interfaz intuitiva** con formularios validados
- **Soporte para CABA y AMBA** (55 departamentos/comunas)
- **Variables climáticas** (temperatura, humedad, precipitación)
- **Google Trends** como indicador de interés público
- **Datos históricos opcionales** para mejorar precisión
- **Niveles de riesgo** visuales (Bajo, Moderado, Alto, Muy Alto)

## 🏗️ Arquitectura

```
dengue_prediction_app/
├── backend/
│   └── app.py              # Servidor Flask con API
├── templates/
│   └── index.html          # Frontend HTML
├── static/
│   ├── style.css           # Estilos CSS
│   └── script.js           # Lógica JavaScript
├── requirements.txt        # Dependencias Python
└── README.md              # Este archivo
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8+
- Modelo entrenado (`dengue_model_optimized.joblib`) en el directorio raíz del proyecto

### 1. Instalar dependencias

```bash
cd dengue_prediction_app
pip install -r requirements.txt
```

### 2. Verificar modelo

Asegúrate de que el archivo `dengue_model_optimized.joblib` esté en el directorio padre:

```
ds_tpo/
├── dengue_model_optimized.joblib  # ← Modelo entrenado
└── dengue_prediction_app/
    ├── backend/
    └── ...
```

### 3. Ejecutar la aplicación

```bash
cd backend
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`

## 📊 Uso de la Aplicación

### 1. Seleccionar Ubicación
- **Provincia**: CABA o Buenos Aires
- **Departamento/Comuna**: Se cargan automáticamente según la provincia

### 2. Configurar Período
- **Año**: 2024-2030
- **Semana Epidemiológica**: 1-52

### 3. Ingresar Condiciones Climáticas
- **Temperatura Promedio**: °C (últimas 2 semanas)
- **Humedad Promedio**: % (últimas 2 semanas)
- **Precipitación Acumulada**: mm (últimas 2 semanas)
- **Google Trends**: 0-100 (interés de búsqueda "dengue")

### 4. Datos Históricos (Opcional)
- Casos de dengue de las últimas 1-4 semanas
- Mejora la precisión de la predicción

### 5. Obtener Predicción
- **Número de casos** estimados
- **Nivel de riesgo** con código de colores
- **Detalles** demográficos y climáticos

## 🎯 Niveles de Riesgo

| Casos | Nivel | Color |
|-------|-------|-------|
| 0-2 | Bajo | 🟢 Verde |
| 3-5 | Moderado | 🟡 Amarillo |
| 6-10 | Alto | 🟠 Naranja |
| 11+ | Muy Alto | 🔴 Rojo |

## 🔧 API Endpoints

### `GET /`
Página principal de la aplicación

### `GET /api/departments`
Lista de departamentos/comunas disponibles

**Respuesta:**
```json
{
  "success": true,
  "departments": [
    {
      "id": 1,
      "name": "Comuna 1",
      "province": "CABA",
      "population": 205886,
      "area": 23.4
    }
  ]
}
```

### `POST /api/predict`
Realizar predicción de casos de dengue

**Parámetros:**
```json
{
  "department_id": 1,
  "year": 2025,
  "week": 10,
  "temperature": 25.5,
  "humidity": 70.0,
  "precipitation": 30.0,
  "trends_dengue": 20,
  "cases_lag1": 0,  // Opcional
  "cases_lag2": 0,  // Opcional
  "cases_lag3": 0,  // Opcional
  "cases_lag4": 0   // Opcional
}
```

**Respuesta:**
```json
{
  "success": true,
  "prediction": {
    "cases": 3,
    "risk_level": "Moderado",
    "risk_color": "#ffc107",
    "department": "Comuna 1",
    "province": "CABA",
    "week": 10,
    "year": 2025
  }
}
```

### `GET /api/health`
Estado del servidor y modelo

**Respuesta:**
```json
{
  "success": true,
  "status": "healthy",
  "model_loaded": true
}
```

## 🧠 Modelo de Machine Learning

El modelo utiliza las siguientes características:

### Variables Climáticas
- `trends_dengue`: Google Trends búsquedas "dengue"
- `prec_2weeks`: Precipitación acumulada 2 semanas
- `temp_2weeks_avg`: Temperatura promedio 2 semanas
- `humd_2weeks_avg`: Humedad promedio 2 semanas

### Variables Demográficas
- `densidad_pob`: Densidad poblacional (hab/km²)

### Variables Temporales
- `week_sin`, `week_cos`: Componentes estacionales

### Variables Autoregresivas (Opcionales)
- `cases_lag1-4`: Casos de dengue semanas anteriores
- `trends_lag1`: Google Trends semana anterior
- `cases_diff`: Diferencia casos semana actual vs anterior
- `cases_ma_3weeks`: Media móvil 3 semanas
- `cases_max_4weeks`: Máximo casos 4 semanas

### Variables de Eventos Extremos
- `extreme_rain`: Lluvia extrema (>P90)
- `heat_wave`: Ola de calor (>P90)
- `extreme_humidity`: Humedad extrema (>P90)
- `dengue_favorable`: Condiciones climáticas favorables

### Encoding de Departamentos
- One-hot encoding para los 10 departamentos con más casos históricos

## 🛠️ Desarrollo

### Estructura del Backend (Flask)
- `load_model()`: Carga el modelo entrenado
- `create_prediction_features()`: Genera features para predicción
- `predict()`: Endpoint principal de predicción

### Frontend (HTML/CSS/JavaScript)
- **Validación en tiempo real** de formularios
- **Carga dinámica** de departamentos por provincia
- **Interfaz responsive** para móviles y desktop
- **Manejo de errores** y estados de carga

## 🔍 Troubleshooting

### Error: "Modelo no cargado"
- Verificar que `dengue_model_optimized.joblib` existe
- Comprobar permisos de lectura del archivo
- Revisar logs del servidor para errores específicos

### Error: "Error cargando departamentos"
- Verificar conectividad de red
- Comprobar que el servidor Flask está ejecutándose
- Revisar consola del navegador para errores JavaScript

### Predicciones inconsistentes
- Verificar rangos de valores de entrada
- Asegurar que los datos climáticos son realistas
- Considerar usar datos históricos para mejor precisión

## 📈 Mejoras Futuras

- [ ] **Visualizaciones**: Gráficos de tendencias históricas
- [ ] **Mapas interactivos**: Visualización geográfica de riesgo
- [ ] **API de clima**: Integración automática con servicios meteorológicos
- [ ] **Alertas**: Sistema de notificaciones por email/SMS
- [ ] **Exportación**: Descarga de reportes en PDF/Excel
- [ ] **Múltiples modelos**: Ensemble de diferentes algoritmos

## 📄 Licencia

Este proyecto es parte del Trabajo Práctico de Ciencia de Datos y está destinado para fines educativos y de investigación.

## 👥 Contribuciones

Para reportar bugs o sugerir mejoras, por favor crear un issue en el repositorio del proyecto.

---

**⚠️ Disclaimer**: Esta aplicación es para fines educativos y de investigación. Las predicciones no deben ser utilizadas como única fuente para decisiones de salud pública sin validación adicional por expertos en epidemiología. 