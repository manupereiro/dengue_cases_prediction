# 🦟 Predicción de Casos de Dengue

Sistema completo para la predicción de casos de dengue en CABA y AMBA, utilizando variables climáticas, demográficas, Google Trends y datos históricos.

---

## 📁 Estructura del Proyecto

```
dengue_cases_prediction/
│
├── 📁 data/                           # Datasets originales y procesados
│   ├── dengue_not_processed/          # Datos crudos de dengue por año
│   ├── dengue_trends/                 # Datos de Google Trends procesados
│   ├── zones/                         # Datos climáticos por zona
│   └── *.csv                          # Datasets procesados y unificados
│
├── 📁 processors/                     # Scripts de procesamiento de datos
│   ├── process_dengue_*.py            # Procesamiento por año (2018-2025)
│   ├── climate_data_processor.py      # Procesamiento de datos climáticos
│   ├── google_trends_collector.py     # Recolección de Google Trends
│   ├── unify_dengue_datasets.py       # Unificación de todos los datasets
│   └── eda_comprehensive.py           # Análisis exploratorio de datos
│
├── 📁 model_training/                 # Entrenamiento del modelo
│   └── train_dengue_model.py          # Script principal de entrenamiento
│
├── 📁 dengue_prediction_app/          # 🌐 Aplicación web con Flask
│   ├── 📁 backend/
│   │   ├── app.py                     # API Flask para predicciones
│   │   └── dengue_model_optimized.joblib
│   ├── 📁 static/
│   │   ├── script.js                  # JavaScript del frontend
│   │   └── style.css                  # Estilos CSS
│   ├── 📁 templates/
│   │   └── index.html                 # Página principal
│   ├── run_app.py                     # Script para lanzar la aplicación
│   └── requirements.txt               # Dependencias específicas de la app
│
├── 📁 graficos_eda/                   # Gráficos del análisis exploratorio
├── 📁 graficos.jpg/                   # Gráficos del modelo
├── dengue_model_optimized.joblib      # 🤖 Modelo entrenado principal
├── requirements.txt                   # Dependencias del proyecto
└── README.md                          # Este archivo
```

---

## 🚀 Cómo ejecutar la aplicación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo (opcional)

Si no tienes el archivo `dengue_model_optimized.joblib`:

```bash
python model_training/train_dengue_model.py
```

### 3. Ejecutar la aplicación web

```bash
cd dengue_prediction_app
python run_app.py
```

### 4. Abrir en el navegador

Visita: [http://localhost:5000](http://localhost:5000)

---

## 📊 Características de la Aplicación

### 🎯 Funcionalidades principales

- **Selección geográfica**: Elige entre CABA (15 comunas) y Buenos Aires (40 departamentos)
- **Variables climáticas**: Temperatura, humedad, precipitación
- **Google Trends**: Tendencias de búsqueda relacionadas con dengue
- **Datos históricos**: Casos de dengue de las últimas 4 semanas
- **Predicción inteligente**: Usa XGBoost optimizado con 30+ features
- **Nivel de riesgo**: Clasificación automática (Bajo, Moderado, Alto, Muy Alto)

### 🎨 Interfaz de usuario

- Diseño moderno y responsive
- Validación en tiempo real
- Animaciones y feedback visual
- Compatible con dispositivos móviles

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **Python**
- **Flask** - Framework web
- **XGBoost** - Modelo de machine learning
- **Scikit-learn** - Preprocesamiento y métricas
- **Pandas & NumPy** - Manipulación de datos
- **Joblib** - Serialización del modelo

### Frontend
- **HTML5 & CSS3**
- **JavaScript**
- **Bootstrap** - Framework CSS
- **Font Awesome** - Iconos

### Datos
- **Google Trends API (Pytrends)** - Tendencias de búsqueda
- **Datos climáticos** - Por zonas geográficas
- **Datos epidemiológicos** - Casos históricos de dengue

---

## 📈 Pipeline de Datos

1. **Recolección**: Datos de dengue, clima y Google Trends
2. **Procesamiento**: Limpieza y transformación por año
3. **Enriquecimiento**: Agregado de población, superficie y tendencias
4. **Unificación**: Creación del dataset final
5. **Entrenamiento**: Modelo XGBoost con optimización de hiperparámetros
6. **Despliegue**: Aplicación web para predicciones en tiempo real

---

## 💡 Ejemplo de Uso

1. **Selecciona ubicación**: Elige provincia y departamento/comuna
2. **Ingresa datos climáticos**: Temperatura (°C), humedad (%), precipitación (mm)
3. **Configura Google Trends**: Nivel de búsquedas relacionadas (0-100)
4. **Datos históricos**: Casos de dengue de las últimas 4 semanas
5. **Obtén predicción**: Número de casos esperados y nivel de riesgo

---

## 📋 Estructura de Datos

### Variables del modelo (30 features):
- **Geográficas**: Departamento, población, superficie
- **Temporales**: Semana, mes, año, tendencias estacionales
- **Climáticas**: Temperatura, humedad, precipitación
- **Tendencias**: Google Trends para términos relacionados
- **Históricas**: Casos de dengue de semanas anteriores (lag features)

---

## 🎯 Métricas del Modelo

- **Algoritmo**: XGBoost Regressor optimizado
- **Features**: 30 variables predictoras
- **Validación**: Cross-validation temporal
- **Métricas**: MAE, RMSE, R²

---

## 📬 Contacto

Proyecto desarrollado para fines educativos y de investigación.

**Equipo**: Grupo 2  
**Curso**: Ciencia de Datos  
**Año**: 2024-2025

---

## 📄 Licencia

Este proyecto es de uso educativo y de investigación.
