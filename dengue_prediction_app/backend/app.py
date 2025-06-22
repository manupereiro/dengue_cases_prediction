#!/usr/bin/env python3
"""
Backend Flask para predicción de casos de dengue
Carga el modelo entrenado y proporciona API para predicciones
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')
CORS(app)

# Variables globales para el modelo y datos
model = None
scaler = None
onehot_encoder = None
feature_names = None
departments_data = None

# Mapeo de departamentos CABA y AMBA
DEPARTMENTS = {
    # CABA (IDs 1-15)
    1: {"name": "Comuna 1", "province": "CABA", "population": 205886, "area": 23.4},
    2: {"name": "Comuna 2", "province": "CABA", "population": 157932, "area": 6.4},
    3: {"name": "Comuna 3", "province": "CABA", "population": 187237, "area": 6.8},
    4: {"name": "Comuna 4", "province": "CABA", "population": 218245, "area": 21.5},
    5: {"name": "Comuna 5", "province": "CABA", "population": 179005, "area": 6.1},
    6: {"name": "Comuna 6", "province": "CABA", "population": 176076, "area": 6.9},
    7: {"name": "Comuna 7", "province": "CABA", "population": 220591, "area": 9.7},
    8: {"name": "Comuna 8", "province": "CABA", "population": 187237, "area": 21.9},
    9: {"name": "Comuna 9", "province": "CABA", "population": 161797, "area": 6.5},
    10: {"name": "Comuna 10", "province": "CABA", "population": 166538, "area": 6.3},
    11: {"name": "Comuna 11", "province": "CABA", "population": 189832, "area": 7.4},
    12: {"name": "Comuna 12", "province": "CABA", "population": 200116, "area": 7.2},
    13: {"name": "Comuna 13", "province": "CABA", "population": 231331, "area": 12.3},
    14: {"name": "Comuna 14", "province": "CABA", "population": 225970, "area": 13.9},
    15: {"name": "Comuna 15", "province": "CABA", "population": 182574, "area": 11.2},
    
    # AMBA - Buenos Aires (IDs 16-55)
    16: {"name": "Almirante Brown", "province": "Buenos Aires", "population": 552902, "area": 129.3},
    17: {"name": "Avellaneda", "province": "Buenos Aires", "population": 342677, "area": 55.6},
    18: {"name": "Berazategui", "province": "Buenos Aires", "population": 324244, "area": 188.1},
    19: {"name": "Berisso", "province": "Buenos Aires", "population": 88470, "area": 135.0},
    20: {"name": "Brandsen", "province": "Buenos Aires", "population": 26367, "area": 1126.0},
    21: {"name": "Campana", "province": "Buenos Aires", "population": 94461, "area": 954.0},
    22: {"name": "Cañuelas", "province": "Buenos Aires", "population": 58322, "area": 1200.0},
    23: {"name": "Ensenada", "province": "Buenos Aires", "population": 56729, "area": 101.0},
    24: {"name": "Escobar", "province": "Buenos Aires", "population": 213619, "area": 277.4},
    25: {"name": "Esteban Echeverría", "province": "Buenos Aires", "population": 300959, "area": 120.2},
    26: {"name": "Exaltación de la Cruz", "province": "Buenos Aires", "population": 29805, "area": 662.0},
    27: {"name": "Ezeiza", "province": "Buenos Aires", "population": 163722, "area": 226.5},
    28: {"name": "Florencio Varela", "province": "Buenos Aires", "population": 426005, "area": 190.1},
    29: {"name": "General Las Heras", "province": "Buenos Aires", "population": 14889, "area": 1040.0},
    30: {"name": "General Rodríguez", "province": "Buenos Aires", "population": 87042, "area": 360.2},
    31: {"name": "General San Martín", "province": "Buenos Aires", "population": 414196, "area": 56.0},
    32: {"name": "Hurlingham", "province": "Buenos Aires", "population": 181241, "area": 35.3},
    33: {"name": "Ituzaingó", "province": "Buenos Aires", "population": 167824, "area": 38.5},
    34: {"name": "José C. Paz", "province": "Buenos Aires", "population": 265981, "area": 50.4},
    35: {"name": "La Matanza", "province": "Buenos Aires", "population": 1775816, "area": 323.0},
    36: {"name": "Lanús", "province": "Buenos Aires", "population": 459263, "area": 45.0},
    37: {"name": "La Plata", "province": "Buenos Aires", "population": 654324, "area": 940.4},
    38: {"name": "Lomas de Zamora", "province": "Buenos Aires", "population": 616279, "area": 89.0},
    39: {"name": "Luján", "province": "Buenos Aires", "population": 106273, "area": 763.0},
    40: {"name": "Marcos Paz", "province": "Buenos Aires", "population": 54181, "area": 318.0},
    41: {"name": "Malvinas Argentinas", "province": "Buenos Aires", "population": 322375, "area": 63.1},
    42: {"name": "Moreno", "province": "Buenos Aires", "population": 452505, "area": 186.0},
    43: {"name": "Merlo", "province": "Buenos Aires", "population": 528494, "area": 200.0},
    44: {"name": "Morón", "province": "Buenos Aires", "population": 321109, "area": 55.7},
    45: {"name": "Pilar", "province": "Buenos Aires", "population": 299077, "area": 352.0},
    46: {"name": "Presidente Perón", "province": "Buenos Aires", "population": 81141, "area": 113.5},
    47: {"name": "Quilmes", "province": "Buenos Aires", "population": 582943, "area": 94.0},
    48: {"name": "San Fernando", "province": "Buenos Aires", "population": 163240, "area": 877.0},
    49: {"name": "San Isidro", "province": "Buenos Aires", "population": 292878, "area": 48.4},
    50: {"name": "San Miguel", "province": "Buenos Aires", "population": 276190, "area": 83.0},
    51: {"name": "San Vicente", "province": "Buenos Aires", "population": 53433, "area": 666.0},
    52: {"name": "Tigre", "province": "Buenos Aires", "population": 376381, "area": 360.0},
    53: {"name": "Tres de Febrero", "province": "Buenos Aires", "population": 340071, "area": 46.0},
    54: {"name": "Vicente López", "province": "Buenos Aires", "population": 269420, "area": 33.4},
    55: {"name": "Zárate", "province": "Buenos Aires", "population": 114269, "area": 1190.0}
}

def load_model():
    """Cargar el modelo entrenado y sus componentes"""
    global model, scaler, onehot_encoder, feature_names
    
    try:
        # Lista de posibles ubicaciones del modelo
        possible_paths = [
            'dengue_model_optimized.joblib',  # En el directorio actual (backend)
            '../dengue_model_optimized.joblib',  # En el directorio padre (dengue_prediction_app)
            '../../dengue_model_optimized.joblib',  # En el directorio raíz del proyecto
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dengue_model_optimized.joblib'),  # Ruta absoluta al directorio de la app
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dengue_model_optimized.joblib')  # Ruta absoluta al directorio raíz
        ]
        
        model_path = None
        for path in possible_paths:
            print(f"🔍 Buscando modelo en: {os.path.abspath(path)}")
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            print(f"📁 Cargando modelo desde: {os.path.abspath(model_path)}")
            model_data = joblib.load(model_path)
            
            # El modelo puede estar guardado de diferentes formas
            if isinstance(model_data, dict):
                # Buscar el modelo con diferentes claves posibles
                model = model_data.get('model') or model_data.get('best_model')
                scaler = model_data.get('scaler')
                onehot_encoder = model_data.get('onehot_encoder')
                feature_names = model_data.get('feature_names')
                print(f"📊 Modelo cargado como diccionario con claves: {list(model_data.keys())}")
                
                if not model:
                    print("❌ No se encontró modelo en las claves 'model' o 'best_model'")
                    return False
            else:
                # Si es solo el modelo
                model = model_data
                print(f"📊 Modelo cargado directamente: {type(model_data)}")
                
            print(f"✅ Modelo cargado exitosamente desde: {model_path}")
            if feature_names:
                print(f"🔧 Features esperadas: {len(feature_names)}")
            else:
                print("⚠️  No se encontraron nombres de features guardados")
            
            # Verificar que el modelo tiene el método predict
            if hasattr(model, 'predict'):
                print("✅ Modelo tiene método predict")
            else:
                print("❌ Modelo no tiene método predict")
                return False
                
            return True
        else:
            print("❌ No se encontró el modelo en ninguna ubicación:")
            for path in possible_paths:
                print(f"   - {os.path.abspath(path)}")
            return False
            
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_prediction_features(data):
    """
    Crear las features necesarias para la predicción basándose en los datos de entrada
    """
    try:
        # Datos básicos
        dept_id = int(data['department_id'])
        year = int(data['year'])
        week = int(data['week'])
        trends_dengue = float(data['trends_dengue'])
        precipitation = float(data['precipitation'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        
        # Obtener datos del departamento
        dept_info = DEPARTMENTS.get(dept_id)
        if not dept_info:
            raise ValueError(f"Departamento ID {dept_id} no encontrado")
        
        # Calcular densidad poblacional
        density = dept_info['population'] / dept_info['area']
        
        # Features temporales (estacionalidad)
        week_sin = np.sin(2 * np.pi * week / 52)
        week_cos = np.cos(2 * np.pi * week / 52)
        
        # Features de eventos extremos (usando umbrales aproximados)
        extreme_rain = 1 if precipitation > 50 else 0  # >P90 aproximado
        heat_wave = 1 if temperature > 28 else 0      # >P90 aproximado
        extreme_humidity = 1 if humidity > 80 else 0   # >P90 aproximado
        
        # Condiciones favorables para dengue
        dengue_favorable = 1 if (temperature > 25 and humidity > 60 and precipitation > 20) else 0
        
        # Features autoregresivas (usar valores por defecto si no se proporcionan)
        cases_lag1 = float(data.get('cases_lag1', 0))
        cases_lag2 = float(data.get('cases_lag2', 0))
        cases_lag3 = float(data.get('cases_lag3', 0))
        cases_lag4 = float(data.get('cases_lag4', 0))
        trends_lag1 = float(data.get('trends_lag1', trends_dengue * 0.9))  # Aproximación
        
        # Features de tendencia
        cases_diff = cases_lag1 - cases_lag2 if cases_lag2 > 0 else 0
        cases_ma_3weeks = (cases_lag1 + cases_lag2 + cases_lag3) / 3 if cases_lag3 > 0 else cases_lag1
        cases_max_4weeks = max(cases_lag1, cases_lag2, cases_lag3, cases_lag4)
        
        # Crear DataFrame con las features
        features_dict = {
            'trends_dengue': trends_dengue,
            'prec_2weeks': precipitation,
            'temp_2weeks_avg': temperature,
            'humd_2weeks_avg': humidity,
            'densidad_pob': density,
            'week_sin': week_sin,
            'week_cos': week_cos,
            'cases_lag1': cases_lag1,
            'cases_lag2': cases_lag2,
            'cases_lag3': cases_lag3,
            'cases_lag4': cases_lag4,
            'trends_lag1': trends_lag1,
            'cases_diff': cases_diff,
            'cases_ma_3weeks': cases_ma_3weeks,
            'cases_max_4weeks': cases_max_4weeks,
            'extreme_rain': extreme_rain,
            'heat_wave': heat_wave,
            'extreme_humidity': extreme_humidity,
            'dengue_favorable': dengue_favorable
        }
        
        # Encoding de departamentos (top 10)
        top_departments = [1, 2, 3, 4, 5, 35, 36, 37, 38, 47]  # IDs más comunes
        dept_encoded = dept_id if dept_id in top_departments else 99
        
        # One-hot encoding simulado (crear todas las columnas dept_X)
        for i in range(11):  # Asumiendo 11 categorías (10 top + 1 other)
            features_dict[f'dept_{i}'] = 1 if (i < 10 and dept_encoded == top_departments[i]) or (i == 10 and dept_encoded == 99) else 0
        
        return pd.DataFrame([features_dict])
        
    except Exception as e:
        raise ValueError(f"Error creando features: {str(e)}")

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/departments')
def get_departments():
    """Obtener lista de departamentos disponibles"""
    departments_list = []
    for dept_id, info in DEPARTMENTS.items():
        departments_list.append({
            'id': dept_id,
            'name': info['name'],
            'province': info['province'],
            'population': info['population'],
            'area': info['area']
        })
    
    return jsonify({
        'success': True,
        'departments': departments_list
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones"""
    try:
        if not model:
            return jsonify({
                'success': False,
                'error': 'Modelo no cargado'
            }), 500
        
        data = request.json
        
        # Validar datos requeridos
        required_fields = ['department_id', 'year', 'week', 'trends_dengue', 
                          'precipitation', 'temperature', 'humidity']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Campo requerido faltante: {field}'
                }), 400
        
        # Crear features para predicción
        X = create_prediction_features(data)
        
        # Reordenar features según el modelo
        if feature_names:
            X_ordered = X[feature_names]
        else:
            X_ordered = X
        
        # Aplicar escalado a features continuas
        if scaler:
            continuous_features = [f for f in ['trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg',
                                              'densidad_pob', 'cases_lag1', 'cases_lag2', 'cases_lag3', 'cases_lag4',
                                              'trends_lag1', 'cases_diff', 'cases_ma_3weeks', 'cases_max_4weeks'] 
                                  if f in X_ordered.columns]
            
            X_scaled = X_ordered.copy()
            X_scaled[continuous_features] = scaler.transform(X_ordered[continuous_features])
        else:
            X_scaled = X_ordered
        
        # Realizar predicción
        prediction_log = model.predict(X_scaled)[0]
        
        # Revertir transformación logarítmica
        prediction = np.expm1(prediction_log)
        prediction = max(0, round(prediction))  # Asegurar que sea positivo y entero
        
        # Obtener información del departamento
        dept_info = DEPARTMENTS[int(data['department_id'])]
        
        # Calcular nivel de riesgo
        if prediction <= 2:
            risk_level = "Bajo"
            risk_color = "#28a745"
        elif prediction <= 5:
            risk_level = "Moderado"
            risk_color = "#ffc107"
        elif prediction <= 10:
            risk_level = "Alto"
            risk_color = "#fd7e14"
        else:
            risk_level = "Muy Alto"
            risk_color = "#dc3545"
        
        return jsonify({
            'success': True,
            'prediction': {
                'cases': int(prediction),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'department': dept_info['name'],
                'province': dept_info['province'],
                'week': int(data['week']),
                'year': int(data['year'])
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Endpoint de salud"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask para predicción de dengue...")
    
    # Cargar modelo
    if load_model():
        print("✅ Modelo cargado correctamente")
    else:
        print("⚠️  Modelo no cargado - algunas funciones pueden no estar disponibles")
    
    # Iniciar servidor
    app.run(debug=True, host='0.0.0.0', port=5000) 