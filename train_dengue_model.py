#!/usr/bin/env python3
"""
Script simplificado para entrenar modelo de predicción de dengue
Metodología tradicional con transformación logarítmica optimizada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class DenguePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.models = {}
        self.feature_names = None
        self.use_log_transform = True  # Transformación logarítmica para picos
        
    def load_and_diagnose_data(self):
        """Cargar y procesar datos básicos"""
        print("📊 Cargando y analizando datos...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        
        # Calcular densidad poblacional
        # Manejar superficies problemáticas
        caba_avg_surface = self.df[(self.df['province'] == 'CABA') & (self.df['sup_km2'] > 0)]['sup_km2'].mean()
        bsas_avg_surface = self.df[(self.df['province'] == 'Buenos Aires') & (self.df['sup_km2'] > 0)]['sup_km2'].mean()
        
        caba_mask = (self.df['province'] == 'CABA') & ((self.df['sup_km2'] == 0) | (self.df['sup_km2'].isnull()))
        bsas_mask = (self.df['province'] == 'Buenos Aires') & ((self.df['sup_km2'] == 0) | (self.df['sup_km2'].isnull()))
        
        self.df.loc[caba_mask, 'sup_km2'] = caba_avg_surface
        self.df.loc[bsas_mask, 'sup_km2'] = bsas_avg_surface
        
        # Eliminar filas con población nula y calcular densidad
        self.df = self.df.dropna(subset=['total_pobl'])
        self.df['densidad_pob'] = self.df['total_pobl'] / self.df['sup_km2']
        
        # Eliminar valores faltantes en features críticos
        critical_features = ['trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg', 'densidad_pob']
        self.df = self.df.dropna(subset=critical_features)
        
        print(f"Dataset final: {self.df.shape[0]} filas")
        return self.df
    
    def visualize_data(self):
        """Crear visualizaciones de los datos"""
        print("📈 Generando gráficos...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis del Dataset de Dengue', fontsize=16)
        
        # Tendencia anual
        yearly_cases = self.df.groupby('year')['target_cases'].sum()
        axes[0,0].plot(yearly_cases.index, yearly_cases.values, 'o-', linewidth=2)
        axes[0,0].set_title('Casos Totales por Año')
        axes[0,0].set_xlabel('Año')
        axes[0,0].set_ylabel('Casos')
        axes[0,0].grid(True, alpha=0.3)
        
        # Estacionalidad
        weekly_cases = self.df.groupby('week')['target_cases'].mean()
        axes[0,1].plot(weekly_cases.index, weekly_cases.values, 's-', color='orange', linewidth=2)
        axes[0,1].set_title('Estacionalidad por Semana')
        axes[0,1].set_xlabel('Semana')
        axes[0,1].set_ylabel('Casos Promedio')
        axes[0,1].grid(True, alpha=0.3)
        
        # Distribución de casos
        axes[1,0].hist(self.df['target_cases'], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1,0].set_title('Distribución de Casos')
        axes[1,0].set_xlabel('Casos')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].grid(True, alpha=0.3)
        
        # Densidad vs casos
        density_cases = self.df.groupby('departament_id').agg({
            'densidad_pob': 'first',
            'target_cases': 'mean'
        })
        axes[1,1].scatter(density_cases['densidad_pob'], density_cases['target_cases'], alpha=0.7, color='purple')
        axes[1,1].set_xlabel('Densidad Poblacional (hab/km²)')
        axes[1,1].set_ylabel('Casos Promedio')
        axes[1,1].set_title('Densidad vs Casos')
        axes[1,1].set_xscale('log')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Guardado: exploratory_analysis.png")
        
        # Matriz de correlación
        plt.figure(figsize=(10, 8))
        numeric_cols = ['trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg', 'densidad_pob', 'target_cases']
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Guardado: correlation_matrix.png")
    
    def analyze_target_distribution(self):
        """Analizar transformación logarítmica del target"""
        print("🔬 Analizando transformación logarítmica...")
        
        original_target = self.df['target_cases']
        log_target = np.log1p(original_target)
        
        print(f"Target original - Media: {original_target.mean():.1f}, Asimetría: {original_target.skew():.2f}")
        print(f"Target log1p - Media: {log_target.mean():.2f}, Asimetría: {log_target.skew():.2f}")
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Transformación Logarítmica del Target', fontsize=14)
        
        axes[0].hist(original_target, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0].set_title('Original')
        axes[0].set_xlabel('Casos')
        axes[0].set_ylabel('Frecuencia')
        
        axes[1].hist(log_target, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1].set_title('Log1p Transformado')
        axes[1].set_xlabel('log1p(Casos)')
        axes[1].set_ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.savefig('target_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Guardado: target_distribution_analysis.png")
        
        return {
            'original_skew': original_target.skew(),
            'log_skew': log_target.skew(),
            'improvement': original_target.skew() - log_target.skew()
        }
    
    def create_lag_features(self, df):
        """Crear features autoregresivas"""
        df_sorted = df.sort_values(['departament_id', 'year', 'week']).copy()
        
        # Lags básicos
        df_sorted['cases_lag1'] = df_sorted.groupby('departament_id')['target_cases'].shift(1)
        df_sorted['cases_lag2'] = df_sorted.groupby('departament_id')['target_cases'].shift(2)
        df_sorted['cases_lag3'] = df_sorted.groupby('departament_id')['target_cases'].shift(3)
        df_sorted['cases_lag4'] = df_sorted.groupby('departament_id')['target_cases'].shift(4)
        df_sorted['trends_lag1'] = df_sorted.groupby('departament_id')['trends_dengue'].shift(1)
        
        # Features de tendencia
        df_sorted['cases_diff'] = df_sorted.groupby('departament_id')['target_cases'].diff()
        df_sorted['cases_ma_3weeks'] = df_sorted.groupby('departament_id')['target_cases'].rolling(window=3).mean().reset_index(0, drop=True)
        df_sorted['cases_max_4weeks'] = df_sorted.groupby('departament_id')['target_cases'].rolling(window=4).max().reset_index(0, drop=True)
        
        return df_sorted
    
    def create_extreme_weather_features(self, df):
        """Crear features de eventos extremos"""
        df_processed = df.copy()
        
        # Umbrales de eventos extremos
        rain_p90 = df['prec_2weeks'].quantile(0.90)
        temp_p90 = df['temp_2weeks_avg'].quantile(0.90)
        humidity_p90 = df['humd_2weeks_avg'].quantile(0.90)
        
        df_processed['extreme_rain'] = (df_processed['prec_2weeks'] > rain_p90).astype(int)
        df_processed['heat_wave'] = (df_processed['temp_2weeks_avg'] > temp_p90).astype(int)
        df_processed['extreme_humidity'] = (df_processed['humd_2weeks_avg'] > humidity_p90).astype(int)
        
        # Condiciones favorables para dengue
        df_processed['dengue_favorable'] = (
            (df_processed['temp_2weeks_avg'] > 25) & 
            (df_processed['humd_2weeks_avg'] > 60) & 
            (df_processed['prec_2weeks'] > 20)
        ).astype(int)
        
        return df_processed
    
    def create_outbreak_labels(self, df, threshold=5):
        """Crear etiquetas de brote"""
        df_processed = df.copy()
        df_processed['outbreak_label'] = (df_processed['target_cases'] > threshold).astype(int)
        return df_processed
    
    def prepare_features(self):
        """Preparar features y aplicar transformación logarítmica"""
        print("🔧 Preparando features...")
        
        # Análisis de transformación
        log_analysis = self.analyze_target_distribution()
        
        # Crear features avanzadas
        self.df = self.create_lag_features(self.df)
        self.df = self.create_extreme_weather_features(self.df)
        self.df = self.create_outbreak_labels(self.df, threshold=5)
        
        # División temporal granular
        print("📅 División temporal personalizada:")
        print("   Train: 2018 - 6 primeros meses 2024 (hasta semana 26)")
        print("   Val: 3 meses siguientes 2024 (semanas 27-39)")  
        print("   Test: Últimos 3 meses 2024 + 2025 (semana 40+ y 2025)")
        
        # Train: 2018-2023 completos + primeros 6 meses de 2024 (semanas 1-26)
        train_condition = (
            (self.df['year'].isin([2018, 2019, 2020, 2021, 2022, 2023])) |
            ((self.df['year'] == 2024) & (self.df['week'] <= 12))
        )
        
        # Val: Semanas 27-39 de 2024 (aproximadamente 3 meses)
        val_condition = (
            (self.df['year'] == 2024) & 
            (self.df['week'] >= 13) & 
            (self.df['week'] <= 26)
        )
        
        # Test: Semanas 40-52 de 2024 + todo 2025
        test_condition = (
            ((self.df['year'] == 2024) & (self.df['week'] >= 27)) |
            (self.df['year'] == 2025)
        )
        
        self.train_df = self.df[train_condition].copy()
        self.val_df = self.df[val_condition].copy()
        self.test_df = self.df[test_condition].copy()
        
        print(f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # Verificar rangos de fechas
        if len(self.train_df) > 0:
            train_range = f"{self.train_df['year'].min()}-{self.train_df['year'].max()}"
            if self.train_df['year'].max() == 2024:
                train_range += f" (hasta semana {self.train_df[self.train_df['year']==2024]['week'].max()})"
            print(f"   Train range: {train_range}")
            
        if len(self.val_df) > 0:
            val_weeks = f"semanas {self.val_df['week'].min()}-{self.val_df['week'].max()}"
            print(f"   Val range: 2024 {val_weeks}")
            
        if len(self.test_df) > 0:
            test_2024_weeks = self.test_df[self.test_df['year']==2024]['week']
            test_range = f"2024 semanas {test_2024_weeks.min()}-{test_2024_weeks.max()}" if len(test_2024_weeks) > 0 else ""
            if 2025 in self.test_df['year'].values:
                test_range += " + 2025 completo"
            print(f"   Test range: {test_range}")
        
        def process_features(df, is_train=False):
            df_processed = df.copy()
            
            # Encoding de departamentos (top 10)
            top_departments = self.df.groupby('departament_id')['target_cases'].sum().nlargest(10).index
            df_processed['dept_encoded'] = df_processed['departament_id'].apply(
                lambda x: x if x in top_departments else 99
            )
            
            if is_train:
                self.onehot_encoder.fit(df_processed[['dept_encoded']])
            
            dept_encoded = self.onehot_encoder.transform(df_processed[['dept_encoded']])
            dept_df = pd.DataFrame(dept_encoded, columns=[f'dept_{i}' for i in range(dept_encoded.shape[1])], index=df_processed.index)
            
            # Features temporales
            df_processed['week_sin'] = np.sin(2 * np.pi * df_processed['week'] / 52)
            df_processed['week_cos'] = np.cos(2 * np.pi * df_processed['week'] / 52)
            
            # Features principales
            feature_cols = [
                'trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg',
                'densidad_pob', 'week_sin', 'week_cos',
                'cases_lag1', 'cases_lag2', 'cases_lag3', 'cases_lag4', 'trends_lag1',
                'cases_diff', 'cases_ma_3weeks', 'cases_max_4weeks',
                'extreme_rain', 'heat_wave', 'extreme_humidity', 'dengue_favorable'
            ]
            
            available_features = [f for f in feature_cols if f in df_processed.columns]
            X = pd.concat([df_processed[available_features], dept_df], axis=1)
            
            # Target con transformación logarítmica
            y_original = df_processed['target_cases'].copy()
            y = np.log1p(y_original) if self.use_log_transform else y_original
            
            return X, y, y_original
        
        # Procesar conjuntos
        self.X_train, self.y_train, self.y_train_original = process_features(self.train_df, is_train=True)
        self.X_val, self.y_val, self.y_val_original = process_features(self.val_df)
        self.X_test, self.y_test, self.y_test_original = process_features(self.test_df)
        
        # Eliminar NaN
        for X, y, y_orig in [(self.X_train, self.y_train, self.y_train_original), 
                            (self.X_val, self.y_val, self.y_val_original), 
                            (self.X_test, self.y_test, self.y_test_original)]:
            mask = ~X.isnull().any(axis=1)
            if X is self.X_train:
                self.X_train, self.y_train, self.y_train_original = X[mask], y[mask], y_orig[mask]
            elif X is self.X_val:
                self.X_val, self.y_val, self.y_val_original = X[mask], y[mask], y_orig[mask]
            else:
                self.X_test, self.y_test, self.y_test_original = X[mask], y[mask], y_orig[mask]
        
        # Escalado
        continuous_features = [f for f in ['trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg',
                                          'densidad_pob', 'cases_lag1', 'cases_lag2', 'cases_lag3', 'cases_lag4',
                                          'trends_lag1', 'cases_diff', 'cases_ma_3weeks', 'cases_max_4weeks'] 
                              if f in self.X_train.columns]
        
        self.scaler.fit(self.X_train[continuous_features])
        for X in [self.X_train, self.X_val, self.X_test]:
            X[continuous_features] = self.scaler.transform(X[continuous_features])
        
        self.feature_names = self.X_train.columns.tolist()
        print(f"Features preparadas: {len(self.feature_names)}")
        
        if self.use_log_transform:
            print(f"✅ Transformación LOG aplicada - Asimetría reducida: {log_analysis['improvement']:.2f}")
        
        # Mostrar resumen de features
        self.show_features_summary()
    
    def show_features_summary(self):
        """Mostrar resumen detallado de todas las features del modelo"""
        print("\n🔍 RESUMEN DE VARIABLES DEL MODELO")
        print("="*50)
        
        # Categorizar features
        feature_categories = {
            'Climáticas': [],
            'Demográficas': [],
            'Temporales': [],
            'Autoregresivas (Lags)': [],
            'Tendencias': [],
            'Eventos Extremos': [],
            'Departamentos': []
        }
        
        for feature in self.feature_names:
            if any(climate in feature for climate in ['trends_dengue', 'prec_2weeks', 'temp_2weeks', 'humd_2weeks']):
                feature_categories['Climáticas'].append(feature)
            elif 'densidad_pob' in feature:
                feature_categories['Demográficas'].append(feature)
            elif any(time in feature for time in ['week_sin', 'week_cos']):
                feature_categories['Temporales'].append(feature)
            elif 'lag' in feature:
                feature_categories['Autoregresivas (Lags)'].append(feature)
            elif any(trend in feature for trend in ['cases_diff', 'cases_ma', 'cases_max']):
                feature_categories['Tendencias'].append(feature)
            elif any(extreme in feature for extreme in ['extreme_', 'heat_wave', 'dengue_favorable']):
                feature_categories['Eventos Extremos'].append(feature)
            elif 'dept_' in feature:
                feature_categories['Departamentos'].append(feature)
        
        # Mostrar cada categoría
        for category, features in feature_categories.items():
            if features:
                print(f"\n📊 {category} ({len(features)} variables):")
                for feature in features:
                    if category == 'Departamentos':
                        if features.index(feature) == 0:
                            print(f"  • dept_0 a dept_{len(features)-1} (encoding de departamentos)")
                        break
                    else:
                        # Descripción de cada feature
                        description = self.get_feature_description(feature)
                        print(f"  • {feature}: {description}")
        
        print(f"\n📈 TOTAL: {len(self.feature_names)} variables")
        print(f"   - Variables continuas escaladas: {len([f for f in ['trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg', 'densidad_pob', 'cases_lag1', 'cases_lag2', 'cases_lag3', 'cases_lag4', 'trends_lag1', 'cases_diff', 'cases_ma_3weeks', 'cases_max_4weeks'] if f in self.feature_names])}")
        print(f"   - Variables binarias: {len([f for f in self.feature_names if any(x in f for x in ['extreme_', 'heat_wave', 'dengue_favorable'])])}")
        print(f"   - Variables departamento: {len([f for f in self.feature_names if 'dept_' in f])}")
    
    def get_feature_description(self, feature):
        """Obtener descripción de cada feature"""
        descriptions = {
            'trends_dengue': 'Google Trends búsquedas dengue',
            'prec_2weeks': 'Precipitación acumulada 2 semanas',
            'temp_2weeks_avg': 'Temperatura promedio 2 semanas',
            'humd_2weeks_avg': 'Humedad promedio 2 semanas',
            'densidad_pob': 'Densidad poblacional (hab/km²)',
            'week_sin': 'Componente sinusoidal semana (estacionalidad)',
            'week_cos': 'Componente cosinusoidal semana (estacionalidad)',
            'cases_lag1': 'Casos dengue 1 semana atrás',
            'cases_lag2': 'Casos dengue 2 semanas atrás',
            'cases_lag3': 'Casos dengue 3 semanas atrás',
            'cases_lag4': 'Casos dengue 4 semanas atrás',
            'trends_lag1': 'Google Trends 1 semana atrás',
            'cases_diff': 'Diferencia casos semana actual vs anterior',
            'cases_ma_3weeks': 'Media móvil casos 3 semanas',
            'cases_max_4weeks': 'Máximo casos 4 semanas',
            'extreme_rain': 'Lluvia extrema (>P90)',
            'heat_wave': 'Ola de calor (>P90)',
            'extreme_humidity': 'Humedad extrema (>P90)',
            'dengue_favorable': 'Condiciones climáticas favorables dengue'
        }
        return descriptions.get(feature, 'Variable procesada')
    
    def create_sample_weights(self, y, aggressive=True):
        """Crear pesos para priorizar picos (opción agresiva para mejor captura)"""
        weights = np.ones(len(y))
        if self.use_log_transform:
            y_original = np.expm1(y)
        else:
            y_original = y
            
        if aggressive:
            # Pesos MÁS AGRESIVOS para mejor captura de picos
            weights[y_original <= 2] = 1.0
            weights[(y_original > 2) & (y_original <= 5)] = 3.0
            weights[(y_original > 5) & (y_original <= 10)] = 6.0
            weights[y_original > 10] = 10.0  # 10x para picos grandes
        else:
            # Pesos originales (menos agresivos)
            weights[y_original <= 2] = 1.0
            weights[(y_original > 2) & (y_original <= 5)] = 2.0
            weights[(y_original > 5) & (y_original <= 10)] = 4.0
            weights[y_original > 10] = 6.0
        
        return weights
    
    def temporal_cross_validation(self):
        """Validación cruzada temporal con TimeSeriesSplit"""
        from sklearn.model_selection import TimeSeriesSplit
        
        print("🔄 Aplicando validación cruzada temporal...")
        
        # Combinar train y val para CV temporal
        X_combined = pd.concat([self.X_train, self.X_val], axis=0)
        y_combined = pd.concat([self.y_train, self.y_val], axis=0)
        
        tscv = TimeSeriesSplit(n_splits=4)
        
        # Modelo optimizado para CV
        xgb_cv_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=2,  # Más conservador para CV
            learning_rate=0.02,
            n_estimators=600,
            subsample=0.6,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            n_jobs=-1
        )
        
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_combined)):
            X_train_fold = X_combined.iloc[train_idx]
            y_train_fold = y_combined.iloc[train_idx]
            X_val_fold = X_combined.iloc[val_idx]
            y_val_fold = y_combined.iloc[val_idx]
            
            # Crear pesos para este fold (agresivos)
            fold_weights = self.create_sample_weights(y_train_fold, aggressive=True)
            
            # Entrenar en fold
            xgb_cv_model.fit(X_train_fold, y_train_fold, sample_weight=fold_weights, verbose=False)
            
            # Predecir
            y_pred_fold = xgb_cv_model.predict(X_val_fold)
            
            # Convertir de log si es necesario
            if self.use_log_transform:
                y_val_true = np.expm1(y_val_fold)
                y_pred_fold = np.expm1(y_pred_fold)
            else:
                y_val_true = y_val_fold
                
            y_pred_fold = np.maximum(0, y_pred_fold)
            
            # Calcular MAE
            mae = mean_absolute_error(y_val_true, y_pred_fold)
            fold_scores.append(mae)
            
        cv_mean = np.mean(fold_scores)
        cv_std = np.std(fold_scores)
        
        print(f"   CV MAE: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Entrenar modelo final con todos los datos (pesos agresivos)
        all_weights = self.create_sample_weights(y_combined, aggressive=True)
        xgb_cv_model.fit(X_combined, y_combined, sample_weight=all_weights, verbose=False)
        
        # Guardar modelo CV
        self.models['xgboost_cv'] = xgb_cv_model
        self.cv_score = cv_mean
        
        return cv_mean

    def train_outbreak_classifier(self, threshold=10):
        """PASO 1: Entrenar clasificador binario de brotes"""
        print(f"🎯 PASO 1: Entrenando clasificador de brotes (umbral >{threshold} casos)...")
        
        # Crear etiquetas binarias para brotes
        if self.use_log_transform:
            y_train_binary = (np.expm1(self.y_train) > threshold).astype(int)
            y_val_binary = (self.y_val_original > threshold).astype(int)
        else:
            y_train_binary = (self.y_train > threshold).astype(int)
            y_val_binary = (self.y_val > threshold).astype(int)
        
        outbreak_rate = y_train_binary.mean()
        print(f"   Tasa de brotes en entrenamiento: {outbreak_rate:.1%}")
        
        # Clasificador optimizado para brotes
        classifier_params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 600,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'scale_pos_weight': 1/outbreak_rate if outbreak_rate > 0 else 1,  # Balance clases
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.outbreak_classifier = xgb.XGBClassifier(**classifier_params)
        self.outbreak_classifier.fit(self.X_train, y_train_binary, verbose=False)
        
        # Evaluar clasificador
        y_val_prob = self.outbreak_classifier.predict_proba(self.X_val)[:, 1]
        y_val_pred_class = (y_val_prob > 0.5).astype(int)
        
        if y_val_pred_class.sum() > 0:
            precision = (y_val_binary & y_val_pred_class).sum() / y_val_pred_class.sum()
        else:
            precision = 0
            
        if y_val_binary.sum() > 0:
            recall = (y_val_binary & y_val_pred_class).sum() / y_val_binary.sum()
        else:
            recall = 0
            
        print(f"   Precisión: {precision:.3f}, Recall: {recall:.3f}")
        return y_train_binary, y_val_binary
    
    def train_magnitude_regressor(self, y_train_binary, threshold=10):
        """PASO 2: Entrenar regresor especializado para magnitud de brotes"""
        print(f"🎯 PASO 2: Entrenando regresor de magnitud para brotes...")
        
        # Filtrar solo casos de brote para entrenamiento especializado
        outbreak_mask = y_train_binary == 1
        
        if outbreak_mask.sum() == 0:
            print("   ⚠️  No hay casos de brote suficientes para regresor especializado")
            return None
            
        X_outbreak = self.X_train[outbreak_mask]
        y_outbreak = self.y_train[outbreak_mask]
        
        print(f"   Casos de brote para entrenar: {outbreak_mask.sum()}")
        
        # Pesos especiales para casos de brote (más agresivos)
        outbreak_weights = self.create_sample_weights(y_outbreak, aggressive=True)
        
        # Regresor Tweedie especializado en brotes (mejor para conteos con picos)
        magnitude_params = {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': 1.2,  # Optimizado para conteos con varianza alta
            'max_depth': 4,  # Más profundo para capturar patrones de brotes
            'learning_rate': 0.03,
            'n_estimators': 800,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.1,  # Menos regularización para capturar picos
            'reg_lambda': 0.1,
            'max_delta_step': 2,  # Control de gradientes en picos
            'gamma': 0.1,  # min_split_loss para menos nodos en outliers
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.magnitude_regressor = xgb.XGBRegressor(**magnitude_params)
        self.magnitude_regressor.fit(X_outbreak, y_outbreak, sample_weight=outbreak_weights, verbose=False)
        
        print("   ✅ Regresor de magnitud entrenado")
        return True

    def train_models(self):
        """Entrenar modelos con enfoque de dos etapas + modelos tradicionales"""
        print("🚀 Entrenando modelos con estrategia de 2 etapas...")
        
        # === PRIMERO: MODELOS BASE TRADICIONALES ===
        sample_weights = self.create_sample_weights(self.y_train, aggressive=True)
        
        # XGBoost principal con hiperparámetros optimizados para picos
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.02,
            'n_estimators': 800,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'max_delta_step': 1,  # Control gradientes en picos
            'gamma': 0.05,  # Menos nodos en outliers
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.models['xgboost'] = xgb.XGBRegressor(**xgb_params)
        self.models['xgboost'].fit(self.X_train, self.y_train, sample_weight=sample_weights, verbose=False)
        
        # Random Forest
        rf_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'min_samples_split': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.models['random_forest'] = RandomForestRegressor(**rf_params)
        self.models['random_forest'].fit(self.X_train, self.y_train, sample_weight=sample_weights)
        
        # XGBoost con validación cruzada temporal
        cv_score = self.temporal_cross_validation()
        
        # === ESTRATEGIA DE DOS ETAPAS (después de tener modelo base) ===
        y_train_binary, y_val_binary = self.train_outbreak_classifier(threshold=10)
        magnitude_trained = self.train_magnitude_regressor(y_train_binary, threshold=10)
        
        if magnitude_trained:
            # Crear modelo híbrido de dos etapas (ahora que tenemos xgboost base)
            self.models['two_stage'] = self.create_two_stage_model()
        
        # Seleccionar mejor modelo
        self.select_best_model(cv_score)

    def create_two_stage_model(self):
        """Crear modelo híbrido de dos etapas"""
        class TwoStageModel:
            def __init__(self, classifier, regressor, base_model, threshold=10, use_log=True):
                self.classifier = classifier
                self.regressor = regressor
                self.base_model = base_model
                self.threshold = threshold
                self.use_log = use_log
                
            def predict(self, X):
                # PASO 1: Clasificar si es brote
                outbreak_prob = self.classifier.predict_proba(X)[:, 1]
                outbreak_pred = outbreak_prob > 0.5
                
                # PASO 2: Predecir magnitud según clasificación
                predictions = np.zeros(len(X))
                
                if outbreak_pred.sum() > 0:
                    # Para brotes: usar regresor especializado
                    outbreak_magnitude = self.regressor.predict(X[outbreak_pred])
                    predictions[outbreak_pred] = outbreak_magnitude
                
                if (~outbreak_pred).sum() > 0:
                    # Para no-brotes: usar modelo base
                    base_pred = self.base_model.predict(X[~outbreak_pred])
                    predictions[~outbreak_pred] = base_pred
                
                return predictions
        
        return TwoStageModel(
            self.outbreak_classifier, 
            self.magnitude_regressor, 
            self.models['xgboost'], 
            threshold=10, 
            use_log=self.use_log_transform
        )
    
    def select_best_model(self, cv_score):
        """Seleccionar mejor modelo entre todas las estrategias"""
        print("\n🏆 Evaluando todos los modelos...")
        
        best_mae = float('inf')
        for name, model in self.models.items():
            if name == 'xgboost_cv':
                mae = cv_score
            else:
                y_val_pred = model.predict(self.X_val)
                if self.use_log_transform:
                    y_val_true = self.y_val_original
                    if name != 'two_stage':  # Two-stage ya maneja la transformación
                        y_val_pred = np.maximum(0, np.expm1(y_val_pred))
                    else:
                        y_val_pred = np.maximum(0, y_val_pred)
                else:
                    y_val_true = self.y_val
                    y_val_pred = np.maximum(0, y_val_pred)
                    
                mae = mean_absolute_error(y_val_true, y_val_pred)
                
            print(f"   {name}: MAE = {mae:.3f}")
            
            if mae < best_mae:
                best_mae = mae
                self.best_model = model
                self.best_model_name = name
        
        print(f"✅ Mejor modelo: {self.best_model_name} (MAE: {best_mae:.3f})")
        
        # Mostrar detalles del modelo ganador
        if self.best_model_name == 'two_stage':
            print("   🎯 Modelo de dos etapas seleccionado - Especializado en picos!")
    
    def evaluate_model(self):
        """Evaluar modelo y calcular métricas"""
        print("📊 Evaluando modelo...")
        
        def calculate_metrics(y_true_original, y_pred_transformed, dataset_name):
            # Revertir transformación
            if self.use_log_transform:
                y_pred_final = np.maximum(0, np.expm1(y_pred_transformed))
            else:
                y_pred_final = np.maximum(0, y_pred_transformed)
                
            y_true_final = y_true_original
            
            # Métricas generales
            mae = mean_absolute_error(y_true_final, y_pred_final)
            rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
            r2 = r2_score(y_true_final, y_pred_final)
            
            # Métricas para picos
            peak_mask = y_true_final > 5
            if peak_mask.sum() > 0:
                peak_mae = mean_absolute_error(y_true_final[peak_mask], y_pred_final[peak_mask])
                underest_pct = np.mean((y_true_final[peak_mask] - y_pred_final[peak_mask]) / y_true_final[peak_mask]) * 100
            else:
                peak_mae = 0
                underest_pct = 0
            
            print(f"{dataset_name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
            if peak_mask.sum() > 0:
                print(f"  Picos >5: {peak_mask.sum()} casos, MAE: {peak_mae:.3f}, Subestimación: {underest_pct:.1f}%")
            
            return {
                'MAE': mae, 'RMSE': rmse, 'R2': r2,
                'peak_mae': peak_mae, 'peak_count': peak_mask.sum(),
                'underest_pct': underest_pct
            }
        
        # Evaluar en todos los conjuntos
        y_train_pred = self.best_model.predict(self.X_train)
        y_val_pred = self.best_model.predict(self.X_val)
        y_test_pred = self.best_model.predict(self.X_test)
        
        train_metrics = calculate_metrics(self.y_train_original, y_train_pred, "Train")
        val_metrics = calculate_metrics(self.y_val_original, y_val_pred, "Validación")
        test_metrics = calculate_metrics(self.y_test_original, y_test_pred, "Test")
        
        # Visualización
        self.plot_results(val_metrics, test_metrics, y_val_pred, y_test_pred)
        
        return {'train': train_metrics, 'validation': val_metrics, 'test': test_metrics}
    
    def plot_results(self, val_metrics, test_metrics, y_val_pred, y_test_pred):
        """Crear gráficos de resultados"""
        print("📈 Generando gráficos de evaluación...")
        
        # Revertir transformación para plotting
        if self.use_log_transform:
            y_val_pred_plot = np.maximum(0, np.expm1(y_val_pred))
            y_test_pred_plot = np.maximum(0, np.expm1(y_test_pred))
        else:
            y_val_pred_plot = np.maximum(0, y_val_pred)
            y_test_pred_plot = np.maximum(0, y_test_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Evaluación del Modelo: {self.best_model_name}', fontsize=16)
        
        # Scatter plots
        axes[0,0].scatter(self.y_val_original, y_val_pred_plot, alpha=0.6, color='blue')
        max_val = max(self.y_val_original.max(), y_val_pred_plot.max())
        axes[0,0].plot([0, max_val], [0, max_val], 'r--', lw=2)
        axes[0,0].set_xlabel('Casos Reales')
        axes[0,0].set_ylabel('Casos Predichos')
        axes[0,0].set_title(f'Validación (MAE: {val_metrics["MAE"]:.2f})')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].scatter(self.y_test_original, y_test_pred_plot, alpha=0.6, color='green')
        max_val = max(self.y_test_original.max(), y_test_pred_plot.max())
        axes[0,1].plot([0, max_val], [0, max_val], 'r--', lw=2)
        axes[0,1].set_xlabel('Casos Reales')
        axes[0,1].set_ylabel('Casos Predichos')
        axes[0,1].set_title(f'Test (MAE: {test_metrics["MAE"]:.2f})')
        axes[0,1].grid(True, alpha=0.3)
        
        # Series temporales
        val_df_clean = self.val_df[~self.val_df.isnull().any(axis=1)].copy()
        val_weekly = val_df_clean.groupby('week').agg({
            'target_cases': 'sum'
        }).reset_index()
        val_pred_weekly = val_df_clean.copy()
        val_pred_weekly['predicted'] = y_val_pred_plot
        val_pred_weekly = val_pred_weekly.groupby('week')['predicted'].sum().reset_index()
        
        axes[1,0].plot(val_weekly['week'], val_weekly['target_cases'], 'b-', label='Real', linewidth=2)
        axes[1,0].plot(val_pred_weekly['week'], val_pred_weekly['predicted'], 'r--', label='Predicho', linewidth=2)
        axes[1,0].set_title('Serie Temporal - Validación 2023')
        axes[1,0].set_xlabel('Semana')
        axes[1,0].set_ylabel('Casos Totales')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        test_df_clean = self.test_df[~self.test_df.isnull().any(axis=1)].copy()
        test_weekly = test_df_clean.groupby('week').agg({
            'target_cases': 'sum'
        }).reset_index()
        test_pred_weekly = test_df_clean.copy()
        test_pred_weekly['predicted'] = y_test_pred_plot
        test_pred_weekly = test_pred_weekly.groupby('week')['predicted'].sum().reset_index()
        
        axes[1,1].plot(test_weekly['week'], test_weekly['target_cases'], 'b-', label='Real', linewidth=2)
        axes[1,1].plot(test_pred_weekly['week'], test_pred_weekly['predicted'], 'r--', label='Predicho', linewidth=2)
        axes[1,1].set_title('Serie Temporal - Test 2024')
        axes[1,1].set_xlabel('Semana')
        axes[1,1].set_ylabel('Casos Totales')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("✅ Guardado: model_evaluation.png")
    
    def analyze_feature_importance(self):
        """Analizar importancia de variables"""
        print("🔍 Analizando importancia de features...")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 features más importantes:")
            for _, row in feature_importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
            # Gráfico
            plt.figure(figsize=(10, 6))
            top_features = feature_importance_df.head(15)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Importancia de Variables - {self.best_model_name}')
            plt.xlabel('Importancia')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✅ Guardado: feature_importance.png")
            
            return feature_importance_df
    
    def save_model(self, model_path='dengue_model_optimized.joblib'):
        """Guardar modelo entrenado incluyendo componentes de dos etapas"""
        print(f"💾 Guardando modelo...")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'onehot_encoder': self.onehot_encoder,
            'feature_names': self.feature_names,
            'use_log_transform': self.use_log_transform
        }
        
        # Guardar componentes adicionales si es modelo de dos etapas
        if hasattr(self, 'outbreak_classifier'):
            model_data['outbreak_classifier'] = self.outbreak_classifier
        if hasattr(self, 'magnitude_regressor'):
            model_data['magnitude_regressor'] = self.magnitude_regressor
        
        joblib.dump(model_data, model_path)
        print(f"✅ Modelo guardado: {model_path}")
        
        if self.best_model_name == 'two_stage':
            print("   🎯 Incluye clasificador de brotes + regresor de magnitud")

def main():
    """Función principal simplificada"""
    print("🦟 MODELO DE PREDICCIÓN DE DENGUE")
    print("🔬 Metodología tradicional con transformación logarítmica")
    print("=" * 50)
    
    predictor = DenguePredictor('data/dengue_unified_dataset_enhanced.csv')
    
    try:
        # Pipeline simplificado
        predictor.load_and_diagnose_data()
        predictor.visualize_data()
        predictor.prepare_features()
        predictor.train_models()
        metrics = predictor.evaluate_model()
        predictor.analyze_feature_importance()
        predictor.save_model('dengue_model_optimized.joblib')
        
        # Resumen final
        print("\n" + "="*50)
        print("📊 RESUMEN FINAL")
        print("="*50)
        test_metrics = metrics['test']
        print(f"MAE Test: {test_metrics['MAE']:.3f}")
        print(f"RMSE Test: {test_metrics['RMSE']:.3f}")
        print(f"R² Test: {test_metrics['R2']:.3f}")
        
        if test_metrics['peak_count'] > 0:
            print(f"Picos detectados: {test_metrics['peak_count']}")
            print(f"MAE en picos: {test_metrics['peak_mae']:.3f}")
            print(f"Subestimación: {test_metrics['underest_pct']:.1f}%")
        
        print("\n📁 Archivos generados:")
        print("  - exploratory_analysis.png")
        print("  - target_distribution_analysis.png")
        print("  - correlation_matrix.png")
        print("  - model_evaluation.png")
        print("  - feature_importance.png")
        print("  - dengue_model_optimized.joblib")
        
        # Mostrar estadísticas específicas del modelo ganador
        print(f"\n🔬 ANÁLISIS DEL MODELO GANADOR: {predictor.best_model_name}")
        print("="*30)
        
        if predictor.best_model_name == 'two_stage':
            print("🎯 ESTRATEGIA DE DOS ETAPAS IMPLEMENTADA:")
            print("   1. Clasificador de brotes (umbral >10 casos)")
            print("   2. Regresor Tweedie especializado en magnitud")
            print("   3. Pesos de muestra 10x más agresivos")
            print("   4. Hiperparámetros optimizados para picos")
        else:
            print("🔧 ESTRATEGIAS APLICADAS:")
            print("   1. Sample weights 10x más agresivos")
            print("   2. Hiperparámetros optimizados (max_delta_step=2, gamma=0.1)")
            print("   3. Validación cruzada temporal")
        
        print("\n✅ Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
