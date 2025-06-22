#!/usr/bin/env python3
"""
Análisis de Importancia de Variables del Modelo de Dengue
========================================================

Este script analiza las variables más importantes del modelo entrenado
y genera visualizaciones para entender qué factores son más predictivos
para los casos de dengue.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class FeatureImportanceAnalyzer:
    """
    Analizador de importancia de variables para el modelo de dengue
    """
    
    def __init__(self, model_path='dengue_model_optimized.joblib'):
        """
        Inicializa el analizador
        
        Args:
            model_path (str): Ruta al modelo entrenado
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def load_model(self):
        """
        Carga el modelo entrenado y extrae la información necesaria
        """
        try:
            # Intentar cargar desde diferentes ubicaciones
            possible_paths = [
                self.model_path,
                f'dengue_prediction_app/{self.model_path}',
                f'dengue_prediction_app/backend/{self.model_path}'
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    print(f"Cargando modelo desde: {path}")
                    self.model_data = joblib.load(path)
                    break
            
            if self.model_data is None:
                raise FileNotFoundError(f"No se pudo encontrar el modelo en ninguna ubicación")
            
            # Extraer el modelo
            if isinstance(self.model_data, dict):
                # Buscar el modelo en diferentes claves posibles
                model_keys = ['best_model', 'model', 'trained_model']
                for key in model_keys:
                    if key in self.model_data:
                        self.model = self.model_data[key]
                        print(f"Modelo encontrado en clave: {key}")
                        break
                
                if self.model is None:
                    print("Claves disponibles en el archivo:", list(self.model_data.keys()))
                    # Tomar el primer valor que sea un modelo
                    for key, value in self.model_data.items():
                        if hasattr(value, 'feature_importances_'):
                            self.model = value
                            print(f"Usando modelo de la clave: {key}")
                            break
            else:
                self.model = self.model_data
            
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                raise ValueError("No se pudo extraer un modelo válido con feature_importances_")
            
            print(f"Modelo cargado exitosamente: {type(self.model).__name__}")
            print(f"Número de features: {len(self.model.feature_importances_)}")
            
            return True
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False
    
    def get_feature_names(self):
        """
        Obtiene los nombres de las variables del modelo desde el archivo guardado
        """
        # Intentar obtener los nombres reales del modelo guardado
        if isinstance(self.model_data, dict) and 'feature_names' in self.model_data:
            self.feature_names = self.model_data['feature_names']
            print(f"✅ Feature names obtenidos del modelo: {len(self.feature_names)}")
            return self.feature_names
        
        # Si no están guardados, usar nombres genéricos
        n_features = len(self.model.feature_importances_)
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
        print(f"⚠️ Usando nombres genéricos: {len(self.feature_names)}")
        return self.feature_names
    
    def calculate_importance(self):
        """
        Calcula la importancia de las variables
        """
        if self.model is None:
            print("Error: Modelo no cargado")
            return None
        
        if self.feature_names is None:
            self.get_feature_names()
        
        # Obtener importancias del modelo
        importances = self.model.feature_importances_
        
        # Crear DataFrame con importancias
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def plot_top_features(self, top_n=10, save_path='feature_importance_top10.png'):
        """
        Genera gráfico de las variables más importantes
        
        Args:
            top_n (int): Número de variables a mostrar
            save_path (str): Ruta donde guardar el gráfico
        """
        if self.feature_importance is None:
            self.calculate_importance()
        
        # Obtener top N features
        top_features = self.feature_importance.head(top_n).copy()
        
        # Crear nombres más legibles
        top_features['feature_clean'] = top_features['feature'].apply(self._clean_feature_name)
        
        # Crear el gráfico
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras horizontal
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        
        # Personalizar el gráfico
        plt.yticks(range(len(top_features)), top_features['feature_clean'])
        plt.xlabel('Importancia de la Variable')
        plt.title(f'Top {top_n} Variables Más Importantes del Modelo de Dengue', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Agregar valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # Invertir el orden del eje Y para mostrar la más importante arriba
        plt.gca().invert_yaxis()
        
        # Ajustar diseño
        plt.tight_layout()
        plt.grid(axis='x', alpha=0.3)
        
        # Guardar gráfico
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
        
        # Mostrar gráfico
        plt.show()
    
    def plot_importance_by_category(self, save_path='feature_importance_by_category.png'):
        """
        Genera gráfico agrupado por categorías de variables
        """
        if self.feature_importance is None:
            self.calculate_importance()
        
        # Categorizar variables
        def categorize_feature(feature_name):
            if 'cases_' in feature_name:
                return 'Casos Históricos'
            elif 'week_' in feature_name:
                return 'Variables Temporales'
            elif 'temp_' in feature_name or 'temperatura' in feature_name:
                return 'Temperatura'
            elif 'humd_' in feature_name or 'humedad' in feature_name:
                return 'Humedad'
            elif 'prec_' in feature_name or 'precipitacion' in feature_name:
                return 'Precipitación'
            elif 'trends_' in feature_name or 'google_trends' in feature_name:
                return 'Google Trends'
            elif 'dept_' in feature_name or 'departamento' in feature_name:
                return 'Ubicación Geográfica'
            else:
                return 'Otras'
        
        # Agregar categorías
        df_cat = self.feature_importance.copy()
        df_cat['category'] = df_cat['feature'].apply(categorize_feature)
        
        # Agrupar por categoría
        category_importance = df_cat.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
        category_importance = category_importance.sort_values('sum', ascending=False)
        
        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Gráfico 1: Importancia total por categoría
        bars1 = ax1.bar(range(len(category_importance)), category_importance['sum'],
                       color=plt.cm.Set3(np.linspace(0, 1, len(category_importance))))
        ax1.set_xticks(range(len(category_importance)))
        ax1.set_xticklabels(category_importance.index, rotation=45, ha='right')
        ax1.set_ylabel('Importancia Total')
        ax1.set_title('Importancia Total por Categoría de Variables')
        
        # Agregar valores en las barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Importancia promedio por categoría
        bars2 = ax2.bar(range(len(category_importance)), category_importance['mean'],
                       color=plt.cm.Set3(np.linspace(0, 1, len(category_importance))))
        ax2.set_xticks(range(len(category_importance)))
        ax2.set_xticklabels(category_importance.index, rotation=45, ha='right')
        ax2.set_ylabel('Importancia Promedio')
        ax2.set_title('Importancia Promedio por Categoría de Variables')
        
        # Agregar valores en las barras
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico por categorías guardado en: {save_path}")
        plt.show()
        
        return category_importance
    
    def _clean_feature_name(self, feature_name):
        """
        Convierte nombres de features a versiones más legibles
        """
        name_mapping = {
            # Variables de casos históricos (más importantes)
            'cases_ma_3weeks': 'Media Móvil 3 Semanas (Casos)',
            'cases_lag1': 'Casos Semana Anterior',
            'cases_lag2': 'Casos 2 Semanas Atrás',
            'cases_lag3': 'Casos 3 Semanas Atrás',
            'cases_lag4': 'Casos 4 Semanas Atrás',
            'cases_max_4weeks': 'Máximo Casos 4 Semanas',
            'cases_diff': 'Diferencia de Casos',
            
            # Variables temporales
            'week_cos': 'Semana del Año (Coseno)',
            'week_sin': 'Semana del Año (Seno)',
            
            # Variables climáticas
            'temp_2weeks_avg': 'Temperatura Promedio 2 Semanas',
            'humd_2weeks_avg': 'Humedad Promedio 2 Semanas',
            'prec_2weeks_sum': 'Precipitación Total 2 Semanas',
            
            # Google Trends
            'trends_dengue': 'Google Trends Dengue',
            'trends_lag1': 'Google Trends Lag 1',
            
            # Departamentos
            'dept_1': 'CABA - Comuna 1',
            'dept_2': 'CABA - Comuna 2', 
            'dept_3': 'CABA - Comuna 3',
            'dept_4': 'CABA - Comuna 4',
            'dept_5': 'CABA - Comuna 5',
            'dept_6': 'CABA - Comuna 6',
            'dept_7': 'CABA - Comuna 7',
            'dept_8': 'CABA - Comuna 8',
            'dept_9': 'CABA - Comuna 9',
            'dept_10': 'CABA - Comuna 10',
            'dept_11': 'CABA - Comuna 11',
            'dept_12': 'CABA - Comuna 12',
            'dept_13': 'CABA - Comuna 13',
            'dept_14': 'CABA - Comuna 14',
            'dept_15': 'CABA - Comuna 15',
        }
        
        if feature_name in name_mapping:
            return name_mapping[feature_name]
        elif feature_name.startswith('dept_'):
            dept_id = feature_name.split('_')[1]
            try:
                dept_num = int(dept_id)
                if dept_num <= 15:
                    return f'CABA - Comuna {dept_num}'
                else:
                    return f'Buenos Aires - Depto {dept_num}'
            except:
                return feature_name.replace('_', ' ').title()
        else:
            return feature_name.replace('_', ' ').title()
    
    def generate_report(self, save_path='feature_importance_report.txt'):
        """
        Genera un reporte detallado de la importancia de variables
        """
        if self.feature_importance is None:
            self.calculate_importance()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE IMPORTANCIA DE VARIABLES - MODELO DE DENGUE\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Modelo: {type(self.model).__name__}\n")
            f.write(f"Total de variables: {len(self.feature_importance)}\n")
            f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TOP 10 VARIABLES MÁS IMPORTANTES:\n")
            f.write("-" * 40 + "\n")
            
            for i, row in self.feature_importance.head(10).iterrows():
                f.write(f"{i+1:2d}. {self._clean_feature_name(row['feature']):30s} "
                       f"Importancia: {row['importance']:.4f}\n")
            
            f.write("\n\nTODAS LAS VARIABLES (ordenadas por importancia):\n")
            f.write("-" * 50 + "\n")
            
            for i, row in self.feature_importance.iterrows():
                f.write(f"{i+1:2d}. {row['feature']:35s} {row['importance']:.6f}\n")
            
            # Estadísticas adicionales
            f.write("\n\nESTADÍSTICAS DE IMPORTANCIA:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Importancia máxima: {self.feature_importance['importance'].max():.6f}\n")
            f.write(f"Importancia mínima: {self.feature_importance['importance'].min():.6f}\n")
            f.write(f"Importancia promedio: {self.feature_importance['importance'].mean():.6f}\n")
            f.write(f"Desviación estándar: {self.feature_importance['importance'].std():.6f}\n")
            
            # Top 10 representan qué porcentaje del total
            top10_sum = self.feature_importance.head(10)['importance'].sum()
            total_sum = self.feature_importance['importance'].sum()
            f.write(f"\nTop 10 variables representan: {(top10_sum/total_sum)*100:.1f}% de la importancia total\n")
        
        print(f"Reporte guardado en: {save_path}")
    
    def run_complete_analysis(self):
        """
        Ejecuta el análisis completo de importancia de variables
        """
        print("ANÁLISIS DE IMPORTANCIA DE VARIABLES - MODELO DE DENGUE")
        print("=" * 60)
        
        # Cargar modelo
        if not self.load_model():
            return False
        
        # Calcular importancias
        print("\nCalculando importancia de variables...")
        importance_df = self.calculate_importance()
        
        if importance_df is None:
            return False
        
        # Mostrar top 10
        print(f"\nTOP 10 VARIABLES MÁS IMPORTANTES:")
        print("-" * 50)
        for i, row in importance_df.head(10).iterrows():
            print(f"{i+1:2d}. {self._clean_feature_name(row['feature']):30s} "
                  f"Importancia: {row['importance']:.4f}")
        
        # Generar gráficos
        print("\nGenerando visualizaciones...")
        self.plot_top_features(top_n=10)
        self.plot_importance_by_category()
        
        # Generar reporte
        print("\nGenerando reporte detallado...")
        self.generate_report()
        
        print("\n✅ Análisis completado exitosamente!")
        print("\nArchivos generados:")
        print("- feature_importance_top10.png")
        print("- feature_importance_by_category.png")
        print("- feature_importance_report.txt")
        
        return True


def main():
    """
    Función principal
    """
    analyzer = FeatureImportanceAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if not success:
        print("\n❌ Error en el análisis. Verifica que el modelo esté disponible.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())