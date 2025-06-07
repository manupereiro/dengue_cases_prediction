#!/usr/bin/env python3
"""
Script para unificar todos los archivos dengue_processed_20xx_with_trends.csv
en un solo dataset consolidado para análisis y machine learning

El script:
1. Busca todos los archivos dengue_processed_20xx_with_trends.csv
2. Los combina en un solo DataFrame
3. Ordena por año, semana y departamento
4. Valida la consistencia de los datos
5. Guarda el resultado en un archivo unificado
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

def find_trends_files():
    """
    Encuentra todos los archivos dengue_processed_*_with_trends.csv
    """
    pattern = "data/dengue_processed_*_with_trends.csv"
    files = glob.glob(pattern)
    
    # Extraer años de los nombres de archivos
    files_with_years = []
    for file in files:
        try:
            filename = os.path.basename(file)
            # Extraer año: dengue_processed_YYYY_with_trends.csv
            parts = filename.split('_')
            year = int(parts[2])
            files_with_years.append((year, file))
        except:
            print(f"⚠️  No se pudo extraer año de: {file}")
    
    # Ordenar por año
    files_with_years.sort()
    
    return files_with_years

def load_and_validate_file(year, filepath):
    """
    Carga y valida un archivo individual
    """
    try:
        df = pd.read_csv(filepath)
        
        # Validaciones básicas
        required_columns = [
            'year', 'week', 'departament_name', 'departament_id', 
            'province', 'trends_dengue', 'prec_2weeks', 
            'temp_2weeks_avg', 'humd_2weeks_avg', 'target_cases'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️  {year}: Faltan columnas: {missing_cols}")
        
        # Verificar que el año coincide
        unique_years = df['year'].unique()
        if len(unique_years) != 1 or unique_years[0] != year:
            print(f"⚠️  {year}: Años inconsistentes en archivo: {unique_years}")
        
        # Estadísticas básicas
        total_records = len(df)
        unique_weeks = len(df['week'].unique())
        unique_depts = len(df['departament_name'].unique())
        
        # Datos de trends
        trends_filled = df['trends_dengue'].notna().sum()
        trends_missing = df['trends_dengue'].isna().sum()
        
        print(f"✓ {year}: {total_records} registros, {unique_weeks} semanas, {unique_depts} departamentos")
        print(f"    Trends: {trends_filled} completos, {trends_missing} faltantes")
        
        return df
        
    except Exception as e:
        print(f"❌ Error cargando {year}: {e}")
        return pd.DataFrame()

def analyze_combined_data(df_combined):
    """
    Analiza el dataset combinado y muestra estadísticas
    """
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DEL DATASET UNIFICADO")
    print(f"{'='*60}")
    
    # Estadísticas generales
    total_records = len(df_combined)
    years_range = f"{df_combined['year'].min()}-{df_combined['year'].max()}"
    unique_years = sorted(df_combined['year'].unique())
    unique_depts = len(df_combined['departament_name'].unique())
    unique_provinces = df_combined['province'].unique()
    
    print(f"📊 RESUMEN GENERAL:")
    print(f"   Total registros: {total_records:,}")
    print(f"   Años: {years_range} ({len(unique_years)} años)")
    print(f"   Departamentos únicos: {unique_depts}")
    print(f"   Provincias: {list(unique_provinces)}")
    
    # Distribución por año
    print(f"\n📅 DISTRIBUCIÓN POR AÑO:")
    year_counts = df_combined['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"   {year}: {count:,} registros")
    
    # Distribución por provincia
    print(f"\n🗺️  DISTRIBUCIÓN POR PROVINCIA:")
    province_counts = df_combined['province'].value_counts()
    for province, count in province_counts.items():
        print(f"   {province}: {count:,} registros")
    
    # Análisis de completitud de datos
    print(f"\n🔍 COMPLETITUD DE DATOS:")
    for col in ['trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg']:
        total = len(df_combined)
        complete = df_combined[col].notna().sum()
        percentage = (complete / total) * 100
        print(f"   {col}: {complete:,}/{total:,} ({percentage:.1f}%)")
    
    # Análisis de casos de dengue
    print(f"\n🦟 ANÁLISIS DE CASOS DE DENGUE:")
    total_cases = df_combined['target_cases'].sum()
    max_cases_week = df_combined['target_cases'].max()
    avg_cases_week = df_combined['target_cases'].mean()
    
    print(f"   Total casos: {total_cases:,}")
    print(f"   Máximo semanal: {max_cases_week}")
    print(f"   Promedio semanal: {avg_cases_week:.2f}")
    
    # Top departamentos con más casos
    dept_cases = df_combined.groupby('departament_name')['target_cases'].sum().sort_values(ascending=False)
    print(f"\n🏆 TOP 10 DEPARTAMENTOS CON MÁS CASOS:")
    for i, (dept, cases) in enumerate(dept_cases.head(10).items(), 1):
        print(f"   {i:2d}. {dept}: {cases:,} casos")
    
    # Análisis temporal
    print(f"\n📈 ANÁLISIS TEMPORAL:")
    weekly_cases = df_combined.groupby(['year', 'week'])['target_cases'].sum()
    peak_week = weekly_cases.idxmax()
    peak_cases = weekly_cases.max()
    print(f"   Pico máximo: Año {peak_week[0]}, Semana {peak_week[1]} ({peak_cases} casos)")
    
    # Análisis de Google Trends
    if not df_combined['trends_dengue'].isna().all():
        print(f"\n🔍 ANÁLISIS GOOGLE TRENDS:")
        trends_stats = df_combined['trends_dengue'].describe()
        print(f"   Mínimo: {trends_stats['min']:.0f}")
        print(f"   Máximo: {trends_stats['max']:.0f}")
        print(f"   Promedio: {trends_stats['mean']:.1f}")
        print(f"   Mediana: {trends_stats['50%']:.1f}")

def save_unified_dataset(df_combined, output_filename="data/dengue_unified_dataset.csv"):
    """
    Guarda el dataset unificado con validaciones finales
    """
    print(f"\n💾 GUARDANDO DATASET UNIFICADO...")
    
    try:
        # Ordenar datos
        df_sorted = df_combined.sort_values(['year', 'week', 'province', 'departament_name']).reset_index(drop=True)
        
        # Guardar archivo principal
        df_sorted.to_csv(output_filename, index=False, encoding='utf-8')
        file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
        print(f"✓ Dataset principal guardado: {output_filename}")
        print(f"   Tamaño: {file_size:.2f} MB")
        print(f"   Registros: {len(df_sorted):,}")
        
        # Crear versión resumida (sin duplicados por departamento-semana)
        summary_filename = output_filename.replace('.csv', '_summary.csv')
        df_summary = df_sorted.groupby(['year', 'week', 'province']).agg({
            'trends_dengue': 'first',  # Tomar el primer valor (debería ser igual para todos)
            'prec_2weeks': 'mean',
            'temp_2weeks_avg': 'mean', 
            'humd_2weeks_avg': 'mean',
            'target_cases': 'sum',  # Sumar casos por provincia-semana
            'departament_name': 'count'  # Contar departamentos
        }).rename(columns={'departament_name': 'num_departments'}).reset_index()
        
        df_summary.to_csv(summary_filename, index=False, encoding='utf-8')
        print(f"✓ Resumen por provincia-semana guardado: {summary_filename}")
        print(f"   Registros: {len(df_summary):,}")
        
        # Crear dataset solo para machine learning (sin NaN)
        ml_filename = output_filename.replace('.csv', '_ml_ready.csv')
        df_ml = df_sorted.dropna(subset=['trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg'])
        
        if len(df_ml) > 0:
            df_ml.to_csv(ml_filename, index=False, encoding='utf-8')
            print(f"✓ Dataset ML listo guardado: {ml_filename}")
            print(f"   Registros: {len(df_ml):,} (sin valores faltantes)")
        else:
            print(f"⚠️  No se pudo crear dataset ML (todos los registros tienen valores faltantes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error guardando dataset: {e}")
        return False

def main():
    """
    Función principal
    """
    print("=== UNIFICACIÓN DE DATASETS DE DENGUE ===")
    print("Combinando todos los archivos dengue_processed_*_with_trends.csv")
    
    # 1. Encontrar archivos
    files_with_years = find_trends_files()
    
    if not files_with_years:
        print("❌ No se encontraron archivos dengue_processed_*_with_trends.csv")
        return
    
    print(f"\n📁 Archivos encontrados: {len(files_with_years)}")
    for year, filepath in files_with_years:
        print(f"   {year}: {filepath}")
    
    # 2. Cargar y combinar archivos
    print(f"\n🔄 CARGANDO Y VALIDANDO ARCHIVOS...")
    
    dataframes = []
    successful_loads = 0
    
    for year, filepath in files_with_years:
        df = load_and_validate_file(year, filepath)
        if not df.empty:
            dataframes.append(df)
            successful_loads += 1
        else:
            print(f"❌ Fallo cargando {year}")
    
    if not dataframes:
        print("❌ No se pudo cargar ningún archivo exitosamente")
        return
    
    print(f"\n✓ Archivos cargados exitosamente: {successful_loads}/{len(files_with_years)}")
    
    # 3. Combinar todos los DataFrames
    print(f"\n🔗 COMBINANDO DATASETS...")
    df_combined = pd.concat(dataframes, ignore_index=True)
    
    # 4. Analizar datos combinados
    analyze_combined_data(df_combined)
    
    # 5. Guardar dataset unificado
    if save_unified_dataset(df_combined):
        print(f"\n🎉 UNIFICACIÓN COMPLETADA EXITOSAMENTE")
    else:
        print(f"\n❌ ERROR EN LA UNIFICACIÓN")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main() 