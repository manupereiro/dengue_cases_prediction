#!/usr/bin/env python3
"""
Script para hacer matching de datos de Google Trends con datos procesados de dengue
Actualiza la columna trends_dengue en los archivos de dengue procesados

Lógica:
- Para departamentos de provincia Buenos Aires → usar trends_dengue_buenos_aires_20xx.csv
- Para comunas de CABA → usar trends_dengue_caba_20xx.csv
- Hacer match por year y week
"""

import pandas as pd
import numpy as np
import os
import glob

def load_trends_data(year):
    """
    Carga los datos de Google Trends para CABA y Buenos Aires
    
    Returns:
    tuple: (df_caba_trends, df_bsas_trends)
    """
    try:
        # Cargar datos de CABA
        caba_file = f"data/dengue_trends/trends_dengue_caba_{year}.csv"
        if os.path.exists(caba_file):
            df_caba = pd.read_csv(caba_file)
            print(f"✓ Datos de CABA cargados: {len(df_caba)} registros")
        else:
            print(f"❌ No se encontró archivo: {caba_file}")
            df_caba = pd.DataFrame()
        
        # Cargar datos de Buenos Aires
        bsas_file = f"data/dengue_trends/trends_dengue_buenos_aires_{year}.csv"
        if os.path.exists(bsas_file):
            df_bsas = pd.read_csv(bsas_file)
            print(f"✓ Datos de Buenos Aires cargados: {len(df_bsas)} registros")
        else:
            print(f"❌ No se encontró archivo: {bsas_file}")
            df_bsas = pd.DataFrame()
        
        return df_caba, df_bsas
        
    except Exception as e:
        print(f"❌ Error cargando datos de trends: {e}")
        return pd.DataFrame(), pd.DataFrame()

def match_trends_to_dengue(df_dengue, df_caba_trends, df_bsas_trends, year):
    """
    Hace match de los datos de trends con los datos de dengue
    
    Parameters:
    - df_dengue: DataFrame con datos de dengue procesados
    - df_caba_trends: DataFrame con trends de CABA
    - df_bsas_trends: DataFrame con trends de Buenos Aires
    - year: año a procesar
    
    Returns:
    DataFrame actualizado con columna trends_dengue llena
    """
    print(f"\n=== HACIENDO MATCHING PARA {year} ===")
    
    # Hacer una copia del DataFrame original
    df_result = df_dengue.copy()
    
    # Inicializar la columna trends_dengue si no existe
    if 'trends_dengue' not in df_result.columns:
        df_result['trends_dengue'] = np.nan
    
    total_records = len(df_result)
    matched_caba = 0
    matched_bsas = 0
    no_match = 0
    
    print(f"📊 Procesando {total_records} registros...")
    
    # Iterar por cada registro
    for idx, row in df_result.iterrows():
        year_val = row['year']
        week_val = row['week']
        province = row['province']
        
        # Determinar qué dataset de trends usar
        if province == 'CABA':
            # Usar datos de CABA
            if not df_caba_trends.empty:
                match = df_caba_trends[
                    (df_caba_trends['year'] == year_val) & 
                    (df_caba_trends['week'] == week_val)
                ]
                if not match.empty:
                    df_result.at[idx, 'trends_dengue'] = match.iloc[0]['trends_dengue']
                    matched_caba += 1
                else:
                    no_match += 1
            else:
                no_match += 1
                
        elif province == 'Buenos Aires':
            # Usar datos de Buenos Aires
            if not df_bsas_trends.empty:
                match = df_bsas_trends[
                    (df_bsas_trends['year'] == year_val) & 
                    (df_bsas_trends['week'] == week_val)
                ]
                if not match.empty:
                    df_result.at[idx, 'trends_dengue'] = match.iloc[0]['trends_dengue']
                    matched_bsas += 1
                else:
                    no_match += 1
            else:
                no_match += 1
        else:
            # Provincia no reconocida
            no_match += 1
    
    print(f"\n📈 RESULTADOS DEL MATCHING:")
    print(f"   ✓ Matched CABA: {matched_caba}")
    print(f"   ✓ Matched Buenos Aires: {matched_bsas}")
    print(f"   ❌ Sin match: {no_match}")
    print(f"   📊 Total procesados: {matched_caba + matched_bsas + no_match}")
    
    # Verificar que no hay valores nulos en trends_dengue
    null_trends = df_result['trends_dengue'].isna().sum()
    if null_trends > 0:
        print(f"⚠️  Advertencia: {null_trends} registros sin datos de trends")
    else:
        print("✓ Todos los registros tienen datos de trends")
    
    return df_result

def process_year(year):
    """
    Procesa un año específico
    """
    print(f"\n{'='*60}")
    print(f"PROCESANDO AÑO {year}")
    print(f"{'='*60}")
    
    # 1. Cargar datos de trends
    df_caba_trends, df_bsas_trends = load_trends_data(year)
    
    if df_caba_trends.empty and df_bsas_trends.empty:
        print(f"❌ No hay datos de trends disponibles para {year}")
        return False
    
    # 2. Cargar datos de dengue procesados
    dengue_file = f"data/dengue_processed_{year}_updated.csv"
    if not os.path.exists(dengue_file):
        print(f"❌ No se encontró archivo: {dengue_file}")
        return False
    
    try:
        df_dengue = pd.read_csv(dengue_file)
        print(f"✓ Datos de dengue cargados: {len(df_dengue)} registros")
    except Exception as e:
        print(f"❌ Error cargando datos de dengue: {e}")
        return False
    
    # 3. Hacer matching
    df_result = match_trends_to_dengue(df_dengue, df_caba_trends, df_bsas_trends, year)
    
    # 4. Guardar resultado
    output_file = f"data/dengue_processed_{year}_with_trends.csv"
    try:
        df_result.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Archivo actualizado guardado: {output_file}")
        
        # Mostrar muestra de los datos actualizados
        print(f"\n📋 MUESTRA DE DATOS ACTUALIZADOS:")
        sample = df_result[['year', 'week', 'departament_name', 'province', 'trends_dengue', 'target_cases']].head(10)
        print(sample.to_string(index=False))
        
        # Estadísticas de trends_dengue
        if not df_result['trends_dengue'].isna().all():
            print(f"\n📊 ESTADÍSTICAS DE TRENDS_DENGUE:")
            print(f"   Mínimo: {df_result['trends_dengue'].min()}")
            print(f"   Máximo: {df_result['trends_dengue'].max()}")
            print(f"   Promedio: {df_result['trends_dengue'].mean():.2f}")
            print(f"   Registros con datos: {df_result['trends_dengue'].notna().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error guardando archivo: {e}")
        return False

def find_available_years():
    """
    Encuentra los años disponibles basándose en los archivos existentes
    """
    years = set()
    
    # Buscar archivos de dengue procesados
    dengue_files = glob.glob("data/dengue_processed_*_updated.csv")
    for file in dengue_files:
        try:
            filename = os.path.basename(file)
            # Extraer año del nombre del archivo
            parts = filename.split('_')
            if len(parts) >= 3:
                year_str = parts[2]
                year = int(year_str)
                years.add(year)
        except:
            pass
    
    # Buscar archivos de trends
    trends_files = glob.glob("data/dengue_trends/trends_dengue_*_*.csv")
    for file in trends_files:
        try:
            filename = os.path.basename(file)
            # Extraer año del nombre del archivo (último número)
            parts = filename.replace('.csv', '').split('_')
            year_str = parts[-1]
            year = int(year_str)
            years.add(year)
        except:
            pass
    
    return sorted(list(years))

def main():
    """
    Función principal
    """
    print("=== MATCHING DE DATOS GOOGLE TRENDS CON DENGUE ===")
    print("Este script actualiza la columna trends_dengue en los archivos procesados")
    
    # Encontrar años disponibles
    available_years = find_available_years()
    
    if not available_years:
        print("❌ No se encontraron archivos de datos para procesar")
        return
    
    print(f"\n📅 Años disponibles: {available_years}")
    
    # Procesar cada año disponible
    success_count = 0
    for year in available_years:
        try:
            if process_year(year):
                success_count += 1
        except Exception as e:
            print(f"❌ Error procesando año {year}: {e}")
    
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"✓ Años procesados exitosamente: {success_count}")
    print(f"❌ Años con errores: {len(available_years) - success_count}")
    
    if success_count > 0:
        print(f"\n🎉 Los archivos actualizados están en:")
        for year in available_years:
            output_file = f"data/dengue_processed_{year}_with_trends.csv"
            if os.path.exists(output_file):
                print(f"   - {output_file}")

if __name__ == "__main__":
    main() 