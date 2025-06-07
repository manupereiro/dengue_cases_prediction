#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pytrends.request import TrendReq
import time
from datetime import datetime
import random
import os

def get_google_trends_data(keywords, start_date, end_date, geo="AR", retries=3, delay=5):
    """
    Obtiene datos de Google Trends para términos específicos en un rango de fechas.
    
    Parámetros:
    - keywords: lista de términos de búsqueda (ej: ["dengue"])
    - start_date: fecha inicio en formato "YYYY-MM-DD" 
    - end_date: fecha fin en formato "YYYY-MM-DD"
    - geo: código geográfico (ej: "AR" para Argentina)
    - retries: número de reintentos en caso de error
    - delay: segundos a esperar entre reintentos
    
    Retorna:
    DataFrame con columnas: year, week, trends_<keyword>
    """
    
    for attempt in range(retries):
        try:
            # Pausa aleatoria para evitar rate limiting
            if attempt > 0:
                wait_time = delay + random.uniform(1, 5)
                print(f"Reintento {attempt + 1}/{retries}. Esperando {wait_time:.1f} segundos...")
                time.sleep(wait_time)
            
            # Inicializar pytrends con configuración SIMPLE para evitar problemas de compatibilidad
            print(f"Inicializando conexión a Google Trends...")
            pytrends = TrendReq(hl='es-AR', tz=360)
            
            # Construir timeframe
            timeframe = f"{start_date} {end_date}"
            print(f"Consultando Google Trends para '{keywords[0]}' en {geo} ({timeframe})")
            
            # Pausa antes de construir payload
            time.sleep(3)
            
            # Construir payload
            print("Construyendo consulta...")
            pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            
            # Pausa antes de la request principal
            print("Obteniendo datos...")
            time.sleep(2)
            
            # Obtener datos de interés a lo largo del tiempo
            interest_over_time_df = pytrends.interest_over_time()
            
            if interest_over_time_df.empty:
                print(f"No se encontraron datos para los términos: {keywords}")
                return pd.DataFrame()
            
            # Resetear index para trabajar con 'date' como columna
            df = interest_over_time_df.reset_index()
            
            # Extraer el primer keyword (término principal)
            main_keyword = keywords[0]
            
            # Verificar que la columna existe
            if main_keyword not in df.columns:
                print(f"Columna '{main_keyword}' no encontrada en los datos")
                print(f"Columnas disponibles: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Crear DataFrame resultado con las columnas requeridas
            result_df = pd.DataFrame()
            result_df['date'] = df['date']
            result_df['year'] = df['date'].dt.year
            result_df['week'] = df['date'].dt.isocalendar().week
            result_df[f'trends_{main_keyword}'] = df[main_keyword]
            
            # Eliminar la columna 'date' y quedarnos solo con year, week y trends
            final_df = result_df[['year', 'week', f'trends_{main_keyword}']].copy()
            
            print(f"✓ Datos obtenidos exitosamente para '{main_keyword}' desde {start_date} hasta {end_date}")
            print(f"Número de registros: {len(final_df)}")
            print(f"Rango de años: {final_df['year'].min()} - {final_df['year'].max()}")
            print(f"Rango de semanas: {final_df['week'].min()} - {final_df['week'].max()}")
            
            return final_df
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error en intento {attempt + 1}/{retries}: {error_msg}")
            
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < retries - 1:
                    wait_time = delay * (2 ** attempt) + random.uniform(5, 15)  # Backoff exponencial
                    print(f"Rate limit detectado. Esperando {wait_time:.1f} segundos antes del siguiente intento...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("⚠️  Demasiadas solicitudes. Recomendaciones:")
                    print("   1. Espera 10-15 minutos antes de volver a intentar")
                    print("   2. Usa períodos de tiempo más largos (ej: todo el año)")
                    print("   3. Reduce la frecuencia de consultas")
                    return pd.DataFrame()
            elif "method_whitelist" in error_msg or "Retry" in error_msg:
                print("❌ Error de compatibilidad de librerías detectado.")
                print("💡 Soluciones posibles:")
                print("   1. pip install --upgrade pytrends")
                print("   2. pip install 'urllib3<2.0' requests")
                print("   3. Crear un entorno virtual nuevo")
                return pd.DataFrame()
            else:
                print(f"Error diferente: {error_msg}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    return pd.DataFrame()
    
    return pd.DataFrame()

def get_dengue_trends_caba_bsas_20xx(year):
    """
    Función para obtener datos de dengue para CABA y Buenos Aires por separado durante 2020
    """
    print(f"=== OBTENIENDO DATOS DE DENGUE {year} - CABA Y BUENOS AIRES ===")
    print("⚠️  Esta consulta hará 2 requests separadas con pausas para evitar rate limiting...")
    
    # Parámetros para 2020
    keywords = ["dengue"]
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    results = {}
    
    # 1. Obtener datos para CABA (Ciudad Autónoma de Buenos Aires)
    print("\n📊 Obteniendo datos para CABA (AR-C)...")
    df_caba = get_google_trends_data(keywords, start_date, end_date, geo="AR-C")
    
    if not df_caba.empty:
        print("✓ Datos de CABA obtenidos exitosamente")
        df_caba['region'] = 'CABA'
        results['CABA'] = df_caba
        
        # Guardar CABA
        output_file_caba = f"data/dengue_trends/trends_dengue_caba_{year}.csv"
        df_caba.to_csv(output_file_caba, index=False, encoding='utf-8')
        print(f"✓ Datos de CABA guardados en: {output_file_caba}")
        
        # Mostrar estadísticas de CABA
        print(f"   - Total semanas CABA: {len(df_caba)}")
        print(f"   - Promedio CABA: {df_caba['trends_dengue'].mean():.2f}")
        print(f"   - Máximo CABA: {df_caba['trends_dengue'].max()}")
    else:
        print("❌ No se pudieron obtener datos de CABA")
    
    # PAUSA IMPORTANTE entre requests para evitar rate limiting
    print("\n⏳ Esperando 30 segundos antes de la siguiente consulta...")
    time.sleep(30)
    
    # 2. Obtener datos para Provincia de Buenos Aires
    print("\n📊 Obteniendo datos para Provincia de Buenos Aires (AR-B)...")
    df_bsas = get_google_trends_data(keywords, start_date, end_date, geo="AR-B")
    
    if not df_bsas.empty:
        print("✓ Datos de Provincia de Buenos Aires obtenidos exitosamente")
        df_bsas['region'] = 'Buenos Aires'
        results['Buenos Aires'] = df_bsas
        
        # Guardar Buenos Aires
        output_file_bsas = f"data/dengue_trends/trends_dengue_buenos_aires_{year}.csv"
        df_bsas.to_csv(output_file_bsas, index=False, encoding='utf-8')
        print(f"✓ Datos de Buenos Aires guardados en: {output_file_bsas}")
        
        # Mostrar estadísticas de Buenos Aires
        print(f"   - Total semanas Buenos Aires: {len(df_bsas)}")
        print(f"   - Promedio Buenos Aires: {df_bsas['trends_dengue'].mean():.2f}")
        print(f"   - Máximo Buenos Aires: {df_bsas['trends_dengue'].max()}")
    else:
        print("❌ No se pudieron obtener datos de Provincia de Buenos Aires")
    
    if 'CABA' in results:
        print("\n✓ Solo se obtuvieron datos de CABA")
        return results['CABA']
    
    elif 'Buenos Aires' in results:
        print("\n✓ Solo se obtuvieron datos de Buenos Aires")
        return results['Buenos Aires']
    
    else:
        print("\n❌ No se pudieron obtener datos de ninguna región")
        return pd.DataFrame()

if __name__ == "__main__":
    print("=== RECOLECTOR DE DATOS GOOGLE TRENDS - DENGUE ===")
    year = 2025
    
    get_dengue_trends_caba_bsas_20xx(year)
    
    print("\n" + "="*60)
    print("✅ Script completado. Revisa los archivos CSV generados.") 