#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pytrends.request import TrendReq
import time
from datetime import datetime
import random
import os

# =============================================================================
# CONFIGURACIÓN MANUAL DE AÑOS - FUNCIONALIDAD AÑADIDA
# =============================================================================

def get_dengue_trends_custom_year(year, region="AR"):
    """
    Función simple para obtener datos de un año específico
    
    Parámetros:
    - year: año a procesar (ej: 2019)
    - region: código de región ("AR" para Argentina, "AR-C" para CABA)
    """
    print(f"=== OBTENIENDO DATOS DE DENGUE {year} ===")
    print(f"Región: {region}")
    
    keywords = ["dengue"]
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    data = get_google_trends_data(keywords, start_date, end_date, geo=region, retries=3, delay=8)
    
    if not data.empty:
        # Guardar el archivo
        filename = f"trends_dengue_{region.lower()}_{year}.csv"
        data.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Datos guardados en: {filename}")
        
        # Mostrar estadísticas
        print(f"📊 Estadísticas:")
        print(f"   - Registros: {len(data)}")
        print(f"   - Semanas: {data['week'].min()}-{data['week'].max()}")
        print(f"   - Promedio: {data['trends_dengue'].mean():.2f}")
        print(f"   - Máximo: {data['trends_dengue'].max()}")
        
        return data
    else:
        print(f"❌ No se pudieron obtener datos para {year}")
        return pd.DataFrame()

# =============================================================================
# FUNCIONES PRINCIPALES DE RECOLECCIÓN (CÓDIGO ORIGINAL)
# =============================================================================

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

def get_available_regions(geo="AR"):
    """
    Obtiene las regiones disponibles para un país específico en Google Trends.
    
    Parámetros:
    - geo: código del país (ej: "AR" para Argentina)
    
    Retorna:
    DataFrame con regiones disponibles
    """
    try:
        print(f"Investigando regiones disponibles para {geo}...")
        time.sleep(3)  # Pausa inicial
        
        pytrends = TrendReq(hl='es-AR', tz=360, timeout=(10, 25))
        
        # Construir payload básico
        pytrends.build_payload(['dengue'], timeframe='2018-01-01 2018-12-31', geo=geo)
        
        time.sleep(2)  # Pausa antes de consulta
        
        # Intentar obtener datos por región
        try:
            regional_data = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True, inc_geo_code=True)
            print("✓ Regiones disponibles:")
            print(regional_data.head(10))
            return regional_data
        except:
            print("No se pudieron obtener datos regionales específicos")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error al obtener regiones disponibles: {str(e)}")
        return pd.DataFrame()

def test_google_trends_safe():
    """
    Función de prueba SEGURA para verificar el funcionamiento de Google Trends
    """
    print("=== PRUEBA SEGURA DE GOOGLE TRENDS ===")
    
    # Solo probar Argentina completo para evitar rate limiting
    keywords = ["dengue"]
    start_date = "2018-01-01"
    end_date = "2018-12-31"
    
    print(f"\nProbando con: {keywords}, {start_date} a {end_date}, geo='AR'")
    print("⚠️  Esta consulta puede tomar un momento...")
    
    df_ar = get_google_trends_data(keywords, start_date, end_date, geo="AR", retries=3, delay=10)
    
    if not df_ar.empty:
        print("\n--- Primeras 5 filas ---")
        print(df_ar.head())
        print("\n--- Últimas 5 filas ---")
        print(df_ar.tail())
        print(f"\n--- Estadísticas de trends_dengue ---")
        print(df_ar['trends_dengue'].describe())
        
        # Guardar resultados para uso posterior
        output_file = f"google_trends_dengue_{start_date}_{end_date}.csv"
        df_ar.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Datos guardados en: {output_file}")
    else:
        print("\n❌ No se pudieron obtener datos")

def get_dengue_trends_2018_simple():
    """
    Función SIMPLE para obtener datos de dengue en Argentina durante 2018
    """
    print("=== OBTENIENDO DATOS DE DENGUE 2018 (VERSIÓN SIMPLE) ===")
    print("⚠️  Esta consulta puede tomar varios minutos...")
    
    # Parámetros para 2018
    keywords = ["dengue"]
    start_date = "2018-01-01"
    end_date = "2018-12-31"
    
    # Intentar obtener datos con configuración simple
    print("📊 Obteniendo datos para Argentina completo...")
    df_argentina = get_google_trends_data(keywords, start_date, end_date, geo="AR")
    
    if not df_argentina.empty:
        print("✓ Datos de Argentina obtenidos exitosamente")
        
        # Mostrar muestra de datos
        print("\n--- Muestra de datos obtenidos ---")
        print(df_argentina.head(10))
        print(f"\n--- Estadísticas ---")
        print(f"Total semanas: {len(df_argentina)}")
        print(f"Valor mínimo: {df_argentina['trends_dengue'].min()}")
        print(f"Valor máximo: {df_argentina['trends_dengue'].max()}")
        print(f"Promedio: {df_argentina['trends_dengue'].mean():.2f}")
        
        # Guardar datos
        output_file = "trends_dengue_argentina_2018.csv"
        df_argentina.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ Datos guardados en: {output_file}")
        return df_argentina
    else:
        print("❌ No se pudieron obtener datos de Argentina")
        return pd.DataFrame()

def test_simple_connection():
    """
    Prueba simple de conexión a Google Trends
    """
    print("=== PRUEBA SIMPLE DE CONEXIÓN ===")
    
    try:
        print("1. Inicializando pytrends...")
        pytrends = TrendReq(hl='es-AR', tz=360)
        print("✓ Conexión inicializada")
        
        print("2. Probando consulta básica...")
        time.sleep(2)
        pytrends.build_payload(['dengue'], timeframe='2018-01-01 2018-03-31', geo='AR')
        print("✓ Consulta construida")
        
        print("3. Obteniendo datos...")
        time.sleep(2)
        data = pytrends.interest_over_time()
        print(f"✓ Datos obtenidos: {len(data)} registros")
        
        if not data.empty:
            print("📊 Muestra de datos:")
            print(data.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {str(e)}")
        return False

def get_dengue_trends_caba_bsas_20xx():
    """
    Función para obtener datos de dengue para CABA y Buenos Aires por separado durante 2020
    """
    print("=== OBTENIENDO DATOS DE DENGUE 2020 - CABA Y BUENOS AIRES ===")
    print("⚠️  Esta consulta hará 2 requests separadas con pausas para evitar rate limiting...")
    
    # Parámetros para 2020
    keywords = ["dengue"]
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    
    results = {}
    
    # 1. Obtener datos para CABA (Ciudad Autónoma de Buenos Aires)
    print("\n📊 Obteniendo datos para CABA (AR-C)...")
    df_caba = get_google_trends_data(keywords, start_date, end_date, geo="AR-C")
    
    if not df_caba.empty:
        print("✓ Datos de CABA obtenidos exitosamente")
        df_caba['region'] = 'CABA'
        results['CABA'] = df_caba
        
        # Guardar CABA
        output_file_caba = "data/dengue_trends/trends_dengue_caba_2020.csv"
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
        output_file_bsas = "data/dengue_trends/trends_dengue_buenos_aires_2020.csv"
        df_bsas.to_csv(output_file_bsas, index=False, encoding='utf-8')
        print(f"✓ Datos de Buenos Aires guardados en: {output_file_bsas}")
        
        # Mostrar estadísticas de Buenos Aires
        print(f"   - Total semanas Buenos Aires: {len(df_bsas)}")
        print(f"   - Promedio Buenos Aires: {df_bsas['trends_dengue'].mean():.2f}")
        print(f"   - Máximo Buenos Aires: {df_bsas['trends_dengue'].max()}")
    else:
        print("❌ No se pudieron obtener datos de Provincia de Buenos Aires")
    
    # 3. Combinar datos si ambos existen
    if 'CABA' in results and 'Buenos Aires' in results:
        print("\n📊 Combinando datos de ambas regiones...")
        
        df_combined = pd.concat([results['CABA'], results['Buenos Aires']], ignore_index=True)
        
        # Guardar archivo combinado
        output_file_combined = "data/dengue_trends/trends_dengue_caba_bsas_2020.csv"
        df_combined.to_csv(output_file_combined, index=False, encoding='utf-8')
        print(f"✓ Datos combinados guardados en: {output_file_combined}")
        
        # Mostrar comparación
        print("\n--- COMPARACIÓN CABA vs BUENOS AIRES ---")
        comparison = df_combined.groupby('region')['trends_dengue'].agg(['mean', 'max', 'min', 'std']).round(2)
        print(comparison)
        
        return df_combined
    
    elif 'CABA' in results:
        print("\n✓ Solo se obtuvieron datos de CABA")
        return results['CABA']
    elif 'Buenos Aires' in results:
        print("\n✓ Solo se obtuvieron datos de Buenos Aires")
        return results['Buenos Aires']
    else:
        print("\n❌ No se pudieron obtener datos de ninguna región")
        return pd.DataFrame()

def get_dengue_trends_by_region(regions_dict, year=2018):
    """
    Función general para obtener datos de múltiples regiones
    
    Parámetros:
    - regions_dict: diccionario con formato {'nombre_region': 'codigo_geo'}
    - year: año para obtener los datos
    
    Ejemplo:
    regions = {
        'CABA': 'AR-C',
        'Buenos Aires': 'AR-B',
        'Argentina': 'AR'
    }
    """
    print(f"=== OBTENIENDO DATOS DE DENGUE {year} POR REGIONES ===")
    
    keywords = ["dengue"]
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    results = {}
    total_regions = len(regions_dict)
    
    for i, (region_name, geo_code) in enumerate(regions_dict.items(), 1):
        print(f"\n📊 [{i}/{total_regions}] Obteniendo datos para {region_name} ({geo_code})...")
        
        df_region = get_google_trends_data(keywords, start_date, end_date, geo=geo_code)
        
        if not df_region.empty:
            print(f"✓ Datos de {region_name} obtenidos exitosamente")
            df_region['region'] = region_name
            df_region['geo_code'] = geo_code
            results[region_name] = df_region
            
            # Guardar archivo individual
            safe_name = region_name.replace(' ', '_').lower()
            output_file = f"trends_dengue_{safe_name}_{year}.csv"
            df_region.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✓ Guardado en: {output_file}")
            
            # Estadísticas
            print(f"   - Semanas: {len(df_region)}")
            print(f"   - Promedio: {df_region['trends_dengue'].mean():.2f}")
            print(f"   - Máximo: {df_region['trends_dengue'].max()}")
        else:
            print(f"❌ No se pudieron obtener datos de {region_name}")
        
        # Pausa entre regiones (excepto en la última)
        if i < total_regions:
            wait_time = 20 + random.uniform(5, 15)
            print(f"⏳ Esperando {wait_time:.1f} segundos antes de la siguiente consulta...")
            time.sleep(wait_time)
    
    # Combinar todos los resultados
    if results:
        print(f"\n📊 Combinando datos de {len(results)} regiones...")
        df_all = pd.concat(results.values(), ignore_index=True)
        
        # Guardar archivo combinado
        output_file_all = f"trends_dengue_all_regions_{year}.csv"
        df_all.to_csv(output_file_all, index=False, encoding='utf-8')
        print(f"✓ Datos combinados guardados en: {output_file_all}")
        
        # Mostrar comparación
        if len(results) > 1:
            print(f"\n--- COMPARACIÓN ENTRE REGIONES ---")
            comparison = df_all.groupby('region')['trends_dengue'].agg(['mean', 'max', 'min', 'std']).round(2)
            print(comparison)
        
        return df_all
    else:
        print("\n❌ No se pudieron obtener datos de ninguna región")
        return pd.DataFrame()

if __name__ == "__main__":
    print("=== RECOLECTOR DE DATOS GOOGLE TRENDS - DENGUE ===")
    print("Opciones disponibles:")
    print("1. Solo Argentina completo")
    print("2. CABA y Buenos Aires por separado")
    print("3. Todas las regiones (Argentina, CABA, Buenos Aires)")
    print("4. Prueba simple de conexión")
    print("5. 🆕 AÑO PERSONALIZADO - Especifica el año que quieres")
    
    # Para automatizar, puedes cambiar esto por el número de opción que quieras
    # Opción por defecto: año personalizado
    print("\n💡 NUEVO: Opción 5 para especificar año manualmente")
    opcion = input("Elige una opción (1-5): ").strip()
    
    try:
        opcion = int(opcion)
    except:
        opcion = 2  # Por defecto año personalizado
    
    if opcion == 1:
        print("\n>>> Opción 1: Argentina completo")
        if test_simple_connection():
            get_dengue_trends_2018_simple()
        else:
            print("❌ La prueba básica falló. Revisa la instalación de pytrends.")
    
    elif opcion == 2:
        print("\n>>> Opción 2: CABA y Buenos Aires por separado")
        get_dengue_trends_caba_bsas_20xx()
    
    elif opcion == 3:
        print("\n>>> Opción 3: Todas las regiones")
        regions = {
            'Argentina': 'AR',
            'CABA': 'AR-C', 
            'Buenos Aires': 'AR-B'
        }
        get_dengue_trends_by_region(regions, 2020)
    
    elif opcion == 4:
        print("\n>>> Opción 4: Prueba de conexión")
        test_simple_connection()
    
    elif opcion == 5:
        print("\n>>> Opción 5: Año personalizado")
        try:
            year = int(input("¿Qué año quieres recolectar? (ej: 2019): "))
            region = input("¿Qué región? (AR para Argentina, AR-C para CABA) [AR]: ").strip() or "AR"
            
            print(f"\n🎯 Recolectando datos de {year} para región {region}")
            get_dengue_trends_custom_year(year, region)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    else:
        print("❌ Opción no válida")
    
    print("\n" + "="*60)
    print("✅ Script completado. Revisa los archivos CSV generados.") 