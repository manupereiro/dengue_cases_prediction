import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Mapeo de zonas a comunas/municipios según las especificaciones del usuario
ZONE_MAPPING = {
    1: {  # CABA (Comuna 1 - 15)
        "name": "CABA",
        "areas": [f"COMUNA {i}" for i in range(1, 16)],
        "province": "CABA"
    },
    2: {  # Lujan, General Rodriguez y Pilar
        "name": "Zona_2",
        "areas": ["Luján", "General Rodríguez", "Pilar"],
        "province": "Buenos Aires"
    },
    3: {  # Marcos Paz, Cañuelas y General las Heras
        "name": "Zona_3", 
        "areas": ["Marcos Paz", "Cañuelas", "General Las Heras"],
        "province": "Buenos Aires"
    },
    4: {  # Ezeiza, Esteban Echeverria, Presidente Peron y San vicente
        "name": "Zona_4",
        "areas": ["Ezeiza", "Esteban Echeverría", "Presidente Perón", "San Vicente"],
        "province": "Buenos Aires"
    },
    5: {  # Ensenada, Berisso, La Plata y Coronel Brandsen
        "name": "Zona_5",
        "areas": ["Ensenada", "Berisso", "La Plata", "Brandsen"],
        "province": "Buenos Aires"
    },
    6: {  # Almirante Brown, Florencio Varela, Berazategui y Quilmes
        "name": "Zona_6",
        "areas": ["Almirante Brown", "Florencio Varela", "Berazategui", "Quilmes"],
        "province": "Buenos Aires"
    },
    7: {  # Lanus, Avellaneda y Lomas de Zamora
        "name": "Zona_7",
        "areas": ["Lanús", "Avellaneda", "Lomas de Zamora", "LOMAS DE ZAMORA"],
        "province": "Buenos Aires"
    },
    8: {  # Merlo, La Matanza, Moron, Ituzaingo, Hurlingham, Tres de Febrero
        "name": "Zona_8",
        "areas": ["Merlo", "MERLO", "La Matanza", "Morón", "Ituzaingó", "Hurlingham", "Tres de Febrero"],
        "province": "Buenos Aires"
    },
    9: {  # Moreno, Jose C. Paz, Malvinas Argentinas, San Miguel
        "name": "Zona_9", 
        "areas": ["Moreno", "José C. Paz", "Malvinas Argentinas", "San Miguel"],
        "province": "Buenos Aires"
    },
    10: {  # Escobar, Tigre
        "name": "Zona_10",
        "areas": ["Escobar", "Tigre"],
        "province": "Buenos Aires"
    },
    11: {  # General San Martin, San Fernando, San isidro, Vicente Lopez
        "name": "Zona_11",
        "areas": ["General San Martín", "San Fernando", "San Isidro", "Vicente López"],
        "province": "Buenos Aires"
    },
    12: {  # Zarate, Exaltacion de la cruz, Campana
        "name": "Zona_12",
        "areas": ["Zárate", "Exaltación de la Cruz", "Campana"],
        "province": "Buenos Aires"
    }
}

def load_climate_data(zona_number):
    """
    Carga los datos climáticos para una zona específica
    """
    file_path = f"data/zones/zona{zona_number}.clima.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return None
    
    print(f"📁 Cargando datos de {file_path}")
    
    # Leer el archivo saltando el header (líneas 1-11, los datos empiezan en línea 12)
    df = pd.read_csv(file_path, skiprows=11, sep=',')
    
    # Convertir DOY (Day of Year) y YEAR a fecha
    df['date'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str).str.zfill(3), format='%Y%j')
    
    # Renombrar columnas para mayor claridad
    df = df.rename(columns={
        'T2M': 'temperature_c',
        'PRECTOTCORR': 'precipitation_mm', 
        'RH2M': 'humidity_percent'
    })
    
    # Agregar información de zona
    df['zona'] = zona_number
    
    return df[['date', 'zona', 'temperature_c', 'precipitation_mm', 'humidity_percent']]

def create_weekly_aggregates(df):
    """
    Crea agregados semanales de los datos climáticos diarios
    """
    # Agregar columnas de año, semana epidemiológica
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week
    
    # Crear agregados semanales
    weekly_agg = df.groupby(['year', 'week', 'zona']).agg({
        'temperature_c': 'mean',
        'precipitation_mm': 'sum', 
        'humidity_percent': 'mean'
    }).round(2)
    
    # Aplanar nombres de columnas
    weekly_agg.columns = ['temp_avg', 'precip_total', 'humidity_avg']
    weekly_agg = weekly_agg.reset_index()
    
    return weekly_agg

def create_lag_features(df):
    """
    Crea características de lag de 2 semanas (suma de t-1 y t-2 para precipitación, promedio para temp y humedad)
    Para semanas 1 y 2, usa los datos de la misma semana ya que no hay suficientes semanas anteriores
    """
    # Ordenar por zona, año y semana
    df = df.sort_values(['zona', 'year', 'week']).copy()
    
    lag_features = []
    
    # Para cada zona, calcular lag features
    for zona in df['zona'].unique():
        zona_df = df[df['zona'] == zona].copy().reset_index(drop=True)
        
        # Crear características de lag de 2 semanas
        # Para precipitación: suma de las 2 semanas anteriores (t-1 y t-2)
        zona_df['precip_lag1'] = zona_df['precip_total'].shift(1)
        zona_df['precip_lag2'] = zona_df['precip_total'].shift(2)
        zona_df['prec_2weeks'] = zona_df['precip_lag1'] + zona_df['precip_lag2']
        
        # Para temperatura: promedio de las 2 semanas anteriores
        zona_df['temp_lag1'] = zona_df['temp_avg'].shift(1)
        zona_df['temp_lag2'] = zona_df['temp_avg'].shift(2)
        zona_df['temp_2weeks_avg'] = (zona_df['temp_lag1'] + zona_df['temp_lag2']) / 2
        
        # Para humedad: promedio de las 2 semanas anteriores
        zona_df['humidity_lag1'] = zona_df['humidity_avg'].shift(1)
        zona_df['humidity_lag2'] = zona_df['humidity_avg'].shift(2)
        zona_df['humd_2weeks_avg'] = (zona_df['humidity_lag1'] + zona_df['humidity_lag2']) / 2
        
        # Manejo especial para semanas 1 y 2 (usar datos de la misma semana)
        for idx, row in zona_df.iterrows():
            week = row['week']
            
            if week == 1:
                # Para semana 1: usar datos de la semana 1
                zona_df.at[idx, 'prec_2weeks'] = row['precip_total']
                zona_df.at[idx, 'temp_2weeks_avg'] = row['temp_avg']
                zona_df.at[idx, 'humd_2weeks_avg'] = row['humidity_avg']
            elif week == 2:
                # Para semana 2: usar datos de la semana 2
                zona_df.at[idx, 'prec_2weeks'] = row['precip_total']
                zona_df.at[idx, 'temp_2weeks_avg'] = row['temp_avg'] 
                zona_df.at[idx, 'humd_2weeks_avg'] = row['humidity_avg']
        
        lag_features.append(zona_df)
    
    # Combinar todos los datos
    result_df = pd.concat(lag_features, ignore_index=True)
    result_df = result_df.sort_values(['zona', 'year', 'week']).reset_index(drop=True)
    
    return result_df

def find_area_zone(area_name):
    """
    Encuentra la zona correspondiente a un área/comuna/municipio específico
    """
    for zona_num, zona_info in ZONE_MAPPING.items():
        if area_name in zona_info['areas']:
            return zona_num
    return None

def update_dengue_file_with_climate(year):
    """
    Actualiza un archivo dengue_processed específico con datos climáticos
    """
    # Cargar el archivo dengue del año especificado
    dengue_file = f"data/dengue_processed_{year}.csv"
    
    if not os.path.exists(dengue_file):
        print(f"❌ Archivo no encontrado: {dengue_file}")
        return None
    
    print(f"📋 Procesando archivo: {dengue_file}")
    dengue_df = pd.read_csv(dengue_file)
    
    # Procesar todos los datos climáticos para el año
    all_climate_data = []
    
    for zona_num in range(1, 13):  # Zonas 1-12
        climate_df = load_climate_data(zona_num)
        if climate_df is not None:
            # Filtrar por año
            climate_df = climate_df[climate_df['date'].dt.year == year]
            
            if len(climate_df) > 0:
                # Crear agregados semanales
                weekly_climate = create_weekly_aggregates(climate_df)
                
                # Crear características de lag
                lag_climate = create_lag_features(weekly_climate)
                
                all_climate_data.append(lag_climate)
    
    if len(all_climate_data) == 0:
        print(f"❌ No se encontraron datos climáticos para el año {year}")
        return None
    
    # Combinar todos los datos climáticos
    combined_climate = pd.concat(all_climate_data, ignore_index=True)
    
    # Crear un mapeo de clima por semana y zona
    climate_mapping = {}
    for _, row in combined_climate.iterrows():
        key = (int(row['year']), int(row['week']), int(row['zona']))
        climate_mapping[key] = {
            'prec_2weeks': row['prec_2weeks'],
            'temp_2weeks_avg': row['temp_2weeks_avg'],
            'humd_2weeks_avg': row['humd_2weeks_avg']
        }
    
    # Agregar datos climáticos al dataframe de dengue
    print(f"🔄 Agregando datos climáticos a {len(dengue_df)} registros...")
    
    for idx, row in dengue_df.iterrows():
        area_name = row['departament_name']
        week = int(row['week'])
        year_val = int(row['year'])
        
        # Encontrar la zona correspondiente
        zona = find_area_zone(area_name)
        
        if zona is not None:
            # Buscar datos climáticos
            key = (year_val, week, zona)
            if key in climate_mapping:
                climate_data = climate_mapping[key]
                dengue_df.at[idx, 'prec_2weeks'] = round(climate_data['prec_2weeks'], 2) if not pd.isna(climate_data['prec_2weeks']) else ''
                dengue_df.at[idx, 'temp_2weeks_avg'] = round(climate_data['temp_2weeks_avg'], 2) if not pd.isna(climate_data['temp_2weeks_avg']) else ''
                dengue_df.at[idx, 'humd_2weeks_avg'] = round(climate_data['humd_2weeks_avg'], 2) if not pd.isna(climate_data['humd_2weeks_avg']) else ''
            else:
                # Si no hay datos climáticos, dejar vacío
                dengue_df.at[idx, 'prec_2weeks'] = ''
                dengue_df.at[idx, 'temp_2weeks_avg'] = ''
                dengue_df.at[idx, 'humd_2weeks_avg'] = ''
        else:
            print(f"⚠️  No se encontró zona para: {area_name}")
            dengue_df.at[idx, 'prec_2weeks'] = ''
            dengue_df.at[idx, 'temp_2weeks_avg'] = ''
            dengue_df.at[idx, 'humd_2weeks_avg'] = ''
    
    # Guardar el archivo actualizado
    output_file = f"data/dengue_processed_{year}_updated.csv"
    dengue_df.to_csv(output_file, index=False)
    
    print(f"✅ Archivo actualizado guardado en: {output_file}")
    print(f"📊 Registros procesados: {len(dengue_df)}")
    
    return dengue_df

def process_all_years():
    """
    Procesa todos los años disponibles
    """
    # Encontrar todos los archivos dengue_processed disponibles
    data_dir = "data"
    dengue_files = [f for f in os.listdir(data_dir) if f.startswith("dengue_processed_") and f.endswith(".csv")]
    
    years = []
    for file in dengue_files:
        try:
            year = int(file.split("_")[2].split(".")[0])
            years.append(year)
        except:
            continue
    
    years.sort()
    print(f"🎯 Años encontrados para procesar: {years}")
    
    processed_files = []
    
    for year in years:
        print(f"\n🔄 Procesando año {year}...")
        result = update_dengue_file_with_climate(year)
        if result is not None:
            processed_files.append(f"dengue_processed_{year}_updated.csv")
        else:
            print(f"❌ Error procesando año {year}")
    
    print(f"\n✅ Procesamiento completado!")
    print(f"📁 Archivos generados: {processed_files}")
    
    return processed_files

def test_processing_step_by_step(year=2019):
    """
    Función de prueba para verificar el procesamiento paso a paso
    """
    print("🧪 INICIANDO VERIFICACIÓN PASO A PASO")
    print("=" * 60)
    
    # Paso 1: Verificar mapeo de zonas
    print("📍 PASO 1: Verificando mapeo de zonas")
    test_areas = ["COMUNA 1", "COMUNA 4", "Avellaneda", "La Matanza", "LOMAS DE ZAMORA", "MERLO"]
    
    for area in test_areas:
        zona = find_area_zone(area)
        if zona:
            print(f"   ✅ {area} → Zona {zona} ({ZONE_MAPPING[zona]['name']})")
        else:
            print(f"   ❌ {area} → No encontrada")
    
    # Paso 2: Cargar y mostrar datos de una zona
    print(f"\n🌡️ PASO 2: Cargando datos climáticos de Zona 1 (CABA) para {year}")
    clima_df = load_climate_data(1)
    if clima_df is not None:
        year_data = clima_df[clima_df['date'].dt.year == year]
        print(f"   📊 Registros diarios para {year}: {len(year_data)}")
        print(f"   📅 Rango de fechas: {year_data['date'].min()} a {year_data['date'].max()}")
        print(f"   🌡️ Temperatura promedio: {year_data['temperature_c'].mean():.2f}°C")
        print(f"   🌧️ Precipitación total: {year_data['precipitation_mm'].sum():.2f}mm")
        print(f"   💧 Humedad promedio: {year_data['humidity_percent'].mean():.2f}%")
        
        # Mostrar algunos datos de ejemplo
        print(f"\n   📋 Primeros 5 registros de {year}:")
        print(year_data[['date', 'temperature_c', 'precipitation_mm', 'humidity_percent']].head().to_string(index=False))
    
    # Paso 3: Crear agregados semanales
    print(f"\n📅 PASO 3: Creando agregados semanales para Zona 1")
    if clima_df is not None:
        year_data = clima_df[clima_df['date'].dt.year == year]
        weekly_data = create_weekly_aggregates(year_data)
        print(f"   📊 Registros semanales: {len(weekly_data)}")
        print(f"   📅 Semanas: {weekly_data['week'].min()} a {weekly_data['week'].max()}")
        
        print(f"\n   📋 Primeras 5 semanas de {year}:")
        print(weekly_data[['year', 'week', 'temp_avg', 'precip_total', 'humidity_avg']].head().to_string(index=False))
    
    # Paso 4: Crear características de lag
    print(f"\n🔄 PASO 4: Creando características de lag")
    if clima_df is not None:
        year_data = clima_df[clima_df['date'].dt.year == year]
        weekly_data = create_weekly_aggregates(year_data)
        lag_data = create_lag_features(weekly_data)
        
        print(f"   📊 Registros con lag: {len(lag_data)}")
        
        # Mostrar específicamente semanas 1, 2 y 3 para verificar el manejo especial
        test_weeks = lag_data[lag_data['week'].isin([1, 2, 3, 4, 5])].copy()
        if len(test_weeks) > 0:
            print(f"\n   📋 Verificación de semanas 1-5 (manejo especial para 1 y 2):")
            print(test_weeks[['week', 'temp_avg', 'temp_2weeks_avg', 'precip_total', 'prec_2weeks', 'humidity_avg', 'humd_2weeks_avg']].to_string(index=False))
    
    # Paso 5: Probar con archivo dengue real
    print(f"\n📋 PASO 5: Probando con archivo dengue_processed_{year}.csv")
    dengue_file = f"data/dengue_processed_{year}.csv"
    
    if os.path.exists(dengue_file):
        dengue_df = pd.read_csv(dengue_file)
        print(f"   📊 Registros en archivo dengue: {len(dengue_df)}")
        print(f"   📅 Semanas únicas: {sorted(dengue_df['week'].unique())}")
        print(f"   🏘️ Áreas únicas: {dengue_df['departament_name'].nunique()}")
        
        # Mostrar algunas áreas de ejemplo
        sample_areas = dengue_df['departament_name'].unique()[:5]
        print(f"\n   📋 Ejemplo de áreas en el archivo:")
        for area in sample_areas:
            zona = find_area_zone(area)
            print(f"      {area} → Zona {zona if zona else 'NO ENCONTRADA'}")
        
        # Mostrar muestra del archivo original
        print(f"\n   📋 Muestra del archivo original:")
        print(dengue_df[['year', 'week', 'departament_name', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg', 'target_cases']].head().to_string(index=False))
        
    else:
        print(f"   ❌ Archivo {dengue_file} no encontrado")
    
    print("=" * 60)
    print("🧪 VERIFICACIÓN COMPLETADA")
    print("\n¿Los datos se ven correctos? Si es así, ejecuta process_all_years() para procesar todo.")

def quick_test_one_year(year=2019):
    """
    Prueba rápida procesando solo un año
    """
    print(f"🧪 PRUEBA RÁPIDA: Procesando solo año {year}")
    print("=" * 50)
    
    result = update_dengue_file_with_climate(year)
    
    if result is not None:
        # Mostrar estadísticas del resultado
        print(f"\n📊 RESULTADO DEL PROCESAMIENTO:")
        print(f"   📁 Archivo generado: data/dengue_processed_{year}_updated.csv")
        print(f"   📊 Total registros: {len(result)}")
        
        # Verificar que se agregaron los datos climáticos
        with_climate = result[(result['prec_2weeks'] != '') & (result['temp_2weeks_avg'] != '')].copy()
        print(f"   ✅ Registros con datos climáticos: {len(with_climate)}")
        print(f"   ❌ Registros sin datos climáticos: {len(result) - len(with_climate)}")
        
        if len(with_climate) > 0:
            print(f"\n   📋 Muestra de registros con datos climáticos:")
            sample = with_climate[['week', 'departament_name', 'prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg', 'target_cases']].head()
            print(sample.to_string(index=False))
        
        print(f"\n✅ Prueba completada exitosamente!")
        return True
    else:
        print(f"❌ Error en la prueba")
        return False

if __name__ == "__main__":
    print("🌡️  Iniciando procesamiento de datos climáticos...")
    print("=" * 60)
    
    # Procesar todos los años
    process_all_years()
    
    print("=" * 60)
    print("🎉 ¡Procesamiento completado!") 