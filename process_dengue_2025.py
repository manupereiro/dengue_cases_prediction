import pandas as pd
import numpy as np
import unicodedata
import re
import chardet

# Mapeo de IDs estandarizados para CABA y AMBA
CABA_COMUNAS = {
    "COMUNA 1": 1, "COMUNA 2": 2, "COMUNA 3": 3, "COMUNA 4": 4, "COMUNA 5": 5,
    "COMUNA 6": 6, "COMUNA 7": 7, "COMUNA 8": 8, "COMUNA 9": 9, "COMUNA 10": 10,
    "COMUNA 11": 11, "COMUNA 12": 12, "COMUNA 13": 13, "COMUNA 14": 14, "COMUNA 15": 15
}

AMBA_MUNICIPIOS = {
    "Almirante Brown": 16, "Avellaneda": 17, "Berazategui": 18, "Berisso": 19,
    "Brandsen": 20, "Campana": 21, "Cañuelas": 22, "Ensenada": 23, "Escobar": 24,
    "Esteban Echeverría": 25, "Exaltación de la Cruz": 26, "Ezeiza": 27,
    "Florencio Varela": 28, "General Las Heras": 29, "General Rodríguez": 30,
    "General San Martín": 31, "Hurlingham": 32, "Ituzaingó": 33, "José C. Paz": 34,
    "La Matanza": 35, "Lanús": 36, "La Plata": 37, "Lomas de Zamora": 38,
    "Luján": 39, "Marcos Paz": 40, "Malvinas Argentinas": 41, "Moreno": 42,
    "Merlo": 43, "Morón": 44, "Pilar": 45, "Presidente Perón": 46, "Quilmes": 47,
    "San Fernando": 48, "San Isidro": 49, "San Miguel": 50, "San Vicente": 51,
    "Tigre": 52, "Tres de Febrero": 53, "Vicente López": 54, "Zárate": 55
}

def detect_encoding(file_path):
    """
    Detecta automáticamente la codificación de un archivo.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Leer los primeros 10KB
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        print(f"🔍 Codificación detectada: {encoding} (confianza: {confidence:.2%})")
        
        # Si la confianza es baja, probar codificaciones comunes
        if confidence < 0.8:
            common_encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'cp1252']
            for enc in common_encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as test_file:
                        test_file.read(1000)
                        print(f"✅ Codificación alternativa probada exitosamente: {enc}")
                        return enc
                except UnicodeDecodeError:
                    continue
        
        return encoding

def read_csv_with_encoding_and_separator(file_path):
    """
    Lee un CSV detectando automáticamente la codificación correcta y usando punto y coma como separador.
    """
    encoding = detect_encoding(file_path)
    
    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=';')
        print(f"✅ Archivo leído exitosamente con codificación: {encoding} y separador: ';'")
        print(f"📋 {len(df.columns)} columnas detectadas: {list(df.columns)}")
        return df
    except UnicodeDecodeError:
        print(f"❌ Error con codificación {encoding}, probando alternativas...")
        
        # Probar codificaciones alternativas
        alternative_encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'cp1252', 'latin-1']
        
        for enc in alternative_encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc, sep=';')
                print(f"✅ Archivo leído exitosamente con codificación alternativa: {enc} y separador: ';'")
                print(f"📋 {len(df.columns)} columnas detectadas: {list(df.columns)}")
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError("No se pudo determinar la codificación correcta del archivo")

def normalize_text(text):
    """
    Normaliza texto removiendo acentos, convirtiendo a mayúsculas y limpiando espacios.
    """
    if pd.isna(text):
        return ""
    
    # Convertir a string y quitar espacios extra
    text = str(text).strip()
    
    # Remover acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Convertir a mayúsculas
    text = text.upper()
    
    return text

def expand_abbreviations(text):
    """
    Expande abreviaciones comunes en nombres de lugares.
    """
    # Diccionario de abreviaciones comunes
    abbreviations = {
        'GRAL.': 'GENERAL',
        'GRL.': 'GENERAL',
        'GRAL': 'GENERAL',
        'GRL': 'GENERAL',
        'SAN': 'SAN',
        'STA.': 'SANTA',
        'STA': 'SANTA',
        'PTE.': 'PRESIDENTE',
        'PDTE.': 'PRESIDENTE',
        'PDTE': 'PRESIDENTE',
        'PTE': 'PRESIDENTE',
        'DR.': 'DOCTOR',
        'DR': 'DOCTOR',
        'ALTE.': 'ALMIRANTE',
        'ALTE': 'ALMIRANTE',
        'ALMT': 'ALMIRANTE'
    }
    
    # Aplicar reemplazos
    for abbrev, full in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, text)
    
    return text

def find_best_match(target_name, candidates_dict):
    """
    Encuentra la mejor coincidencia en un diccionario usando normalización de texto.
    Retorna tanto el ID como el nombre correcto (con acentos).
    """
    target_normalized = normalize_text(expand_abbreviations(target_name))
    
    # Buscar coincidencia exacta primero
    for candidate_name, candidate_id in candidates_dict.items():
        candidate_normalized = normalize_text(candidate_name)
        if target_normalized == candidate_normalized:
            return candidate_id, candidate_name
    
    # Buscar coincidencias parciales (palabras clave)
    target_words = set(target_normalized.split())
    best_match_id = None
    best_match_name = None
    best_score = 0
    
    for candidate_name, candidate_id in candidates_dict.items():
        candidate_normalized = normalize_text(candidate_name)
        candidate_words = set(candidate_normalized.split())
        
        # Calcular score basado en palabras en común
        common_words = target_words.intersection(candidate_words)
        if len(common_words) > 0:
            # Score: palabras en común / total de palabras únicas
            score = len(common_words) / len(target_words.union(candidate_words))
            if score > best_score and score > 0.5:  # Umbral mínimo de similitud
                best_score = score
                best_match_id = candidate_id
                best_match_name = candidate_name
    
    return best_match_id, best_match_name

def get_department_mapping(department_name, province_name):
    """
    Obtiene el ID estandarizado y el nombre correcto basado en el nombre del departamento y provincia.
    Maneja variaciones, acentos y abreviaciones.
    """
    if pd.isna(department_name) or pd.isna(province_name):
        return None, None
    
    # Mapear CABA
    if province_name == "CABA":
        return find_best_match(department_name, CABA_COMUNAS)
    
    # Mapear Buenos Aires (AMBA)
    elif province_name == "Buenos Aires":
        return find_best_match(department_name, AMBA_MUNICIPIOS)
    
    # Si no es CABA ni Buenos Aires, retornar None
    return None, None

def process_dengue_data_2025(input_file, output_file=None):
    """
    Procesa los datos de dengue/zika de 2025 al formato requerido para el modelo.
    Incluye todos los eventos transmitidos por Aedes aegypti (dengue y zika).
    Maneja problemas de codificación automáticamente y el formato de columnas de 2025.
    
    Formato objetivo:
    | year | week | departament_name | departament_id | province | target_cases |
    
    Mapeo de columnas para 2025:
    - departamento_residencia → departament_name
    - provincia_residencia → province  
    - ANIO_MIN → year
    - SEPI_MIN → week
    - cantidad → target_cases
    """
    
    # Leer el archivo CSV con detección automática de codificación y separador punto y coma
    df = read_csv_with_encoding_and_separator(input_file)
    
    print(f"📊 PROCESANDO DATOS DE 2025")
    print(f"Datos originales: {len(df)} registros")
    
    # Verificar que existen las columnas esperadas
    required_columns = ['departamento_residencia', 'provincia_residencia', 'ANIO_MIN', 'SEPI_MIN', 'cantidad']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ Columnas faltantes: {missing_columns}")
        print("Columnas disponibles:", list(df.columns))
        
        # Intentar mapear columnas similares
        column_mapping = {}
        for required_col in required_columns:
            for actual_col in df.columns:
                if required_col.lower() in actual_col.lower() or actual_col.lower() in required_col.lower():
                    column_mapping[required_col] = actual_col
                    print(f"🔄 Mapeando '{required_col}' a '{actual_col}'")
                    break
        
        if len(column_mapping) < len(required_columns):
            return None
        
        # Renombrar columnas según el mapeo encontrado
        df = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Filtrar registros válidos (sin valores nulos en columnas importantes)
    df_clean = df.dropna(subset=['departamento_residencia', 'provincia_residencia', 'ANIO_MIN', 'SEPI_MIN', 'cantidad'])
    print(f"🧹 Datos después de limpiar valores nulos: {len(df_clean)} registros")
    
    # Verificar si hay información de eventos
    if 'evento_nombre' in df.columns:
        print(f"Eventos incluidos: {sorted(df['evento_nombre'].unique())}")
    
    # Agrupar por año, semana epidemiológica, departamento y provincia
    # Sumar la cantidad de casos de todos los grupos de edad y eventos
    grouped = df_clean.groupby([
        'ANIO_MIN',
        'SEPI_MIN', 
        'departamento_residencia',
        'provincia_residencia'
    ])['cantidad'].sum().reset_index()
    
    print(f"📊 Datos agrupados: {len(grouped)} registros únicos")
    
    # Renombrar columnas al formato objetivo
    result = grouped.rename(columns={
        'ANIO_MIN': 'year',
        'SEPI_MIN': 'week',
        'departamento_residencia': 'departament_name',
        'provincia_residencia': 'province',
        'cantidad': 'target_cases'
    })
    
    # Convertir columnas numéricas a enteros para evitar decimales innecesarios
    result['year'] = result['year'].astype(int)
    result['week'] = result['week'].astype(int)
    result['target_cases'] = result['target_cases'].astype(int)
    
    print(f"\n🔍 ANALIZANDO DEPARTAMENTOS ÚNICOS EN 2025:")
    dept_by_province = result.groupby('province')['departament_name'].unique()
    for province, depts in dept_by_province.items():
        print(f"\n{province}:")
        for dept in sorted(depts):
            print(f"  - {dept}")
    
    # Aplicar el mapeo de IDs estandarizados y nombres correctos
    mapping_results = result.apply(
        lambda row: get_department_mapping(row['departament_name'], row['province']), 
        axis=1
    )
    
    # Separar IDs y nombres correctos
    result['departament_id'] = [mapping[0] for mapping in mapping_results]
    result['departament_name_corrected'] = [mapping[1] for mapping in mapping_results]
    
    # Reemplazar el nombre del departamento con el nombre correcto (con acentos)
    # Solo para los casos que se pudieron mapear
    mask = result['departament_name_corrected'].notna()
    result.loc[mask, 'departament_name'] = result.loc[mask, 'departament_name_corrected']
    
    # Eliminar la columna auxiliar
    result = result.drop('departament_name_corrected', axis=1)
    
    # Identificar registros no mapeados para debugging
    unmapped = result[result['departament_id'].isna()]
    if not unmapped.empty:
        print("\n⚠️  NOMBRES NO MAPEADOS:")
        unmapped_summary = unmapped.groupby(['province', 'departament_name']).size().reset_index(name='count')
        for _, row in unmapped_summary.iterrows():
            print(f"  {row['province']}: '{row['departament_name']}' ({row['count']} registros)")
    
    # Filtrar solo CABA y Buenos Aires (AMBA) - excluir registros sin mapeo
    result = result[result['departament_id'].notna()].copy()
    result['departament_id'] = result['departament_id'].astype(int)
    
    # Agregar columnas vacías para las variables que se completarán después
    result['trends_dengue'] = np.nan
    result['prec_2weeks'] = np.nan
    result['temp_2weeks_avg'] = np.nan
    result['humd_2weeks_avg'] = np.nan
    
    # Reordenar columnas según el formato objetivo
    result = result[[
        'year', 'week', 'departament_name', 'departament_id', 
        'province', 'trends_dengue', 'prec_2weeks', 'temp_2weeks_avg', 
        'humd_2weeks_avg', 'target_cases'
    ]]
    
    # Ordenar por año, semana y departamento
    result = result.sort_values(['year', 'week', 'departament_name']).reset_index(drop=True)
    
    # Guardar resultado si se especifica archivo de salida
    if output_file:
        result.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✅ Datos procesados guardados en: {output_file} (UTF-8)")
    
    # Mostrar información del resultado
    print(f"\n📈 RESUMEN FINAL:")
    print(f"Datos agregados (CABA/AMBA únicamente): {len(result)} registros")
    print(f"Años incluidos: {sorted(result['year'].unique())}")
    print(f"Semanas epidemiológicas: {result['week'].min()} - {result['week'].max()}")
    print(f"Provincias: {sorted(result['province'].unique())}")
    print(f"Departamentos únicos: {result['departament_name'].nunique()}")
    print(f"Rango de IDs: {result['departament_id'].min()} - {result['departament_id'].max()}")
    
    return result

if __name__ == "__main__":
    # Procesar el archivo de 2025
    input_file = "data/dengue_not_processed/dengue-zika.2025-06-02.csv"
    output_file = "data/dengue_processed_2025.csv"
    
    # Nota: Puede ser necesario instalar chardet si no está disponible
    try:
        result_df = process_dengue_data_2025(input_file, output_file)
        
        if result_df is not None:
            # Mostrar una muestra de los resultados
            print("\n📋 PRIMERAS 10 FILAS DEL RESULTADO:")
            print(result_df.head(10)[['year', 'week', 'departament_name', 'departament_id', 'province', 'target_cases']])
            
            # Mostrar algunos departamentos específicos de CABA
            print("\n🏙️ EJEMPLOS DE DATOS DE CABA:")
            caba_data = result_df[result_df['province'] == 'CABA'].head(10)
            if not caba_data.empty:
                print(caba_data[['year', 'week', 'departament_name', 'departament_id', 'province', 'target_cases']])
            else:
                print("No se encontraron datos de CABA")
            
            # Mostrar algunos departamentos específicos de Buenos Aires (AMBA)
            print("\n🏘️ EJEMPLOS DE DATOS DE AMBA:")
            amba_data = result_df[result_df['province'] == 'Buenos Aires'].head(10)
            if not amba_data.empty:
                print(amba_data[['year', 'week', 'departament_name', 'departament_id', 'province', 'target_cases']])
            else:
                print("No se encontraron datos de AMBA")
            
            # Mostrar distribución temporal
            print(f"\n📅 DISTRIBUCIÓN TEMPORAL:")
            week_range = result_df.groupby('year')['week'].agg(['min', 'max'])
            for year, (min_week, max_week) in week_range.iterrows():
                print(f"  {year}: semanas {min_week} - {max_week}")
    except Exception as e:
        print(f"\n❌ Error procesando el archivo: {e}")
        print("Verifica que el archivo existe y tiene el formato esperado.") 