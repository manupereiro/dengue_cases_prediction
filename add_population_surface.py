#!/usr/bin/env python3
"""
Script para agregar superficie (sup_km2) y población (total_pobl) 
al dataset unificado de dengue
"""

import pandas as pd
import numpy as np

def add_population_surface_data():
    """Agregar datos de superficie y población al dataset de dengue"""
    
    print("=== AGREGANDO DATOS DE SUPERFICIE Y POBLACIÓN ===")
    
    # 1. Cargar el dataset principal de dengue
    print("1. Cargando dataset principal de dengue...")
    dengue_df = pd.read_csv('data/dengue_unified_dataset.csv')
    print(f"   Dataset cargado: {dengue_df.shape[0]} filas, {dengue_df.shape[1]} columnas")
    
    # 2. Cargar datos de CABA (Comunas 1-15)
    print("2. Cargando datos de CABA...")
    caba_df = pd.read_csv('caba_poblacion_superficie.csv')
    print(f"   CABA: {len(caba_df)} registros (departament_id {caba_df['department_id'].min()}-{caba_df['department_id'].max()})")
    
    # 3. Cargar datos de Buenos Aires (Municipios 16-55)
    print("3. Cargando datos de Buenos Aires...")
    bsas_df = pd.read_csv('buenos_aires_poblacion_superficie.csv')
    print(f"   Buenos Aires: {len(bsas_df)} registros (departament_id {bsas_df['department_id'].min()}-{bsas_df['department_id'].max()})")
    
    # 4. Combinar ambos datasets de población
    print("4. Combinando datos de población y superficie...")
    # Renombrar columnas para que coincidan
    caba_df = caba_df.rename(columns={'department_id': 'departament_id', 'department_name': 'departament_name'})
    bsas_df = bsas_df.rename(columns={'department_id': 'departament_id', 'department_name': 'departament_name'})
    
    # Combinar ambos datasets
    population_df = pd.concat([caba_df, bsas_df], ignore_index=True)
    print(f"   Dataset combinado: {len(population_df)} registros")
    
    # 5. Verificar datos antes del merge
    print("\n5. Verificando consistencia de datos...")
    dengue_depts = set(dengue_df['departament_id'].unique())
    population_depts = set(population_df['departament_id'].unique())
    
    missing_in_population = dengue_depts - population_depts
    missing_in_dengue = population_depts - dengue_depts
    
    if missing_in_population:
        print(f"   ⚠️  Departamentos en dengue pero no en población: {missing_in_population}")
    if missing_in_dengue:
        print(f"   ℹ️  Departamentos en población pero no en dengue: {missing_in_dengue}")
    
    print(f"   ✅ Departamentos en común: {len(dengue_depts & population_depts)}")
    
    # 6. Realizar el merge
    print("\n6. Realizando merge de datos...")
    initial_rows = len(dengue_df)
    
    # Hacer merge por departament_id
    dengue_enhanced = dengue_df.merge(
        population_df[['departament_id', 'sup_km2', 'total_pobl']], 
        on='departament_id', 
        how='left'
    )
    
    final_rows = len(dengue_enhanced)
    print(f"   Filas antes del merge: {initial_rows}")
    print(f"   Filas después del merge: {final_rows}")
    
    # 7. Verificar el resultado
    print("\n7. Verificando resultado...")
    missing_surface = dengue_enhanced['sup_km2'].isnull().sum()
    missing_population = dengue_enhanced['total_pobl'].isnull().sum()
    
    print(f"   Valores faltantes en sup_km2: {missing_surface}")
    print(f"   Valores faltantes en total_pobl: {missing_population}")
    
    if missing_surface > 0 or missing_population > 0:
        print("\n   Departamentos con datos faltantes:")
        missing_data = dengue_enhanced[
            dengue_enhanced['sup_km2'].isnull() | dengue_enhanced['total_pobl'].isnull()
        ][['departament_id', 'departament_name']].drop_duplicates()
        
        for _, row in missing_data.iterrows():
            print(f"     ID: {row['departament_id']}, Nombre: {row['departament_name']}")
    
    # 8. Mostrar estadísticas
    print(f"\n8. Estadísticas del dataset enriquecido:")
    print(f"   Columnas finales: {list(dengue_enhanced.columns)}")
    print(f"   Shape final: {dengue_enhanced.shape}")
    
    # Estadísticas descriptivas de las nuevas columnas
    if not dengue_enhanced['sup_km2'].isnull().all():
        print(f"\n   Superficie (km²):")
        print(f"     Mínima: {dengue_enhanced['sup_km2'].min():.1f} km²")
        print(f"     Máxima: {dengue_enhanced['sup_km2'].max():.1f} km²")
        print(f"     Promedio: {dengue_enhanced['sup_km2'].mean():.1f} km²")
    
    if not dengue_enhanced['total_pobl'].isnull().all():
        print(f"\n   Población:")
        print(f"     Mínima: {dengue_enhanced['total_pobl'].min():,.0f} habitantes")
        print(f"     Máxima: {dengue_enhanced['total_pobl'].max():,.0f} habitantes")
        print(f"     Promedio: {dengue_enhanced['total_pobl'].mean():,.0f} habitantes")
    
    # 9. Guardar el dataset enriquecido
    output_file = 'data/dengue_unified_dataset_enhanced.csv'
    dengue_enhanced.to_csv(output_file, index=False)
    print(f"\n9. ✅ Dataset enriquecido guardado en: {output_file}")
    
    # También actualizar el archivo original
    dengue_enhanced.to_csv('data/dengue_unified_dataset.csv', index=False)
    print(f"   ✅ Dataset original actualizado: data/dengue_unified_dataset.csv")
    
    # 10. Mostrar muestra del resultado
    print(f"\n10. Muestra del dataset enriquecido:")
    sample_cols = ['year', 'week', 'departament_name', 'departament_id', 'sup_km2', 'total_pobl', 'target_cases']
    print(dengue_enhanced[sample_cols].head(10).to_string(index=False))
    
    return dengue_enhanced

if __name__ == "__main__":
    enhanced_df = add_population_surface_data()
    print("\n🎉 Proceso completado exitosamente!") 