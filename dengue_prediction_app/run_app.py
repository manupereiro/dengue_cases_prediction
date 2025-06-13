#!/usr/bin/env python3
"""
Script para ejecutar la aplicación de predicción de dengue
"""

import os
import sys
import subprocess

def main():
    print("🦟 Iniciando Aplicación de Predicción de Dengue")
    print("=" * 50)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("backend/app.py"):
        print("❌ Error: No se encuentra backend/app.py")
        print("   Asegúrate de ejecutar este script desde dengue_prediction_app/")
        return 1
    
    # Verificar que existe el modelo
    model_paths = [
        "dengue_model_optimized.joblib",
        "../dengue_model_optimized.joblib"
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Modelo encontrado: {path}")
            model_found = True
            break
    
    if not model_found:
        print("⚠️  Advertencia: No se encontró el modelo entrenado")
        print("   Buscar en:", model_paths)
        print("   La aplicación puede no funcionar correctamente")
    
    # Verificar dependencias
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import joblib
        print("✅ Dependencias principales verificadas")
    except ImportError as e:
        print(f"❌ Error: Falta dependencia: {e}")
        print("   Ejecuta: pip install -r requirements.txt")
        return 1
    
    print("\n🚀 Iniciando servidor Flask...")
    print("📍 URL: http://localhost:5000")
    print("⏹️  Para detener: Ctrl+C")
    print("-" * 50)
    
    # Cambiar al directorio backend y ejecutar la aplicación
    try:
        os.chdir("backend")
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Aplicación detenida por el usuario")
    except Exception as e:
        print(f"\n❌ Error ejecutando la aplicación: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 