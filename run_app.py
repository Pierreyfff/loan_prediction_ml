#!/usr/bin/env python3
"""
Script de inicio para LoanPredict Pro
"""

import os
import sys
from app import app

def check_dependencies():
    """Verificar dependencias necesarias"""
    required_dirs = [
        'data',
        'data/uploads', 
        'models',
        'static/css',
        'static/js',
        'templates'
    ]
    
    print("🔍 Verificando estructura de directorios...")
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"📁 Creando directorio: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"✅ {directory}")
    
    # Verificar archivos de datos
    data_files = [
        'data/loan_train2_clean.csv',
        'data/loan_test2_clean.csv', 
        'data/predicciones_loan.csv'
    ]
    
    print("\n📊 Verificando archivos de datos...")
    missing_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"⚠️ {file_path} (opcional)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n💡 Archivos faltantes: {len(missing_files)}")
        print("   El modelo se entrenará con datos sintéticos si es necesario.")
    
    return True

def main():
    """Función principal"""
    print("🚀 Iniciando LoanPredict Pro...")
    print("=" * 50)
    
    # Verificar dependencias
    if not check_dependencies():
        print("❌ Error en verificación de dependencias")
        sys.exit(1)
    
    print("\n✅ Verificación completada")
    print("🌐 Iniciando servidor web...")
    print("📱 Accede a: http://localhost:5000")
    print("🛑 Presiona Ctrl+C para detener")
    print("=" * 50)
    
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error al iniciar servidor: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()