# LoanPredict Pro

Sistema inteligente de predicción de préstamos usando Machine Learning.

## Características

- 🔮 Predicción individual de préstamos
- 📊 Procesamiento por lotes (CSV)
- 📈 Panel de análisis con métricas del modelo
- 🎯 Interfaz web moderna y responsiva
- 🚀 API REST para integraciones

## Instalación

1. Clonar el repositorio
2. Crear entorno virtual:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Colocar archivos de datos en la carpeta `data/`
5. Ejecutar la aplicación:
   ```bash
   python app.py
   ```

## Uso

1. Accede a http://localhost:5000
2. Completa el formulario de predicción individual
3. O sube un archivo CSV para predicción por lotes
4. Revisa las métricas en el panel de análisis

## Estructura del Proyecto

```
loan-prediction-app/
├── data/                    # Datos y archivos subidos
├── models/                  # Modelos de ML y procesamiento
├── static/                  # Archivos estáticos (CSS, JS)
├── templates/               # Plantillas HTML
├── app.py                   # Aplicación principal
└── requirements.txt         # Dependencias
```

## Tecnologías

- **Backend**: Flask, Scikit-learn, Pandas
- **Frontend**: Bootstrap 5, Plotly.js, Font Awesome
- **ML**: Regresión Logística con RandomizedSearchCV
- **Base de datos**: CSV files (escalable a SQL)

## Autor

Desarrollado por Pierreyfff