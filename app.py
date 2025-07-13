from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
from models.loan_predictor import LoanPredictor
from models.data_processor import DataProcessor
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)
app.secret_key = 'loan-prediction-pro-2024'
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Inicializar componentes
predictor = LoanPredictor()
data_processor = DataProcessor()

# Cargar modelo al iniciar
try:
    predictor.load_model()
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"⚠️ Error al cargar modelo: {e}")
    try:
        predictor.train_model(use_ensemble=True)
    except Exception as train_error:
        print(f"❌ Error entrenando modelo: {train_error}")

@app.route('/')
def index():
    """Página principal con estadísticas mejoradas"""
    try:
        stats = data_processor.get_dataset_stats()
        insights = data_processor.get_prediction_insights()
        return render_template('index.html', stats=stats, insights=insights)
    except Exception as e:
        print(f"Error en página principal: {e}")
        return render_template('index.html', stats=None, insights=None, error=str(e))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predicción individual mejorada"""
    if request.method == 'POST':
        try:
            # Obtener y validar datos del formulario
            form_data = {
                'ApplicantIncome': float(request.form['applicant_income']),
                'CoapplicantIncome': float(request.form['coapplicant_income']),
                'LoanAmount': float(request.form['loan_amount']),
                'Loan_Amount_Term': float(request.form['loan_term']),
                'Credit_History': int(request.form['credit_history']),
                'Married': request.form['married'],
                'Education': request.form['education'],
                'Property_Area': request.form['property_area']
            }
            
            # Validar datos
            validation_errors = _validate_form_data(form_data)
            if validation_errors:
                for error in validation_errors:
                    flash(error, 'error')
                return render_template('predict.html', form_data=form_data)
            
            # Realizar predicción
            prediction = predictor.predict_single(form_data)
            
            # Calcular información adicional
            additional_info = _calculate_additional_info(form_data)
            
            return render_template('predict.html', 
                                 prediction=prediction, 
                                 form_data=form_data,
                                 additional_info=additional_info)
                                 
        except Exception as e:
            flash(f'Error en la predicción: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

def _validate_form_data(data):
    """Validar datos del formulario"""
    errors = []
    
    if data['ApplicantIncome'] <= 0:
        errors.append('El ingreso del solicitante debe ser mayor a 0')
    
    if data['CoapplicantIncome'] < 0:
        errors.append('El ingreso del co-solicitante no puede ser negativo')
    
    if data['LoanAmount'] <= 0:
        errors.append('El monto del préstamo debe ser mayor a 0')
    
    if data['Loan_Amount_Term'] <= 0 or data['Loan_Amount_Term'] > 480:
        errors.append('El plazo del préstamo debe estar entre 1 y 480 meses')
    
    # Validar ratio deuda-ingreso
    debt_ratio = data['LoanAmount'] / data['ApplicantIncome']
    if debt_ratio > 10:  # Ratio muy alto, probablemente error
        errors.append('El ratio deuda-ingreso parece excesivo. Verifique los montos.')
    
    return errors

def _calculate_additional_info(data):
    """Calcular información adicional para mostrar"""
    total_income = data['ApplicantIncome'] + data['CoapplicantIncome']
    debt_ratio = data['LoanAmount'] / data['ApplicantIncome']
    monthly_payment = data['LoanAmount'] / data['Loan_Amount_Term']
    
    return {
        'total_household_income': total_income,
        'debt_to_income_ratio': debt_ratio,
        'estimated_monthly_payment': monthly_payment,
        'payment_to_income_ratio': monthly_payment / data['ApplicantIncome']
    }

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    """Predicción por lotes mejorada"""
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No se seleccionó archivo', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No se seleccionó archivo', 'error')
                return redirect(request.url)
            
            if file and file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Validar estructura del archivo
                try:
                    df_check = pd.read_csv(filepath)
                    required_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                                   'Loan_Amount_Term', 'Credit_History', 'Married', 
                                   'Education', 'Property_Area']
                    
                    missing_cols = [col for col in required_cols if col not in df_check.columns]
                    if missing_cols:
                        flash(f'Columnas faltantes: {", ".join(missing_cols)}', 'error')
                        return redirect(request.url)
                    
                except Exception as e:
                    flash(f'Error leyendo archivo: {str(e)}', 'error')
                    return redirect(request.url)
                
                # Procesar archivo
                results = predictor.predict_batch(filepath)
                
                # Guardar resultados
                results_path = os.path.join('data', 'batch_results.csv')
                results.to_csv(results_path, index=False)
                
                flash(f'Procesados {len(results)} registros exitosamente', 'success')
                return redirect(url_for('batch_results'))
            else:
                flash('Solo se permiten archivos CSV', 'error')
                
        except Exception as e:
            flash(f'Error procesando archivo: {str(e)}', 'error')
    
    return render_template('batch_predict.html')

@app.route('/batch_results')
def batch_results():
    """Mostrar resultados mejorados de predicción por lotes"""
    try:
        results_path = os.path.join('data', 'batch_results.csv')
        if os.path.exists(results_path):
            results = pd.read_csv(results_path)
            
            # Estadísticas de resultados
            approved = (results['Predicted_Loan_Status'] == 1).sum()
            rejected = (results['Predicted_Loan_Status'] == 0).sum()
            avg_confidence = results['Probability_Approved'].mean()
            
            # Crear gráfico mejorado
            fig = go.Figure()
            
            # Gráfico de barras con detalles
            fig.add_trace(go.Bar(
                x=['Aprobados', 'Rechazados'], 
                y=[approved, rejected],
                marker_color=['#10B981', '#EF4444'],
                text=[f'{approved}<br>({approved/len(results)*100:.1f}%)', 
                      f'{rejected}<br>({rejected/len(results)*100:.1f}%)'],
                textposition='auto',
                textfont=dict(size=14, color='white')
            ))
            
            fig.update_layout(
                title='Resultados de Predicción por Lotes',
                xaxis_title='Estado del Préstamo',
                yaxis_title='Cantidad',
                template='plotly_white',
                height=400
            )
            
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Análisis de confianza
            confidence_analysis = {
                'high_confidence': len(results[results['Probability_Approved'] >= 0.8]),
                'medium_confidence': len(results[(results['Probability_Approved'] >= 0.6) & 
                                                (results['Probability_Approved'] < 0.8)]),
                'low_confidence': len(results[results['Probability_Approved'] < 0.6])
            }
            
            return render_template('batch_results.html', 
                                 results=results.to_dict('records'),
                                 chart=chart_json,
                                 total=len(results),
                                 approved=approved,
                                 rejected=rejected,
                                 avg_confidence=avg_confidence,
                                 confidence_analysis=confidence_analysis)
        else:
            flash('No hay resultados disponibles', 'info')
            return redirect(url_for('batch_predict'))
            
    except Exception as e:
        flash(f'Error cargando resultados: {str(e)}', 'error')
        return redirect(url_for('batch_predict'))

@app.route('/analytics')
def analytics():
    """Panel de análisis mejorado"""
    try:
        # Obtener métricas del modelo
        metrics = predictor.get_model_metrics()
        
        # Crear gráficos
        charts = data_processor.create_analytics_charts()
        
        # Obtener importancia de características
        feature_importance = predictor.get_feature_importance()
        
        # Insights adicionales
        insights = data_processor.get_prediction_insights()
        
        return render_template('analytics.html', 
                             metrics=metrics, 
                             charts=charts,
                             feature_importance=feature_importance,
                             insights=insights)
    except Exception as e:
        return render_template('analytics.html', 
                             metrics=None, 
                             charts=None, 
                             error=str(e))

@app.route('/retrain_model')
def retrain_model():
    """Reentrenar el modelo"""
    try:
        predictor.train_model(use_ensemble=True)
        flash('Modelo reentrenado exitosamente', 'success')
    except Exception as e:
        flash(f'Error reentrenando modelo: {str(e)}', 'error')
    
    return redirect(url_for('analytics'))

@app.route('/download_results')
def download_results():
    """Descargar resultados de predicción"""
    try:
        results_path = os.path.join('data', 'batch_results.csv')
        if os.path.exists(results_path):
            return send_file(results_path, as_attachment=True, 
                           download_name='predicciones_prestamos.csv')
        else:
            flash('No hay resultados para descargar', 'error')
            return redirect(url_for('batch_predict'))
    except Exception as e:
        flash(f'Error descargando archivo: {str(e)}', 'error')
        return redirect(url_for('batch_predict'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint mejorado para predicciones"""
    try:
        data = request.get_json()
        
        # Validar datos requeridos
        required_fields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                          'Loan_Amount_Term', 'Credit_History', 'Married', 
                          'Education', 'Property_Area']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Campos faltantes: {", ".join(missing_fields)}'}), 400
        
        prediction = predictor.predict_single(data)
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model_info')
def api_model_info():
    """API endpoint para información del modelo"""
    try:
        info = {
            'model_type': type(predictor.model).__name__ if predictor.model else 'No cargado',
            'feature_count': len(predictor.feature_columns) if predictor.feature_columns else 0,
            'feature_importance': predictor.get_feature_importance(),
            'training_data_size': len(pd.read_csv('data/loan_train2_clean.csv')) if os.path.exists('data/loan_train2_clean.csv') else 0
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Crear carpetas si no existen
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Cargar datos iniciales
    data_processor.load_data()
    
    print("🚀 Iniciando LoanPredict Pro...")
    print("📊 Estadísticas del dataset:")
    stats = data_processor.get_dataset_stats()
    if stats:
        print(f"  - Registros de entrenamiento: {stats['total_records']}")
        print(f"  - Tasa de aprobación: {stats['approval_rate']:.1f}%")
        print(f"  - Registros de prueba: {stats['test_records']}")
    
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
