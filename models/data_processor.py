import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json
import os

class DataProcessor:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.predictions_data = None
        
    def load_data(self):
        """Cargar datos de entrenamiento, prueba y predicciones"""
        try:
            if os.path.exists('data/loan_train2_clean.csv'):
                self.train_data = pd.read_csv('data/loan_train2_clean.csv')
                print(f"✅ Datos de entrenamiento cargados: {len(self.train_data)} registros")
                
            if os.path.exists('data/loan_test2_clean.csv'):
                self.test_data = pd.read_csv('data/loan_test2_clean.csv')
                print(f"✅ Datos de prueba cargados: {len(self.test_data)} registros")
                
            if os.path.exists('data/predicciones_loan.csv'):
                self.predictions_data = pd.read_csv('data/predicciones_loan.csv')
                print(f"✅ Predicciones cargadas: {len(self.predictions_data)} registros")
                
        except Exception as e:
            print(f"Error cargando datos: {e}")
    
    def get_dataset_stats(self):
        """Obtener estadísticas detalladas del dataset"""
        if self.train_data is None:
            self.load_data()
        
        if self.train_data is not None:
            # Estadísticas básicas
            total_train = len(self.train_data)
            approved_train = int((self.train_data['Loan_Status'] == 1).sum())
            rejected_train = int((self.train_data['Loan_Status'] == 0).sum())
            approval_rate = float((self.train_data['Loan_Status'] == 1).mean() * 100)
            
            # Estadísticas de prueba si están disponibles
            total_test = len(self.test_data) if self.test_data is not None else 0
            approved_test = int((self.test_data['Loan_Status'] == 1).sum()) if self.test_data is not None else 0
            
            # Estadísticas de predicciones si están disponibles
            prediction_stats = {}
            if self.predictions_data is not None:
                pred_approved = int((self.predictions_data['Predicted_Loan_Status'] == 1).sum())
                pred_rejected = int((self.predictions_data['Predicted_Loan_Status'] == 0).sum())
                pred_accuracy = self._calculate_accuracy() if self.test_data is not None else None
                
                prediction_stats = {
                    'predicted_approved': pred_approved,
                    'predicted_rejected': pred_rejected,
                    'prediction_accuracy': pred_accuracy,
                    'avg_confidence_approved': float(self.predictions_data[self.predictions_data['Predicted_Loan_Status'] == 1]['Prob_Yes (%)'].mean()),
                    'avg_confidence_rejected': float(self.predictions_data[self.predictions_data['Predicted_Loan_Status'] == 0]['Prob_No (%)'].mean())
                }
            
            # Análisis de variables numéricas (desnormalizadas aproximadamente)
            income_stats = self._analyze_income_patterns()
            
            stats = {
                'total_records': total_train,
                'approved_loans': approved_train,
                'rejected_loans': rejected_train,
                'approval_rate': approval_rate,
                'test_records': total_test,
                'test_approved': approved_test,
                'prediction_stats': prediction_stats,
                'income_analysis': income_stats,
                'feature_importance': self._get_feature_importance(),
                'data_quality': self._assess_data_quality()
            }
            return stats
        return None
    
    def _calculate_accuracy(self):
        """Calcular precisión del modelo comparando predicciones con valores reales"""
        if self.test_data is None or self.predictions_data is None:
            return None
        
        # Merge por Loan_ID para comparar
        merged = pd.merge(self.test_data[['Loan_ID', 'Loan_Status']], 
                         self.predictions_data[['Loan_ID', 'Predicted_Loan_Status']], 
                         on='Loan_ID')
        
        accuracy = (merged['Loan_Status'] == merged['Predicted_Loan_Status']).mean()
        return float(accuracy * 100)
    
    def _analyze_income_patterns(self):
        """Analizar patrones de ingresos"""
        if self.train_data is None:
            return {}
        
        # Las variables están normalizadas, pero podemos analizar distribuciones
        income_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        patterns = {}
        
        for col in income_cols:
            if col in self.train_data.columns:
                approved_mean = self.train_data[self.train_data['Loan_Status'] == 1][col].mean()
                rejected_mean = self.train_data[self.train_data['Loan_Status'] == 0][col].mean()
                
                patterns[col] = {
                    'approved_avg': float(approved_mean),
                    'rejected_avg': float(rejected_mean),
                    'difference': float(approved_mean - rejected_mean)
                }
        
        return patterns
    
    def _get_feature_importance(self):
        """Analizar importancia de características basada en correlaciones"""
        if self.train_data is None:
            return {}
        
        # Calcular correlaciones con Loan_Status
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        target_corr = self.train_data[numeric_cols].corrwith(self.train_data['Loan_Status']).abs()
        target_corr = target_corr.drop('Loan_Status', errors='ignore')
        
        # Top 10 características más importantes
        top_features = target_corr.nlargest(10)
        
        return {
            'top_features': top_features.to_dict(),
            'credit_history_impact': float(target_corr.get('Credit_History', 0)),
            'education_impact': float(target_corr.get('Education', 0)),
            'married_impact': float(target_corr.get('Married_Yes', 0))
        }
    
    def _assess_data_quality(self):
        """Evaluar calidad de los datos"""
        if self.train_data is None:
            return {}
        
        return {
            'missing_values': int(self.train_data.isnull().sum().sum()),
            'duplicate_rows': int(self.train_data.duplicated().sum()),
            'feature_count': len(self.train_data.columns) - 1,  # Excluir Loan_Status
            'balance_ratio': float(min(self.train_data['Loan_Status'].value_counts()) / max(self.train_data['Loan_Status'].value_counts()))
        }
    
    def create_analytics_charts(self):
        """Crear gráficos mejorados para análisis"""
        if self.train_data is None:
            self.load_data()
        
        if self.train_data is None:
            return None
        
        charts = {}
        
        # Gráfico 1: Distribución de aprobaciones (mejorado)
        approved = (self.train_data['Loan_Status'] == 1).sum()
        rejected = (self.train_data['Loan_Status'] == 0).sum()
        
        fig1 = go.Figure(data=[
            go.Pie(
                labels=['Aprobados', 'Rechazados'], 
                values=[approved, rejected],
                marker_colors=['#10B981', '#EF4444'],
                hole=0.4,
                textinfo='label+percent+value',
                textfont=dict(size=14)
            )
        ])
        fig1.update_layout(
            title={
                'text': 'Distribución de Aprobaciones de Préstamos',
                'x': 0.5,
                'font': {'size': 16}
            },
            showlegend=True,
            height=400
        )
        charts['approval_distribution'] = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Gráfico 2: Comparación de ingresos normalizados
        fig2 = go.Figure()
        
        # Boxplot para ingresos del solicitante
        approved_income = self.train_data[self.train_data['Loan_Status'] == 1]['ApplicantIncome']
        rejected_income = self.train_data[self.train_data['Loan_Status'] == 0]['ApplicantIncome']
        
        fig2.add_trace(go.Box(
            y=approved_income, 
            name='Aprobados', 
            marker_color='#10B981',
            boxpoints='outliers'
        ))
        fig2.add_trace(go.Box(
            y=rejected_income, 
            name='Rechazados', 
            marker_color='#EF4444',
            boxpoints='outliers'
        ))
        
        fig2.update_layout(
            title='Distribución de Ingresos del Solicitante (Normalizado)',
            yaxis_title='Ingreso Normalizado',
            height=400,
            template='plotly_white'
        )
        charts['income_distribution'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Gráfico 3: Impacto del historial crediticio
        credit_stats = self.train_data.groupby(['Credit_History', 'Loan_Status']).size().unstack(fill_value=0)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=['Sin Historial (0)', 'Con Historial (1)'], 
            y=credit_stats.loc[:, 0] if 0 in credit_stats.columns else [0, 0], 
            name='Rechazados',
            marker_color='#EF4444',
            text=credit_stats.loc[:, 0] if 0 in credit_stats.columns else [0, 0],
            textposition='auto'
        ))
        fig3.add_trace(go.Bar(
            x=['Sin Historial (0)', 'Con Historial (1)'], 
            y=credit_stats.loc[:, 1] if 1 in credit_stats.columns else [0, 0], 
            name='Aprobados',
            marker_color='#10B981',
            text=credit_stats.loc[:, 1] if 1 in credit_stats.columns else [0, 0],
            textposition='auto'
        ))
        
        fig3.update_layout(
            title='Impacto del Historial Crediticio en la Aprobación',
            xaxis_title='Historial Crediticio',
            yaxis_title='Cantidad de Préstamos',
            barmode='group',
            height=400,
            template='plotly_white'
        )
        charts['credit_history_impact'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Gráfico 4: Análisis de características categóricas
        fig4 = go.Figure()
        
        # Análisis de educación
        education_stats = self.train_data.groupby(['Education', 'Loan_Status']).size().unstack(fill_value=0)
        
        fig4.add_trace(go.Bar(
            x=['No Graduado (0)', 'Graduado (1)'],
            y=education_stats.loc[:, 0] if 0 in education_stats.columns else [0, 0],
            name='Rechazados',
            marker_color='#EF4444',
            offsetgroup=1
        ))
        fig4.add_trace(go.Bar(
            x=['No Graduado (0)', 'Graduado (1)'],
            y=education_stats.loc[:, 1] if 1 in education_stats.columns else [0, 0],
            name='Aprobados',
            marker_color='#10B981',
            offsetgroup=1
        ))
        
        fig4.update_layout(
            title='Impacto del Nivel Educativo en la Aprobación',
            xaxis_title='Nivel de Educación',
            yaxis_title='Cantidad de Préstamos',
            barmode='group',
            height=400,
            template='plotly_white'
        )
        charts['education_impact'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Gráfico 5: Análisis de predicciones vs realidad (si hay predicciones)
        if self.predictions_data is not None and self.test_data is not None:
            charts['prediction_analysis'] = self._create_prediction_analysis_chart()
        
        return charts
    
    def _create_prediction_analysis_chart(self):
        """Crear gráfico de análisis de predicciones"""
        try:
            # Merge datos de prueba con predicciones
            merged = pd.merge(
                self.test_data[['Loan_ID', 'Loan_Status']], 
                self.predictions_data[['Loan_ID', 'Predicted_Loan_Status', 'Prob_Yes (%)']], 
                on='Loan_ID'
            )
            
            # Crear matriz de confusión visual
            confusion_data = pd.crosstab(
                merged['Loan_Status'], 
                merged['Predicted_Loan_Status'], 
                margins=True
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=confusion_data.values[:-1, :-1],  # Excluir totales
                x=['Pred: Rechazado', 'Pred: Aprobado'],
                y=['Real: Rechazado', 'Real: Aprobado'],
                colorscale='RdYlGn',
                text=confusion_data.values[:-1, :-1],
                texttemplate="%{text}",
                textfont={"size": 16},
                showscale=True
            ))
            
            fig.update_layout(
                title='Matriz de Confusión - Predicciones vs Realidad',
                xaxis_title='Predicciones del Modelo',
                yaxis_title='Valores Reales',
                height=400
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            print(f"Error creando gráfico de predicciones: {e}")
            return None
    
    def get_prediction_insights(self):
        """Obtener insights de las predicciones"""
        if self.predictions_data is None:
            return None
        
        insights = {
            'high_confidence_approved': len(self.predictions_data[
                (self.predictions_data['Predicted_Loan_Status'] == 1) & 
                (self.predictions_data['Prob_Yes (%)'] >= 80)
            ]),
            'low_confidence_predictions': len(self.predictions_data[
                ((self.predictions_data['Prob_Yes (%)'] >= 40) & 
                 (self.predictions_data['Prob_Yes (%)'] <= 60))
            ]),
            'avg_approval_confidence': float(
                self.predictions_data[self.predictions_data['Predicted_Loan_Status'] == 1]['Prob_Yes (%)'].mean()
            ),
            'avg_rejection_confidence': float(
                self.predictions_data[self.predictions_data['Predicted_Loan_Status'] == 0]['Prob_No (%)'].mean()
            )
        }
        
        return insights