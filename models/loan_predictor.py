import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_path = 'models/loan_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.feature_importance_ = None
        
        # Mapeo de características para la interfaz
        self.feature_mapping = {
            'ApplicantIncome': 'ApplicantIncome',
            'CoapplicantIncome': 'CoapplicantIncome', 
            'LoanAmount': 'LoanAmount',
            'Loan_Amount_Term': 'Loan_Amount_Term',
            'Credit_History': 'Credit_History',
            'Education': 'Education',
            'Married_Yes': 'Married',
            'Property_Area_Semiurban': 'Property_Area',
            'Property_Area_Urban': 'Property_Area',
            'DebtIncomeRatio': 'calculated',
            'Married_CH': 'calculated'
        }
    
    def prepare_features(self, data):
        """Preparar características para el modelo basado en los datos reales"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Crear características derivadas
        df['DebtIncomeRatio'] = df['LoanAmount'] / df['ApplicantIncome']
        df['DebtIncomeRatio'] = df['DebtIncomeRatio'].replace([np.inf, -np.inf], 0)
        
        # Crear Married_CH (Married * Credit_History)
        married_val = 1 if df['Married'].iloc[0] == 'Yes' else 0
        df['Married_CH'] = married_val * df['Credit_History'].iloc[0]
        
        # Codificar variables categóricas según el dataset
        df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
        df['Married_Yes'] = df['Married'].map({'Yes': 1, 'No': 0})
        
        # Codificar Property_Area (One-Hot Encoding)
        df['Property_Area_Semiurban'] = (df['Property_Area'] == 'Semiurban').astype(int)
        df['Property_Area_Urban'] = (df['Property_Area'] == 'Urban').astype(int)
        
        # Seleccionar características en el orden correcto
        feature_cols = [
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
            'Loan_Amount_Term', 'DebtIncomeRatio', 'Education',
            'Married_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban',
            'Credit_History', 'Married_CH'
        ]
        
        return df[feature_cols]
    
    def train_model(self, use_ensemble=True):
        """Entrenar modelo mejorado con ensemble methods"""
        try:
            # Cargar datos
            df_train = pd.read_csv('data/loan_train2_clean.csv')
            print(f"📊 Datos de entrenamiento cargados: {len(df_train)} registros")
            
            # Preparar datos
            X = df_train.drop(columns=['Loan_ID', 'Loan_Status'])
            y = df_train['Loan_Status']
            
            # Guardar nombres de características
            self.feature_columns = X.columns.tolist()
            
            # Dividir en conjunto de validación
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            if use_ensemble:
                # Entrenar ensemble de modelos
                self._train_ensemble_model(X_train, y_train, X_val, y_val)
            else:
                # Entrenar modelo logístico mejorado
                self._train_logistic_model(X_train, y_train, X_val, y_val)
            
            # Guardar modelo
            joblib.dump(self.model, self.model_path)
            if self.scaler:
                joblib.dump(self.scaler, self.scaler_path)
            
            # Evaluar en conjunto completo
            self._evaluate_model(X, y)
            
            print("✅ Modelo entrenado y guardado exitosamente")
            
        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            raise
    
    def _train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Entrenar modelo ensemble"""
        print("🔄 Entrenando modelo ensemble...")
        
        # Crear ensemble de modelos
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        # Entrenar y evaluar cada modelo
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            model_scores[name] = cv_scores.mean()
            
            # Entrenar modelo completo
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            print(f"  {name}: ROC-AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Seleccionar mejor modelo
        best_model_name = max(model_scores, key=model_scores.get)
        self.model = trained_models[best_model_name]
        
        print(f"🏆 Mejor modelo seleccionado: {best_model_name}")
        
        # Obtener importancia de características
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(self.feature_columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance_ = dict(zip(self.feature_columns, abs(self.model.coef_[0])))
    
    def _train_logistic_model(self, X_train, y_train, X_val, y_val):
        """Entrenar modelo logístico optimizado"""
        print("🔄 Entrenando modelo logístico optimizado...")
        
        # Estandarizar características numéricas si es necesario
        numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                           'Loan_Amount_Term', 'DebtIncomeRatio']
        
        # Espacio de búsqueda expandido
        param_distributions = {
            'C': uniform(0.001, 10),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'max_iter': randint(500, 2000),
            'class_weight': [None, 'balanced']
        }
        
        # Búsqueda aleatoria
        base_model = LogisticRegression(random_state=42)
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=100,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        
        print(f"✅ Mejores parámetros: {random_search.best_params_}")
        print(f"✅ Mejor ROC-AUC (CV): {random_search.best_score_:.4f}")
    
    def _evaluate_model(self, X, y):
        """Evaluar modelo en datos completos"""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        print("\n📈 Métricas del modelo en datos de entrenamiento:")
        print(f"  Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"  Precision: {precision_score(y, y_pred):.4f}")
        print(f"  Recall: {recall_score(y, y_pred):.4f}")
        print(f"  F1-Score: {f1_score(y, y_pred):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y, y_proba):.4f}")
    
    def load_model(self):
        """Cargar modelo preentrenado"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            print("✅ Modelo cargado desde archivo")
        else:
            print("⚠️ No se encontró modelo guardado. Entrenando nuevo modelo...")
            self.train_model()
    
    def predict_single(self, data):
        """Realizar predicción individual mejorada"""
        try:
            # Preparar características
            X_processed = self.prepare_features(data)
            
            # Realizar predicción
            prediction = self.model.predict(X_processed)[0]
            probability = self.model.predict_proba(X_processed)[0]
            
            # Calcular nivel de confianza mejorado
            confidence = max(probability)
            
            # Determinar nivel de riesgo
            risk_level = self._assess_risk_level(probability[1])
            
            # Obtener factores de decisión
            decision_factors = self._get_decision_factors(X_processed.iloc[0], prediction)
            
            return {
                'prediction': int(prediction),
                'probability_approved': float(probability[1]),
                'probability_rejected': float(probability[0]),
                'confidence': float(confidence),
                'status': 'Aprobado' if prediction == 1 else 'Rechazado',
                'risk_level': risk_level,
                'decision_factors': decision_factors
            }
            
        except Exception as e:
            raise Exception(f"Error en predicción: {str(e)}")
    
    def _assess_risk_level(self, prob_approved):
        """Evaluar nivel de riesgo basado en probabilidad"""
        if prob_approved >= 0.8:
            return {'level': 'Bajo', 'color': 'success', 'description': 'Excelente perfil crediticio'}
        elif prob_approved >= 0.6:
            return {'level': 'Medio', 'color': 'warning', 'description': 'Perfil crediticio aceptable'}
        elif prob_approved >= 0.4:
            return {'level': 'Alto', 'color': 'danger', 'description': 'Perfil crediticio riesgoso'}
        else:
            return {'level': 'Muy Alto', 'color': 'danger', 'description': 'Perfil crediticio muy riesgoso'}
    
    def _get_decision_factors(self, features, prediction):
        """Obtener factores que influenciaron la decisión"""
        factors = []
        
        # Analizar historial crediticio
        if features['Credit_History'] == 1:
            factors.append({'factor': 'Historial Crediticio', 'impact': 'Positivo', 'description': 'Tiene historial crediticio favorable'})
        else:
            factors.append({'factor': 'Historial Crediticio', 'impact': 'Negativo', 'description': 'No tiene historial crediticio'})
        
        # Analizar educación
        if features['Education'] == 1:
            factors.append({'factor': 'Educación', 'impact': 'Positivo', 'description': 'Nivel educativo universitario'})
        else:
            factors.append({'factor': 'Educación', 'impact': 'Neutro', 'description': 'Educación secundaria'})
        
        # Analizar ratio deuda-ingreso
        debt_ratio = features['DebtIncomeRatio']
        if debt_ratio < 0.3:
            factors.append({'factor': 'Ratio Deuda/Ingreso', 'impact': 'Positivo', 'description': 'Ratio deuda-ingreso saludable'})
        elif debt_ratio > 0.5:
            factors.append({'factor': 'Ratio Deuda/Ingreso', 'impact': 'Negativo', 'description': 'Ratio deuda-ingreso elevado'})
        
        # Analizar estado civil combinado con historial
        if features['Married_CH'] == 1:
            factors.append({'factor': 'Estado Civil + Crédito', 'impact': 'Positivo', 'description': 'Casado con buen historial crediticio'})
        
        return factors[:4]  # Retornar máximo 4 factores principales
    
    def predict_batch(self, file_path):
        """Realizar predicción por lotes mejorada"""
        try:
            # Cargar archivo
            df = pd.read_csv(file_path)
            print(f"📊 Procesando {len(df)} registros...")
            
            # Guardar IDs si existen
            if 'Loan_ID' in df.columns:
                ids = df['Loan_ID']
                df_features = df.drop(columns=['Loan_ID'])
            else:
                ids = pd.Series([f'LOAN_{i:04d}' for i in range(len(df))])
                df_features = df
            
            # Procesar cada fila individualmente para manejar la codificación
            results = []
            for idx, row in df_features.iterrows():
                try:
                    # Preparar características para cada fila
                    X_processed = self.prepare_features(row.to_dict())
                    
                    # Realizar predicción
                    prediction = self.model.predict(X_processed)[0]
                    probabilities = self.model.predict_proba(X_processed)[0]
                    
                    results.append({
                        'Loan_ID': ids.iloc[idx],
                        'Predicted_Loan_Status': prediction,
                        'Probability_Approved': probabilities[1],
                        'Probability_Rejected': probabilities[0],
                        'Status': 'Aprobado' if prediction == 1 else 'Rechazado'
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error procesando fila {idx}: {e}")
                    # Agregar resultado por defecto
                    results.append({
                        'Loan_ID': ids.iloc[idx],
                        'Predicted_Loan_Status': 0,
                        'Probability_Approved': 0.0,
                        'Probability_Rejected': 1.0,
                        'Status': 'Error'
                    })
            
            results_df = pd.DataFrame(results)
            print(f"✅ Predicciones completadas: {len(results_df)} registros")
            
            return results_df
            
        except Exception as e:
            raise Exception(f"Error en predicción por lotes: {str(e)}")
    
    def get_model_metrics(self):
        """Obtener métricas detalladas del modelo"""
        try:
            # Cargar datos de prueba
            df_test = pd.read_csv('data/loan_test2_clean.csv')
            
            X_test = df_test.drop(columns=['Loan_ID', 'Loan_Status'])
            y_test = df_test['Loan_Status']
            
            # Realizar predicciones
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'roc_auc': float(roc_auc_score(y_test, y_proba)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'feature_importance': self.feature_importance_ if self.feature_importance_ else {}
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculando métricas: {e}")
            return None
    
    def get_feature_importance(self):
        """Obtener importancia de características"""
        if self.feature_importance_:
            # Ordenar por importancia
            sorted_features = sorted(
                self.feature_importance_.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return dict(sorted_features)
        return {}