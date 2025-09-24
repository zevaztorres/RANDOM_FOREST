import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class MedicalDiagnosisModel:
    def __init__(self):
        self.model = None
        self.feature_names = ['respira', 'dolor_pecho', 'tos', 'fiebre', 'fatiga', 
                             'mareos', 'nauseas', 'sudoracion', 'palpitaciones', 
                             'dificultad_respirar']
        self.diagnosis_map = {
            0: "Salud normal", 
            1: "Posible problema cardíaco", 
            2: "Posible problema respiratorio", 
            3: "Emergencia inmediata"
        }
    
    def create_enhanced_dataset(self, n_samples=200):
        """Crea un dataset sintético más robusto con más casos y síntomas"""
        np.random.seed(42)
        
        # Generar datos sintéticos más realistas
        data = {
            'respira': np.random.choice([0, 1], n_samples, p=[0.05, 0.95]),  # 95% respira normalmente
            'dolor_pecho': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'tos': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'fiebre': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'fatiga': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'mareos': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'nauseas': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'sudoracion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'palpitaciones': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'dificultad_respirar': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Lógica mejorada para asignar diagnósticos basada en síntomas
        diagnosticos = []
        for i in range(n_samples):
            row = df.iloc[i]
            
            # Emergencia: no respira
            if row['respira'] == 0:
                diagnosticos.append(3)
            # Emergencia: múltiples síntomas graves
            elif (row['dolor_pecho'] == 1 and row['dificultad_respirar'] == 1 and 
                  row['sudoracion'] == 1 and row['palpitaciones'] == 1):
                diagnosticos.append(3)
            # Problema cardíaco
            elif (row['dolor_pecho'] == 1 and row['fatiga'] == 1) or \
                 (row['palpitaciones'] == 1 and row['mareos'] == 1) or \
                 (row['dolor_pecho'] == 1 and row['sudoracion'] == 1):
                diagnosticos.append(1)
            # Problema respiratorio
            elif (row['tos'] == 1 and row['fiebre'] == 1) or \
                 (row['dificultad_respirar'] == 1 and row['tos'] == 1) or \
                 (row['fiebre'] == 1 and row['fatiga'] == 1 and row['tos'] == 1):
                diagnosticos.append(2)
            # Normal (pocos o ningún síntoma)
            else:
                # Si tiene muy pocos síntomas, probablemente normal
                sintomas_count = sum([row['dolor_pecho'], row['tos'], row['fiebre'], 
                                    row['fatiga'], row['mareos'], row['nauseas'], 
                                    row['sudoracion'], row['palpitaciones'], 
                                    row['dificultad_respirar']])
                if sintomas_count <= 1:
                    diagnosticos.append(0)
                else:
                    # Asignar aleatoriamente entre las categorías con pesos
                    diagnosticos.append(np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3]))
        
        df['diagnostico'] = diagnosticos
        return df

    def train_model(self, df=None, n_estimators=100, max_depth=None, min_samples_split=2, 
                   min_samples_leaf=1, test_size=0.2, cv_folds=5, n_samples=200):
        """Entrena el modelo con parámetros personalizables y devuelve métricas detalladas"""
        
        # Si no se proporciona dataset, crear uno nuevo
        if df is None:
            df = self.create_enhanced_dataset(n_samples=n_samples)
        
        X = df[self.feature_names]
        y = df['diagnostico']
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Crear modelo con parámetros personalizados
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        
        # Validación cruzada
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds)
        
        # Evaluación en conjunto de prueba
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, 
                                           target_names=list(self.diagnosis_map.values()),
                                           output_dict=True)
        
        # Compilar métricas
        metrics = {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(self.feature_names),
                'classes': len(self.diagnosis_map)
            },
            'model_params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'test_size': test_size,
                'cv_folds': cv_folds
            }
        }
        
        return self.model, metrics
    
    def save_model(self, filename='enhanced_rf_model.pkl'):
        """Guarda el modelo entrenado"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'diagnosis_map': self.diagnosis_map
            }
            joblib.dump(model_data, filename)
            print(f"Modelo guardado como {filename}")
        else:
            print("No hay modelo entrenado para guardar")
    
    def load_model(self, filename='enhanced_rf_model.pkl'):
        """Carga un modelo previamente entrenado"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.diagnosis_map = model_data['diagnosis_map']
            print(f"Modelo cargado desde {filename}")
            return True
        except FileNotFoundError:
            print(f"Archivo {filename} no encontrado")
            return False
    
    def predict(self, symptoms_dict):
        """Realiza predicción con validación de entrada"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train_model() primero.")
        
        # Validar que todos los síntomas requeridos estén presentes
        for feature in self.feature_names:
            if feature not in symptoms_dict:
                symptoms_dict[feature] = 0  # Valor por defecto
        
        # Crear DataFrame con el orden correcto de características
        input_df = pd.DataFrame([symptoms_dict])[self.feature_names]
        
        # Realizar predicción
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        
        return {
            'diagnosis': self.diagnosis_map[prediction],
            'diagnosis_code': prediction,
            'probability': probabilities[prediction],
            'all_probabilities': {
                self.diagnosis_map[i]: prob for i, prob in enumerate(probabilities)
            }
        }

def main():
    """Función principal para entrenar y guardar el modelo"""
    print("Iniciando entrenamiento del modelo de diagnóstico médico mejorado...")
    
    # Crear instancia del modelo
    medical_model = MedicalDiagnosisModel()
    
    # Crear dataset expandido
    print("Creando dataset sintético expandido...")
    df = medical_model.create_enhanced_dataset()
    print(f"Dataset creado con {len(df)} muestras")
    print(f"Distribución de diagnósticos:")
    print(df['diagnostico'].value_counts().sort_index())
    
    # Entrenar modelo
    model, metrics = medical_model.train_model(df)
    
    # Mostrar métricas
    print(f"\nPrecisión en conjunto de prueba: {metrics['accuracy']:.4f}")
    print(f"Score promedio CV: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std'] * 2:.4f})")
    print(f"\nImportancia de características:")
    print(metrics['feature_importance'])
    
    # Guardar modelo
    medical_model.save_model()
    
    print("\n¡Entrenamiento completado exitosamente!")
    
    # Ejemplo de uso
    print("\nEjemplo de predicción:")
    test_symptoms = {
        'respira': 1,
        'dolor_pecho': 1,
        'tos': 0,
        'fiebre': 0,
        'fatiga': 1,
        'mareos': 0,
        'nauseas': 0,
        'sudoracion': 1,
        'palpitaciones': 1,
        'dificultad_respirar': 0
    }
    
    result = medical_model.predict(test_symptoms)
    print(f"Síntomas: {test_symptoms}")
    print(f"Diagnóstico: {result['diagnosis']}")
    print(f"Probabilidad: {result['probability']:.4f}")
    print(f"Todas las probabilidades: {result['all_probabilities']}")

if __name__ == "__main__":
    main()