import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model as load_keras_model
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from flask import Flask

def crear_app():
    app = Flask(__name__)
    
    # =====================================================================
    # Configuración inicial
    # =====================================================================
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.config['ALLOWED_EXTENSIONS'] = {'csv'}
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

    # =====================================================================
    # Clases y Modelos
    # =====================================================================
    class StressNeuralModel(BaseEstimator, ClassifierMixin):
        """Modelo simplificado que solo usa la red neuronal y el scaler"""
        def __init__(self, nn_model, scaler):
            self.nn_model = nn_model
            self.scaler = scaler
            self.labels = ['Bajo Estrés', 'Medio Estrés', 'Alto Estrés']
            self.classes_ = [0, 1, 2]
        
        def predict(self, X):
            """Predice las clases usando solo la red neuronal"""
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[
                    'self_esteem', 'sleep_quality', 'academic_performance',
                    'social_support', 'depression', 'study_load', 'bullying'
                ])
            X_scaled = self.scaler.transform(X)
            nn_probs = self.nn_model.predict(X_scaled, verbose=0)
            return np.argmax(nn_probs, axis=1)
        
        def predict_proba(self, X):
            """Devuelve las probabilidades de cada clase"""
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[
                    'self_esteem', 'sleep_quality', 'academic_performance',
                    'social_support', 'depression', 'study_load', 'bullying'
                ])
            X_scaled = self.scaler.transform(X)
            return self.nn_model.predict(X_scaled, verbose=0)
        
        def predict_to_labels(self, X):
            """Convierte predicciones numéricas a etiquetas legibles"""
            preds = self.predict(X)
            return [self.labels[p] for p in preds]

    # =====================================================================
    # Carga del Modelo (al iniciar la app)
    # =====================================================================
    def load_stress_model():
        """Carga el modelo y el scaler desde archivos"""
        try:
            print("\n🔄 Cargando modelo de análisis de estrés...")
            
            # Buscar archivos del modelo y scaler
            model_path = None
            scaler_path = None
            
            for file in os.listdir():
                if file.endswith('.h5'):
                    model_path = file
                elif file.endswith('.pkl'):
                    scaler_path = file
            
            if not model_path or not scaler_path:
                raise FileNotFoundError("No se encontraron archivos .h5 o .pkl")
            
            print(f"📥 Modelo: {model_path}, Scaler: {scaler_path}")
            nn_model = load_keras_model(model_path)
            scaler = joblib.load(scaler_path)
            
            return StressNeuralModel(nn_model=nn_model, scaler=scaler)
            
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {str(e)}")
            traceback.print_exc()
            return None

    # Cargar el modelo al iniciar
    stress_model = load_stress_model()

    # =====================================================================
    # Funciones Auxiliares
    # =====================================================================
    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    def validate_data(df):
        """Valida que el DataFrame tenga las columnas correctas"""
        required_columns = [
            'self_esteem', 'sleep_quality', 'academic_performance',
            'social_support', 'depression', 'study_load', 'bullying'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f'Faltan columnas: {", ".join(missing_columns)}'
        return True, "Datos válidos"

    # =====================================================================
    # Rutas de la API
    # =====================================================================
    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'ok',
            'model_loaded': stress_model is not None
        })

    @app.route('/upload', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No se envió ningún archivo'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Nombre de archivo vacío'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Solo se permiten archivos CSV'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
            
            is_valid, msg = validate_data(df)
            if not is_valid:
                os.remove(filepath)
                return jsonify({'error': msg}), 400
            
            preview = df.head().to_dict(orient='records')
            return jsonify({
                'success': True,
                'preview': preview,
                'filename': filename
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/analyze', methods=['POST'])
    def analyze_data():
        if stress_model is None:
            return jsonify({'error': 'Modelo no cargado'}), 500
        
        try:
            data = request.json
            filename = data.get('filename')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            df = pd.read_csv(filepath)
            X = df[['self_esteem', 'sleep_quality', 'academic_performance',
                   'social_support', 'depression', 'study_load', 'bullying']]
            
            predictions = stress_model.predict(X)
            probabilities = stress_model.predict_proba(X)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'id': i + 1,
                    'prediction': stress_model.labels[pred],
                    'confidence': float(np.max(prob)),
                    'probabilities': {
                        'bajo': float(prob[0]),
                        'medio': float(prob[1]),
                        'alto': float(prob[2])
                    }
                })
            
            return jsonify({'results': results})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # =====================================================================
    # Clustering (opcional)
    # =====================================================================
    @app.route('/cluster', methods=['POST'])
    def cluster_data():
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            data = request.json
            filename = data.get('filename')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            df = pd.read_csv(filepath)
            X = df[['self_esteem', 'sleep_quality', 'academic_performance',
                    'social_support', 'depression', 'study_load', 'bullying']]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            return jsonify({
                'success': True,
                'clusters': clusters.tolist()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app

app = crear_app()
# Nota: No incluimos app.run() porque Render usa gunicorn
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)