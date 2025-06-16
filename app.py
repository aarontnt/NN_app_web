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
import pickle
import numpy as np


def crear_app():
    class StressNeuralModel(BaseEstimator, ClassifierMixin):
        """Modelo simplificado que solo usa la red neuronal y el scaler"""
        def __init__(self, nn_model, scaler):
            self.nn_model = nn_model
            self.scaler = scaler
            self.labels = ['Bajo Estrés', 'Medio Estrés', 'Alto Estrés']
            self.classes_ = [0, 1, 2]
        
        def predict(self, X):
            """Predice las clases usando solo la red neuronal"""
            # Convertir a DataFrame si es necesario
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[
                    'self_esteem', 'sleep_quality', 'academic_performance',
                    'social_support', 'depression', 'study_load', 'bullying'
                ])
            
            # Preprocesar para NN
            X_scaled = self.scaler.transform(X)
            
            # Obtener predicciones de la red neuronal
            nn_probs = self.nn_model.predict(X_scaled, verbose=0)
            
            # Devolver las clases predichas
            return np.argmax(nn_probs, axis=1)
        
        def predict_proba(self, X):
            """Devuelve las probabilidades de cada clase"""
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[
                    'self_esteem', 'sleep_quality', 'academic_performance',
                    'social_support', 'depression', 'study_load', 'bullying'
                ])
            
            X_scaled = self.scaler.transform(X)
            nn_probs = self.nn_model.predict(X_scaled, verbose=0)
            
            return nn_probs
        
        def predict_to_labels(self, X):
            """Convierte predicciones numéricas a etiquetas legibles"""
            preds = self.predict(X)
            return [self.labels[p] for p in preds]

    app = Flask(__name__)

    # Configuración de subida de archivos
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'csv'}
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite de 16MB

    # Variable global para el modelo
    stress_model = None

    def allowed_file(filename):
        """Verifica si el archivo tiene una extensión permitida"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    def load_stress_model():
        """Carga el modelo de red neuronal y el scaler"""
        try:
            print("\n🔄 Cargando modelo de análisis de estrés...")
            
            # Lista de posibles nombres de archivos del modelo
            model_files = [
                'best_stress_model_advanced.h5',
                'stress_nn_model.h5',
                'model.h5'
            ]
            
            # Lista de posibles nombres de archivos del scaler
            scaler_files = [
                'robust_scaler.pkl',
                'scaler.pkl'
            ]
            
            # Buscar archivo del modelo
            nn_model = None
            for model_file in model_files:
                if os.path.exists(model_file):
                    print(f"📥 Cargando modelo desde: {model_file}")
                    nn_model = load_keras_model(model_file)
                    break
            
            if nn_model is None:
                raise FileNotFoundError("No se encontró ningún archivo de modelo (.h5)")
            
            # Buscar archivo del scaler
            scaler = None
            for scaler_file in scaler_files:
                if os.path.exists(scaler_file):
                    print(f"📥 Cargando scaler desde: {scaler_file}")
                    scaler = joblib.load(scaler_file)
                    break
            
            if scaler is None:
                raise FileNotFoundError("No se encontró ningún archivo de scaler (.pkl)")
            
            # Crear instancia del modelo
            model = StressNeuralModel(
                nn_model=nn_model,
                scaler=scaler
            )
            
            print("✅ Modelo cargado correctamente")
            return model
            
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {str(e)}")
            traceback.print_exc()
            raise

    def validate_data(df):
        """Valida que el DataFrame tenga las columnas correctas y valores válidos"""
        required_columns = [
            'self_esteem', 'sleep_quality', 'academic_performance',
            'social_support', 'depression', 'study_load', 'bullying'
        ]
        
        # Verificar columnas faltantes
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f'Faltan columnas requeridas: {", ".join(missing_columns)}'
        
        # Verificar valores nulos
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            null_cols = null_counts[null_counts > 0].index.tolist()
            return False, f'Se encontraron valores nulos en: {", ".join(null_cols)}'
        
        # Verificar tipos de datos numéricos
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        return False, f'La columna "{col}" contiene valores no numéricos'
                except:
                    return False, f'No se pudo convertir la columna "{col}" a numérica'
        
        return True, "Datos válidos"

    @app.route('/')
    def home():
        """Página principal"""
        return render_template('index.html')

    @app.route('/health')
    def health_check():
        """Endpoint para verificar el estado del servidor"""
        global stress_model
        model_status = "loaded" if stress_model is not None else "not_loaded"
        return jsonify({
            'status': 'ok',
            'model_status': model_status,
            'message': 'Servidor funcionando correctamente'
        })

    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Maneja la subida y validación inicial de archivos CSV"""
        try:
            # Verificar que se envió un archivo
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No se envió ningún archivo'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Solo se aceptan archivos CSV'}), 400
            
            # Guardar archivo
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            # Leer y validar CSV
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='latin-1')
                except:
                    return jsonify({
                        'success': False, 
                        'error': 'No se pudo leer el archivo CSV. Verifica la codificación.'
                    }), 400
            
            # Validar estructura de datos
            is_valid, validation_message = validate_data(df)
            if not is_valid:
                # Eliminar archivo si no es válido
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    'success': False,
                    'error': validation_message,
                    'required_columns': [
                        'self_esteem', 'sleep_quality', 'academic_performance',
                        'social_support', 'depression', 'study_load', 'bullying'
                    ]
                }), 400
            
            # Generar preview
            preview = df.head(5).to_dict(orient='records')
            
            return jsonify({
                'success': True,
                'filename': filename,
                'preview': preview,
                'total_rows': len(df),
                'message': 'Archivo cargado y validado correctamente'
            })
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Error al procesar archivo: {str(e)}'
            }), 500

    @app.route('/analyze', methods=['POST'])
    def analyze_data():
        """Realiza el análisis de estrés usando el modelo cargado"""
        global stress_model
        
        try:
            # Verificar que el modelo esté cargado
            if stress_model is None:
                return jsonify({
                    'error': 'Modelo no inicializado. Reinicia el servidor.'
                }), 500
            
            # Obtener datos de la solicitud
            data = request.json
            filename = data.get('filename')
            if not filename:
                return jsonify({'error': 'Nombre de archivo no proporcionado'}), 400
            
            # Verificar que el archivo exista
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': 'Archivo no encontrado'}), 404
            
            # Leer archivo
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin-1')
            
            # Validar datos nuevamente
            is_valid, validation_message = validate_data(df)
            if not is_valid:
                return jsonify({'error': validation_message}), 400
            
            # Preparar datos para predicción
            required_columns = [
                'self_esteem', 'sleep_quality', 'academic_performance',
                'social_support', 'depression', 'study_load', 'bullying'
            ]
            X = df[required_columns]
            
            print(f"\n🔍 Analizando {len(X)} muestras...")
            
            # Realizar predicciones
            try:
                predictions = stress_model.predict(X)
                probabilities = stress_model.predict_proba(X)
                confidences = np.max(probabilities, axis=1) * 100
                labels = [stress_model.labels[p] for p in predictions]
                
            except Exception as e:
                error_msg = f'Error en predicción: {str(e)}'
                print(error_msg)
                traceback.print_exc()
                return jsonify({'error': error_msg}), 500
            
            # Preparar resultados detallados
            results = []
            stats = {'alto': 0, 'medio': 0, 'bajo': 0}
            
            for i, (pred, label, conf, probs) in enumerate(zip(predictions, labels, confidences, probabilities)):
                # Determinar color y nivel de riesgo
                if pred == 0:  # Bajo estrés
                    color = 'green'
                    risk = 'Bajo'
                    stats['bajo'] += 1
                elif pred == 1:  # Medio estrés
                    color = 'orange'
                    risk = 'Moderado'
                    stats['medio'] += 1
                else:  # Alto estrés
                    color = 'red'
                    risk = 'Alto'
                    stats['alto'] += 1
                
                results.append({
                    'id': i + 1,
                    'prediction': label,
                    'confidence': round(float(conf), 1),
                    'risk': risk,
                    'color': color,
                    'probabilities': {
                        'bajo': round(float(probs[0]) * 100, 1),
                        'medio': round(float(probs[1]) * 100, 1),
                        'alto': round(float(probs[2]) * 100, 1)
                    }
                })
            
            # Calcular estadísticas generales
            total_samples = len(predictions)
            stats.update({
                'total_muestras': total_samples,
                'porcentaje_bajo': round((stats['bajo'] / total_samples) * 100, 1),
                'porcentaje_medio': round((stats['medio'] / total_samples) * 100, 1),
                'porcentaje_alto': round((stats['alto'] / total_samples) * 100, 1),
                'confianza_promedio': round(float(np.mean(confidences)), 1),
                'confianza_minima': round(float(np.min(confidences)), 1),
                'confianza_maxima': round(float(np.max(confidences)), 1)
            })
            
            print(f"✅ Análisis completado: {stats['bajo']} bajo, {stats['medio']} medio, {stats['alto']} alto estrés")
            
            return jsonify({
                'success': True,
                'results': results,
                'stats': stats,
                'message': f'Análisis completado para {total_samples} muestras'
            })
            
        except Exception as e:
            error_msg = f'Error inesperado en analyze_data: {str(e)}'
            print(f"\n❌ {error_msg}")
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500

    @app.route('/model-info')
    def model_info():
        """Proporciona información sobre el modelo cargado"""
        global stress_model
        
        if stress_model is None:
            return jsonify({
                'model_loaded': False,
                'error': 'Modelo no cargado'
            })
        
        try:
            # Información básica del modelo
            model_summary = {
                'model_loaded': True,
                'model_type': 'Red Neuronal para Análisis de Estrés',
                'classes': stress_model.labels,
                'required_features': [
                    'self_esteem', 'sleep_quality', 'academic_performance',
                    'social_support', 'depression', 'study_load', 'bullying'
                ],
                'feature_descriptions': {
                    'self_esteem': 'Autoestima (escala numérica)',
                    'sleep_quality': 'Calidad del sueño (escala numérica)',
                    'academic_performance': 'Rendimiento académico (escala numérica)',
                    'social_support': 'Soporte social (escala numérica)',
                    'depression': 'Nivel de depresión (escala numérica)',
                    'study_load': 'Carga de estudio (escala numérica)',
                    'bullying': 'Experiencia de bullying (escala numérica)'
                }
            }
            
            return jsonify(model_summary)
            
        except Exception as e:
            return jsonify({
                'model_loaded': False,
                'error': f'Error al obtener información del modelo: {str(e)}'
            })

    @app.route('/cluster', methods=['POST'])
    def cluster_data():
        data = request.json
        filename = data.get('filename')
        n_clusters = data.get('n_clusters', 3)
        
        if not filename:
            return jsonify({'error': 'Nombre de archivo no proporcionado'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64
            
            df = pd.read_csv(filepath)
            X = df[['self_esteem', 'sleep_quality', 'academic_performance', 
                    'social_support', 'depression', 'study_load', 'bullying']]
            
            # Escalar los datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Aplicar K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Calcular distancias a los centroides
            distances = kmeans.transform(X_scaled)
            min_distances = np.min(distances, axis=1)
            
            # Crear gráfico (simplificado para 2 componentes principales)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            centers = pca.transform(kmeans.cluster_centers_)
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
            plt.title('Visualización de Clusters (PCA)')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True)
            
            # Convertir gráfico a base64 para enviar al frontend
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Preparar resultados
            cluster_results = []
            for i, (cluster, dist) in enumerate(zip(clusters, min_distances)):
                cluster_results.append({
                    'id': i+1,
                    'cluster': int(cluster) + 1,  # Mostrar clusters desde 1 en lugar de 0
                    'distance': round(float(dist), 2),
                    'features': {col: float(X.iloc[i][col]) for col in X.columns}
                })
            
            # Estadísticas de clusters
            cluster_stats = []
            for cluster_num in range(n_clusters):
                cluster_data = X[clusters == cluster_num]
                stats = {
                    'cluster': cluster_num + 1,
                    'count': len(cluster_data),
                    'means': cluster_data.mean().round(2).to_dict()
                }
                cluster_stats.append(stats)
            
            return jsonify({
                'success': True,
                'cluster_image': image_base64,
                'results': cluster_results,
                'stats': cluster_stats
            })
            
        except Exception as e:
            return jsonify({'error': f'Error en clustering: {str(e)}'}), 500

    @app.route('/download_sample')
    def download_sample():
        # Crear un CSV de ejemplo con las columnas esperadas
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Encabezados
        headers = ['self_esteem', 'sleep_quality', 'academic_performance', 
                'social_support', 'depression', 'study_load', 'bullying']
        
        # Datos de ejemplo
        example_data = [
            [7, 6, 8, 7, 3, 5, 2],  # Bajo estrés
            [5, 4, 6, 5, 5, 7, 3],   # Medio estrés
            [3, 2, 4, 3, 8, 9, 6]    # Alto estrés
        ]
        
        writer.writerow(headers)
        writer.writerows(example_data)
        
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=ejemplo_estres.csv'
        }


    # Inicialización del servidor
    def initialize_server():
        """Inicializa el servidor cargando el modelo"""
        global stress_model
        try:
            print("🚀 Inicializando servidor de análisis de estrés...")
            stress_model = load_stress_model()
            print("🎉 Servidor inicializado correctamente\n")
        except Exception as e:
            print(f"❌ Error crítico al inicializar servidor: {str(e)}")
            print("⚠️ El servidor se iniciará sin modelo cargado")
            stress_model = None
    return app
if __name__ == '__main__':
    # Inicializar servidor
    initialize_server()
    app = crear_app()
    # Configuración de desarrollo
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )