from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = joblib.load('SVModel.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        expected_keys = ["edad", "fnlwgt", "educacion", "educacion_num", "estado_civil", "relacion", "ocupacion", 
                         "raza", "genero", "ganancia-capital", "perdida-capital", "horas_semana", "pais", "tipo_empleo"]
        for key in expected_keys:
            if key not in data:
                return jsonify({'error': f'Missing key: {key} in JSON request'}), 400

        input_data = np.array([data[key] for key in expected_keys]).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

if __name__ == '__main__':
    app.run(debug=True)
