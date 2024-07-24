# Proyecto de Predicción de Salario

Este proyecto incluye una API creada con Flask y una interfaz web para predecir salarios usando un modelo de Machine Learning.

## Configuración del Entorno

1. **Instalar Python**
   - Asegúrate de tener Python instalado. Puedes descargarlo desde [python.org](https://www.python.org/).

2. **Crear un Entorno Virtual**
   - Abre una terminal y navega a la carpeta de tu proyecto.
   - Ejecuta: `python -m venv myenv` (esto crea un entorno virtual llamado `myenv`).
   - Activa el entorno virtual:
     - En Windows: `myenv\Scripts\activate`
     - En macOS/Linux: `source myenv/bin/activate`

3. **Instalar Dependencias**
   - Con el entorno virtual activado, instala Flask y otras librerías necesarias:
     ```bash
     pip install flask numpy pandas scikit-learn
     ```

## Preparar el Modelo de Machine Learning

1. **Entrenar y Guardar el Modelo**
   - Entrena tu modelo de Machine Learning usando tus datos.
   - Guarda el modelo usando `joblib`:
     ```python
     import joblib
     joblib.dump(model, 'model.pkl')
     ```
   - Asegúrate de tener el archivo `model.pkl` en la misma carpeta que tu script Flask.

## Crear la API con Flask

1. **Crear el Script de Flask (app.py)**
   - Crea un archivo `app.py` en tu carpeta de proyecto:
     ```python
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
     ```

2. **Ejecutar el Servidor Flask**
   - En la terminal, con el entorno virtual activado, ejecuta:
     ```bash
     python app.py
     ```
   - Esto iniciará el servidor en `http://127.0.0.1:5000`.

## Crear la Interfaz Web

1. **Crear el Archivo HTML (index.html)**
   - Crea un archivo `index.html` en una carpeta llamada `static` o directamente en la raíz del proyecto:
     ```html
      <!DOCTYPE html>
      <html lang="en">
      
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Predicción de Salario</title>
          <link rel="stylesheet" href="styles.css">
          <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
      </head>
      
      <body>
          <div class="container">
              <h1>Predicción de Salario</h1>
              <form id="prediction-form">
                  <div class="form-group">
                      <label for="edad">
                          <i class="fas fa-calendar-day"></i> Edad:
                      </label>
                      <input type="number" id="edad" name="edad" class="input-narrow" required>
                  </div>
      
                  <div class="form-group">
                      <label for="fnlwgt">
                          <i class="fas fa-calculator"></i> FNLWGT:
                      </label>
                      <input type="number" id="fnlwgt" name="fnlwgt" class="input-wide" required>
                  </div>
      
                  <div class="form-group">
                      <label for="educacion">
                          <i class="fas fa-graduation-cap"></i> Educación:
                      </label>
                      <select id="educacion" name="educacion" class="input-wide" required>
                          <option value="0">10º Grado</option>
                          <option value="1">11º Grado</option>
                          <option value="2">12º Grado</option>
                          <option value="3">1º-4º Grado</option>
                          <option value="4">5º-6º Grado</option>
                          <option value="5">7º-8º Grado</option>
                          <option value="6">9º Grado</option>
                          <option value="7">Asociado Académico</option>
                          <option value="8">Asociado Profesional</option>
                          <option value="9">Licenciatura</option>
                          <option value="10">Doctorado</option>
                          <option value="11">Graduado de Secundaria</option>
                          <option value="12">Máster</option>
                          <option value="13">Preescolar</option>
                          <option value="14">Escuela Profesional</option>
                          <option value="15">Alguna Universidad</option>
                      </select>
                  </div>
      
                  <div class="form-group">
                      <label for="educacion_num">
                          <i class="fas fa-book"></i> Educación (Num):
                      </label>
                      <input type="number" id="educacion_num" name="educacion_num" class="input-narrow" required>
                  </div>
      
                  <div class="form-group">
                      <label for="estado_civil">
                          <i class="fas fa-heart"></i> Estado Civil:
                      </label>
                      <select id="estado_civil" name="estado_civil" class="input-wide" required>
                          <option value="0">Con Pareja</option>
                          <option value="1">Soltero</option>
                      </select>
                  </div>
      
                  <div class="form-group">
                      <label for="relacion">
                          <i class="fas fa-users"></i> Relación:
                      </label>
                      <select id="relacion" name="relacion" class="input-wide" required>
                          <option value="0">Esposo</option>
                          <option value="1">No en la familia</option>
                          <option value="2">Otro pariente</option>
                          <option value="3">Hijo propio</option>
                          <option value="4">No casado</option>
                          <option value="5">Esposa</option>
                      </select>
                  </div>
      
                  <div class="form-group">
                      <label for="ocupacion">
                          <i class="fas fa-briefcase"></i> Ocupación:
                      </label>
                      <select id="ocupacion" name="ocupacion" class="input-wide" required>
                          <option value="0">Administrativo-Clerical</option>
                          <option value="1">Fuerzas Armadas</option>
                          <option value="2">Artesanía-Reparación</option>
                          <option value="3">Ejecutivo-Gerencial</option>
                          <option value="4">Agricultura-Pesca</option>
                          <option value="5">Manipuladores-Limpiadores</option>
                          <option value="6">Operador-Maquinaria</option>
                          <option value="7">Otro Servicio</option>
                          <option value="8">Servicios en el Hogar</option>
                          <option value="9">Especialidad Profesional</option>
                          <option value="10">Servicios de Protección</option>
                          <option value="11">Ventas</option>
                          <option value="12">Soporte Técnico</option>
                          <option value="13">Transporte-Movimiento</option>
                      </select>
                  </div>
      
                  <div class="form-group">
                      <label for="raza">
                          <i class="fas fa-user-tie"></i> Raza:
                      </label>
                      <select id="raza" name="raza" class="input-wide" required>
                          <option value="0">Amerindio-Eskimo</option>
                          <option value="1">Asiático-Pacífico</option>
                          <option value="2">Negro</option>
                          <option value="3">Otro</option>
                          <option value="4">Blanco</option>
                      </select>
                  </div>
      
                  <div class="form-group">
                      <label for="genero">
                          <i class="fas fa-genderless"></i> Género:
                      </label>
                      <select id="genero" name="genero" class="input-wide" required>
                          <option value="0">Mujer</option>
                          <option value="1">Hombre</option>
                      </select>
                  </div>
      
                  <div class="form-group">
                      <label for="ganancia-capital">
                          <i class="fas fa-dollar-sign"></i> Ganancia Capital:
                      </label>
                      <input type="number" id="ganancia-capital" name="ganancia-capital" class="input-narrow" required>
                  </div>
      
                  <div class="form-group">
                      <label for="perdida-capital">
                          <i class="fas fa-dollar-sign"></i> Pérdida Capital:
                      </label>
                      <input type="number" id="perdida-capital" name="perdida-capital" class="input-narrow" required>
                  </div>
      
                  <div class="form-group">
                      <label for="horas_semana">
                          <i class="fas fa-clock"></i> Horas por Semana:
                      </label>
                      <input type="number" id="horas_semana" name="horas_semana" class="input-narrow" required>
                  </div>
      
                  <div class="form-group">
                      <label for="pais">
                          <i class="fas fa-flag"></i> País:
                      </label>
                      <select id="pais" name="pais" class="input-wide" required>
                          <option value="0">No estadounidense</option>
                          <option value="1">Estadounidense</option>
                      </select>
                  </div>
      
                  <div class="form-group">
                      <label for="tipo_empleo">
                          <i class="fas fa-building"></i> Tipo de Empleo:
                      </label>
                      <select id="tipo_empleo" name="tipo_empleo" class="input-wide" required>
                          <option value="0">Gobierno</option>
                          <option value="1">Privado</option>
                          <option value="2">Autónomo</option>
                          <option value="3">Sin Pago</option>
                      </select>
                  </div>
      
                  <button type="submit">
                      <i class="fas fa-calculator"></i> Predecir
                  </button>
              </form>
              <div id="result"></div>
          </div>
      
          <script src="script.js"></script>
      </body>
      
      </html>

     ```

2. **Crear el Archivo CSS (styles.css)**
   - Crea un archivo `styles.css` en la misma carpeta que `index.html`:
     ```css
      /* styles.css */
      body {
          font-family: Arial, sans-serif;
          display: flex;
          justify-content: center;
          align-items: center;
          height: 100vh;
          margin: 0;
          background-color: #f4f4f4;
      }
      
      .container {
          background: #dfdada;
          padding: 30px;
          border-radius: 12px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
          max-width: 1000px;
          width: 100%;
      }
      
      h1 {
          color: #333;
          margin-bottom: 20px;
          text-align: center;
      }
      
      form {
          display: grid;
          gap: 20px;
          grid-template-columns: 1fr 1fr;
          align-items: center;
      }
      
      .form-group {
          display: flex;
          flex-direction: column;
      }
      
      label {
          font-weight: bold;
          color: #555;
      }
      
      input,
      select {
          border: 1px solid #ddd;
          border-radius: 6px;
          padding: 10px;
          font-size: 16px;
          width: 80%;
      }
      
      button {
          border: 1px solid #ddd;
          border-radius: 6px;
          padding: 10px;
          font-size: 16px;
          width: 100%;
      }
      
      input[type="number"],
      select {
          background: #fafafa;
      }
      
      button {
          background: #007BFF;
          color: #fff;
          border: none;
          cursor: pointer;
          font-size: 18px;
          font-weight: bold;
          transition: background 0.3s ease;
          grid-column: span 2;
          text-align: center;
          margin-top: 20px;
          /* Añadir margen superior */
      }
      
      button:hover {
          background: #0056b3;
      }
      
      #result {
          margin-top: 20px;
          font-size: 18px;
          font-weight: bold;
          text-align: center;
          padding: 15px;
          border-radius: 6px;
      }
      
      #result.success {
          background: #d4edda;
          color: #155724;
          border: 1px solid #c3e6cb;
      }
      
      #result.error {
          background: #f8d7da;
          color: #721c24;
          border: 1px solid #f5c6cb;
      }
      
      @media (max-width: 600px) {
          form {
              grid-template-columns: 1fr;
          }
      }
     ```

3. **Crear el Archivo JavaScript (script.js)**
   - Crea un archivo `script.js` en la misma carpeta que `index.html`:
     ```javascript
      // script.js
      document.getElementById('prediction-form').addEventListener('submit', async function(event) {
          event.preventDefault();
      
          const formData = new FormData(event.target);
          const data = Object.fromEntries(formData);
      
          try {
              const response = await fetch('http://127.0.0.1:5000/predict', {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json'
                  },
                  body: JSON.stringify(data)
              });
      
              if (response.ok) {
                  const result = await response.json();
                  
                  // Traducción de la predicción
                  const predictionText = result.prediction === 0 ? '>50K' : '<=50K';
                  
                  document.getElementById('result').textContent = 'Predicción: ' + predictionText;
              } else {
                  const error = await response.json();
                  document.getElementById('result').textContent = 'Error: ' + error.error;
              }
          } catch (error) {
              document.getElementById('result').textContent = 'Error: ' + error.message;
          }
      });

     ```

## Ejecutar y Probar

1. **Iniciar el Servidor Flask**
   - Asegúrate de que el archivo `app.py` esté en el directorio raíz del proyecto y ejecuta:

     ```bash
     python app.py
     ```

2. **Abrir la Interfaz Web**
   - Abre tu navegador web y navega a `http://127.0.0.1:5000` para ver y probar la interfaz web.
   - El formulario debería funcionar para enviar datos a la API y recibir predicciones.

---

¡Con estos pasos deberías poder configurar y ejecutar el proyecto completo de predicción de salarios!
