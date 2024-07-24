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
     joblib.dump(model, 'model.joblib')
     ```
   - Asegúrate de tener el archivo `model.joblib` en la misma carpeta que tu script Flask.

## Crear la API con Flask

1. **Crear el Script de Flask (app.py)**
   - Crea un archivo `app.py` en tu carpeta de proyecto:
     ```python
     from flask import Flask, request, jsonify
     import joblib
     import numpy as np

     app = Flask(__name__)
     model = joblib.load('model.joblib')

     @app.route('/predict', methods=['POST'])
     def predict():
         data = request.json
         features = np.array([[
             data['edad'], data['fnlwgt'], data['educacion_num'],
             data['estado_civil'], data['relacion'], data['ocupacion'],
             data['raza'], data['genero'], data['ganancia-capital'],
             data['perdida-capital'], data['horas_semana'], data['pais'],
             data['tipo_empleo']
         ]])
         prediction = model.predict(features)[0]
         result = {'prediction': prediction}
         return jsonify(result)

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
                     <label for="edad"><i class="fas fa-calendar-alt"></i> Edad:</label>
                     <input type="number" id="edad" name="edad" required>
                 </div>
                 <!-- Repite para los demás campos -->
                 <button type="submit">Predecir</button>
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
         background: #ffffff;
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
         grid-template-columns: repeat(2, 1fr);
     }

     .form-group {
         display: flex;
         flex-direction: column;
         gap: 10px;
     }

     label {
         font-weight: bold;
         color: #555;
         display: flex;
         align-items: center;
     }

     input, select, button {
         border: 1px solid #ddd;
         border-radius: 6px;
         padding: 10px;
         font-size: 16px;
         width: 100%;
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
     ```

3. **Crear el Archivo JavaScript (script.js)**
   - Crea un archivo `script.js` en la misma carpeta que `index.html`:
     ```javascript
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
                 document.getElementById('result').textContent = result.prediction === 0 ? '>50K' : '<=50K';
                 document.getElementById('result').className = 'success';
             } else {
                 const error = await response.json();
                 document.getElementById('result').textContent = 'Error: ' + error.error;
                 document.getElementById('result').className = 'error';
             }
         } catch (error) {
             document.getElementById('result').textContent = 'Error: ' + error.message;
             document.getElementById('result').className = 'error';
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
   - Abre tu navegador web y navega a `http://127.0.0.1:5000` para ver y probar tu interfaz web. El formulario debería funcionar para enviar datos a la API y recibir predicciones.

---

¡Con estos pasos deberías poder configurar y ejecutar el proyecto completo de predicción de salarios!
