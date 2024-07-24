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
