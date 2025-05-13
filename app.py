from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
import io
import base64

# Crear una aplicación Flask
app = Flask(__name__)

# Cargar el modelo
def load_model():
    # Cargar el archivo JSON
    with open('model_numeros_1_a_10.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Cargar los pesos
    model.load_weights('model_numeros_1_a_10.weights.h5')
    
    # Compilar el modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Cargar el modelo al inicio
model = load_model()

# Función para preprocesar la imagen
def preprocess_image(image_bytes):
    # Convertir la imagen a un formato que TensorFlow pueda usar
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28 píxeles
    image = np.array(image)  # Convertir a array de NumPy
    image = image.astype('float32') / 255.0  # Normalizar
    image = np.expand_dims(image, axis=-1)  # Añadir el canal
    image = np.expand_dims(image, axis=0)  # Añadir la dimensión de batch
    return image

# Endpoint para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la imagen desde la solicitud POST
        img_data = request.json.get('image')
        
        # Decodificar la imagen (asumimos que es una imagen base64)
        img_data = base64.b64decode(img_data)
        
        # Preprocesar la imagen
        image = preprocess_image(img_data)
        
        # Realizar la predicción
        predictions = model.predict(image)
        
        # Obtener la clase con mayor probabilidad
        predicted_class = np.argmax(predictions)
        
        # Devolver la respuesta como JSON
        return jsonify({'prediction': int(predicted_class), 'probability': float(np.max(predictions))})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta principal
@app.route('/')
def index():
    return "Bienvenido a la API de predicción de números del 0 al 9."

if __name__ == '__main__':
    app.run(debug=True)
