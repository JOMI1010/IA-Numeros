import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
import io
import base64

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

# Título de la aplicación
st.title('Predicción de Números MNIST')

# Explicación
st.write("""
    Esta aplicación utiliza un modelo entrenado para reconocer números escritos a mano (0-9) 
    usando el conjunto de datos MNIST. Suba una imagen para hacer una predicción.
""")

# Subir la imagen
uploaded_file = st.file_uploader("Sube una imagen de un número (0-9)", type=["jpg", "png", "jpeg"])

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

# Si se ha subido una imagen
if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption="Imagen Subida", use_column_width=True)

    # Preprocesar la imagen
    image = preprocess_image(uploaded_file.getvalue())

    # Realizar la predicción
    predictions = model.predict(image)

    # Obtener la clase con mayor probabilidad
    predicted_class = np.argmax(predictions)
    predicted_probability = np.max(predictions)

    # Mostrar la predicción
    st.write(f"El número predicho es: **{predicted_class}**")
    st.write(f"Probabilidad de la predicción: **{predicted_probability:.2f}**")

if __name__ == '__main__':
    app.run(debug=True)
