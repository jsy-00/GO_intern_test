import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Define available models
models = {
    "baseline": './model/baseline.keras',
    "baseline+adam": './model/baseline+adam.keras',
    "baseline+rmsprop": './model/baseline+rmsprop.keras',
}

# Define class labels
class_labels = ['cat', 'dog']

def load_model(model_name):
    return models[model_name](weights='imagenet')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict(image, model):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    class_index = np.argmax(predictions)
    class_name = class_labels[class_index]
    confidence = predictions[0][class_index]
    return class_name, confidence

st.title('Dog and Cat Classifier')

# Model selection
model_name = st.selectbox("Choose a model", list(models.keys()))


# Load the selected model
model = tf.keras.models.load_model(models[model_name], compile=False, safe_mode=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write(f"Classifying...")
    class_name, confidence = predict(image, model)
    st.write(f"Prediction: {class_name}")
    st.write(f"Confidence: {confidence:.2f}")