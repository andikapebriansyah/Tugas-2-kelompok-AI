import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import gdown
import os
import zipfile

# File dan direktori model
model_url = "https://drive.google.com/file/d/15sVxzeibKRZxuttAfO6lqfjbLDGfRbS6/view?usp=sharing"  
model_zip_path = "animals_model.zip"
model_dir = "saved_model/animals_model_1"

# Unduh dan ekstrak model jika belum ada
if not os.path.exists(model_dir):
    st.write("Downloading model, please wait...")
    gdown.download(model_url, model_zip_path, quiet=False)
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall("saved_model/")
    st.write("Model downloaded and extracted!")

# Muat model
st.write("Loading model...")
model = tf.saved_model.load(model_dir)
st.write("Model loaded successfully!")

# Class labels
class_labels = ['sheep', 'dog', 'cat', 'elephant', 'butterfly', 'horse', 'spider', 'chicken', 'cow', 'squirrel']

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize sesuai input model
    image = np.array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image.astype(np.float32)

# Antarmuka Streamlit
st.title('Animal Classification Web App')
st.write('Upload an image of an animal to get the prediction.')

# Upload gambar
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess gambar
    preprocessed_image = preprocess_image(image)
    
    # Prediksi
    infer = model.signatures["serving_default"]  # Signature default model
    predictions = infer(tf.convert_to_tensor(preprocessed_image))
    output_key = list(predictions.keys())[0]  # Periksa key output
    output = predictions[output_key].numpy()

    # Tentukan prediksi
    predicted_class = np.argmax(output)
    confidence = np.max(output) * 100

    # Tampilkan hasil
    st.write(f"Prediction: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
