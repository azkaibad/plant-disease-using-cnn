import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# Load the trained model
model = load_model('PlantDiseaseModel.h5')

# Define categories (labels)
categories = ['Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Potato___healthy',
              'Tomato_Late_blight', 'Tomato_Bacterial_spot', 'Pepper__bell___healthy',
              'Tomato_Septoria_leaf_spot', 'Tomato_healthy', 'Tomato__Tomato_mosaic_virus',
              'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Early_blight', 'Potato___Late_blight',
              'Tomato_Leaf_Mold', 'Potato___Early_blight', 'Pepper__bell___Bacterial_spot']

# Function to preprocess image and predict


def predict_image(image):
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = categories[predicted_class[0]]
    return predicted_label

# Function to predict image from array


def predict_image_array(image_array):
    # Konversi NumPy array ke objek Image (diasumsikan image_array adalah array NumPy)
    image = Image.fromarray(image_array)

    # Resize gambar ke ukuran 128x128
    resized_image = image.resize((128, 128))

    # Konversi ke array NumPy dan normalisasi
    img_array = np.array(resized_image)
    img_array = img_array / 255.0

    # Menambahkan dimensi baru untuk batch (1, 128, 128, 3) jika diperlukan
    img_array = np.expand_dims(img_array, axis=0)

    # Lakukan prediksi menggunakan model
    prediction = model.predict(img_array)

    # Ambil kelas prediksi dan label
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = categories[predicted_class[0]]

    return predicted_label


# Streamlit app
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf to predict its disease.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load and predict uploaded image
    img = load_img(uploaded_file, target_size=(128, 128))
    prediction = predict_image(img)
    st.write(f"Prediction: **{prediction}**")

# Camera input
st.write("Or use your camera to take a photo:")


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.image = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.image = img
        return img


ctx = webrtc_streamer(
    key="example", video_transformer_factory=VideoTransformer)

if ctx.video_transformer:
    if ctx.video_transformer.image is not None:
        st.write("Captured image successfully.")
        st.image(ctx.video_transformer.image,
                 caption='Captured Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict_image_array(ctx.video_transformer.image)
        st.write(f"Prediction: **{prediction}**")
    else:
        st.write("No image captured.")
