import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("myModel.keras")

# Define your flower classes (update with your actual labels)
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((128, 128))  # adjust size based on your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if model was trained with normalization
    return img_array

# Set page config with a pink theme
st.set_page_config(page_title="Flower Classifier 🌸", page_icon="🌷", layout="centered")

# Custom CSS for girly pastel vibes
st.markdown("""
    <style>
    .main {
        background-color: #fff0f6;
    }
    .stButton>button {
        background-color: #ffb6c1;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 0.5em 1.5em;
    }
    .stButton>button:hover {
        background-color: #ff69b4;
        color: white;
    }
    .stSuccess {
        background-color: #ffe4e1;
        color: #c71585;
        border-radius: 12px;
        padding: 10px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("💐✨ Flower Classification ✨💐")
st.write("Upload your pretty flower photo and I’ll guess what it is 🌸🌼🌷🌻🌹")

# Upload image
uploaded_file = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="💖 Your Uploaded Flower 💖", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.success(f"🌸 Prediction: **{predicted_class.capitalize()}** 💕 \n✨ Confidence: **{confidence:.2f}** ✨")

# (Removed invalid shell command)
