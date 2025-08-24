import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load YOLOv11n model
model = YOLO("yolo11n.pt")  # Ensure this file is in your folder

st.title("YOLOv11n Object Detection")

option = st.radio("Choose input method:", ("Open Camera", "Upload Image"))

img = None
if option == "Open Camera":
    img_file = st.camera_input("Take a picture")
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, caption="Your photo", use_column_width=True)
elif option == "Upload Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img, caption="Your uploaded image", use_column_width=True)

if img is not None:
    # Run YOLOv11n prediction
    results = model(img)
    # Plot results on image
    res_img = results[0].plot()  # returns a numpy array
    st.image(res_img, caption="YOLOv11n Prediction", use_column_width=True)
