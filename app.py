import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="Chatbot Biển Báo Tốc Độ - MindX CSI22",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Chatbot Nhận diện Biển báo Tốc độ giới hạn")
st.markdown("**Dự án MindX CSI22** - Upload ảnh biển báo để nhận tốc độ được phép di chuyển")

@st.cache_resource
def load_my_model():
    return load_model("speed_sign_cnn.h5")

model = load_my_model()

SPEED_CLASSES = {
    0: "20 km/h",
    1: "30 km/h",
    2: "50 km/h",
    3: "60 km/h",
    4: "70 km/h",
    5: "80 km/h",
    6: "Hết giới hạn 80 km/h",
    7: "100 km/h",
    8: "120 km/h"
}

IMG_SIZE = 32

uploaded_file = st.file_uploader("Chọn ảnh biển báo tốc độ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh bạn upload", use_container_width=True
