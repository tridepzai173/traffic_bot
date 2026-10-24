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
    return load_model("speed_sign_model_csi22.h5")

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

IMG_SIZE = 224

uploaded_file = st.file_uploader("Chọn ảnh biển báo tốc độ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn upload", use_container_width=True)
    
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    with st.spinner("Đang nhận diện..."):
        pred = model.predict(img, verbose=0)[0]
        class_id = np.argmax(pred)
        confidence = np.max(pred) * 100
        speed = list(SPEED_CLASSES.values())[class_id]
    
    st.success(f"**Kết quả nhận diện:** {speed}")
    st.info(f"**Độ tin cậy:** {confidence:.1f}%")
    
    if "Hết giới hạn" in speed:
        st.warning("Bạn không còn bị giới hạn tốc độ 80 km/h nữa.")
    else:
        st.markdown(f"### → Tài xế được phép di chuyển với tốc độ tối đa **{speed}**")
else:
    st.info("Hãy upload ảnh biển báo tốc độ để bắt đầu")