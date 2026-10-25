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
st.markdown("**Dự án MindX CSI22**")

@st.cache_resource
def load_my_model():
    model = load_model("speed_sign_cnn.h5", compile=False)
    return model

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
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh bạn upload", use_container_width=True)

        img = np.array(image)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)
        class_id = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]) * 100)
        speed = list(SPEED_CLASSES.values())[class_id]

        st.success(f"**Kết quả nhận diện:** {speed}")
        st.info(f"**Độ tin cậy:** {confidence:.1f}%")

        if "Hết giới hạn" in speed:
            st.warning("Bạn không còn bị giới hạn tốc độ 80 km/h nữa.")
        else:
            st.markdown(f"### → Tài xế được phép di chuyển với tốc độ tối đa **{speed}**")

    except Exception as e:
        st.error(f"Lỗi khi nhận diện: {e}")
else:
    st.info("Hãy upload ảnh biển báo tốc độ để bắt đầu")
