import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import os
import urllib.request


MODEL_URL = "https://drive.google.com/uc?export=download&id=1gMUXwi20DtmtBU2snacKQNDlrdX8LJV1"
MODEL_PATH = "myna_model.h5"  

@st.cache_resource
def load_model():
    # 如果本機還沒有模型，就先從雲端下載
    if not os.path.exists(MODEL_PATH):
        st.info("首次使用，正在下載模型，請稍候...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("模型下載完成！")

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
labels = ["土八哥", "白尾八哥", "家八哥"]

st.title("八哥辨識器（AIGC Demo）")
st.write("請上傳一張八哥照片，系統會回傳三種八哥的機率。")

uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 讀圖、轉 RGB、resize
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="上傳的圖片", use_column_width=True)

    x = np.array(image)
    x = x.reshape((1, 224, 224, 3))
    x = preprocess_input(x)

    pred = model.predict(x)[0]

    st.subheader("預測結果 (機率)：")
    for i in range(3):
        st.write(f"{labels[i]}：{pred[i]:.4f}")
