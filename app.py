import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import urllib.request

# 你在 Google Drive 的權重檔 myna.weights.h5
WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1gMUXwi20DtmtBU2snacKQNDlrdX8LJV1"
WEIGHTS_PATH = "myna.weights.h5"


def download_weights():
    """如果本地還沒有權重，就從 Google Drive 下載一次"""
    if not os.path.exists(WEIGHTS_PATH):
        st.info("首次使用，正在下載模型權重，請稍候...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        st.success("權重下載完成！")


def build_model():
    """重建和你在 Colab 一樣的模型架構：ResNet50V2 + Dense(3, softmax)"""
    resnet = ResNet50V2(include_top=False, pooling="avg")
    resnet.trainable = False  # 遷移學習：凍結 feature extractor

    model = Sequential()
    model.add(resnet)
    model.add(Dense(3, activation="softmax"))  # 三類：土八哥、白尾八哥、家八哥
    return model


@st.cache_resource
def load_model():
    """下載權重 + 建立模型 + 載入權重"""
    download_weights()
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    return model


model = load_model()
labels = ["土八哥", "白尾八哥", "家八哥"]

st.title("八哥辨識器（AIGC Demo）")
st.write("請上傳一張八哥照片，AI 會回傳三種八哥的機率。")

uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 讀圖、轉 RGB、縮成 224x224
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="上傳的圖片", use_column_width=True)

    x = np.array(image)
    x = x.reshape((1, 224, 224, 3))
    x = preprocess_input(x)

    p
