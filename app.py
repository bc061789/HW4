import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import gdown

# Google Drive 上的權重檔 myna.weights.h5
# 這個 ID 就是你剛給的 weights 連結：
# https://drive.google.com/file/d/1XjxtxZ8oE9ZDsofuUP0BOv8uadw5sxff/view?usp=drive_link
FILE_ID = "1XjxtxZ8oE9ZDsofuUP0BOv8uadw5sxff"
WEIGHTS_PATH = "myna.weights.h5"


def download_weights():
    """如果本地還沒有權重，就從 Google Drive 下載一次，並顯示檔案大小"""
    if not os.path.exists(WEIGHTS_PATH):
        st.info("首次使用，正在下載模型權重，請稍候...")
        # 用 gdown 的 id 下載方式，比自己組 URL 穩定
        gdown.download(id=FILE_ID, output=WEIGHTS_PATH, quiet=False)
        if os.path.exists(WEIGHTS_PATH):
            size = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
            st.success(f"權重下載完成，檔案大小約 {size:.2f} MB")
        else:
            st.error("下載失敗：找不到權重檔。")


def build_model():
    """重建和你在 Colab 一樣的模型架構：ResNet50V2 + Dense(3, softmax)"""
    resnet = ResNet50V2(include_top=False, pooling="avg")
    resnet.trainable = False  # 遷移學習：凍結 feature extractor

    model = Sequential()
    model.add(resnet)
    model.add(Dense(3, activation="softmax"))  # 三類：土八哥、白尾八哥、家八哥
    return model


@st.cache_resource
def load_model_safely():
    """下載權重 + 建立模型 + 載入權重，若失敗在畫面上顯示錯誤"""
    download_weights()
    model = build_model()
    try:
        model.load_weights(WEIGHTS_PATH)
    except Exception as e:
        # 在頁面上顯示完整錯誤，避免只看到灰畫面
        st.error("載入權重時發生錯誤：")
        st.exception(e)
        # 為了不中斷 app，回傳一個還沒載入權重的模型
        return model
    return model


st.title("八哥辨識器（AIGC Demo）")
st.write("請上傳一張八哥照片，AI 會回傳三種八哥的機率。")

# 只有當真的要用到模型時才去載入，避免一開始就整個 app 掛掉
if "model" not in st.session_state:
    with st.spinner("模型初始化中，請稍候..."):
        st.session_state["model"] = load_model_safely()

model = st.session_state["model"]
labels = ["土八哥", "白尾八哥", "家八哥"]

uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="上傳的圖片", use_column_width=True)

    x = np.array(image)
    x = x.reshape((1, 224, 224, 3))
    x = preprocess_input(x)

    pred = model.predict(x)[0]

    st.subheader("預測結果（機率）：")
    for i in range(3):
        st.write(f"{labels[i]}：{pred[i]:.4f}")
