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

# ============================
# 你在 Google Drive 的「myna.weights.h5」權重檔 ID
FILE_ID = "1XjxtxZ8oE9ZDsofuUP0BOv8uadw5sxff"
WEIGHTS_PATH = "myna.weights.h5"
# ============================

def download_weights():
    """如果本地還沒有權重，就從 Google Drive 下載一次"""
    if not os.path.exists(WEIGHTS_PATH):
        st.info("首次使用，正在下載模型權重，請稍候...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)
        st.success("權重下載完成！")

def build_model():
    """重建和你在 Colab 一樣的模型架構：ResNet50V2 + Dense(3, softmax)"""
    resnet = ResNet50V2(include_top=False, pooling="avg")
    resnet.trainable = False  # 遷移學習：凍結 feature extractor

    model = Sequential()
    model.add(resnet)
    model.add(Dense(3, activation="softmax"))  # 三類：土八哥、白尾八哥、家八哥
    r
