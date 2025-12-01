import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

model = tf.keras.models.load_model("model/myna_model.h5")
labels = ["土八哥", "白尾八哥", "家八哥"]

st.title("八哥辨識器（AIGC Demo）")
uploaded_file = st.file_uploader("請上傳一張八哥照片", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image)
    x = np.array(image)
    x = x.reshape((1,224,224,3))
    x = preprocess_input(x)

    pred = model.predict(x)[0]
    for i in range(3):
        st.write(labels[i], "：", round(float(pred[i]), 4))
