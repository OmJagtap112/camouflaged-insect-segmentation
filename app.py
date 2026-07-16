import streamlit as st
from PIL import Image
import cv2

from predict import SINetPredictor

st.set_page_config(
    page_title="Camouflaged Insect Segmentation",
    layout="wide"
)

st.title("Camouflaged Insect Segmentation using SINet")

st.write("Upload an insect image.")

predictor = SINetPredictor()

uploaded = st.file_uploader(
    "Choose an Image",
    type=["jpg","jpeg","png"]  
)

if uploaded is not None:

    image = Image.open(uploaded)

    original,mask,overlay = predictor.predict(image)

    col1,col2,col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.image(original,use_container_width=True)

    with col2:
        st.subheader("Segmentation Mask")
        st.image(mask,use_container_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay,use_container_width=True)

    mask_bgr=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    _,buffer=cv2.imencode(".png",mask_bgr)

    st.download_button(
        "Download Mask",
        buffer.tobytes(),
        "mask.png",
        "image/png"
    )