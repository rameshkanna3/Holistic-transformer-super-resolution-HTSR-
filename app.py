import os
from PIL import Image
import streamlit as st

from predict import loadModel, transform, predict

st.set_page_config(page_title="HTSR Demo", page_icon="🖼️", layout="centered", initial_sidebar_state="auto")

upload_path = "uploads/"

banner_image = Image.open("static/main_banner.png")
st.image(banner_image, use_column_width="auto")

st.sidebar.title("Holistic Transformer Super Resolution demo")

inference_device = st.sidebar.selectbox("Pick your inference hardware", ["CPU", "GPU"])

if inference_device is not None:
    model, device = loadModel(inference_device)

upsample_scale = st.sidebar.selectbox("Super Resolution Scale", ["x2", "x4", "x8"])
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])

img_name, image = None, None

if uploaded_image is not None:
    image_path = os.path.join(upload_path, uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write((uploaded_image).getbuffer())
    st.sidebar.success(f"Saved image {uploaded_image.name} successfully!")
    image, img_name = transform(image_path, device)

scol1, scol2, scol3 = st.sidebar.columns(3)

st.subheader("   Upscale Enlarge and enhance images with deeplearning")

col1, _, col3 = st.columns(3)

with col1:
    st.header("LR Image")
    if uploaded_image is not None:
        st.image(uploaded_image)

with col3:
    st.header("SR Image")

if uploaded_image is None:
    infer_button = scol2.button(label="Generate", key="predict", disabled =True)
else:
    infer_button = scol2.button(label="Generate", key="predict", on_click=predict, args=(image, img_name, model, int(upsample_scale[1:]),))

if infer_button:
    with col3:
        st.image(f"sr_out/{uploaded_image.name.split('.')[0]}_HTSR.png")
