import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="RoadVision AI", layout="wide")

st.title("RoadVision AI")
st.subheader("Intelligent Road Damage Detection System")

# Sidebar
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    st.divider()
    st.markdown("**Upload Image or Video**")

# Main area
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Dashboard"])

with tab1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Damage"):
            st.info("Detection in progress...")
            # Yaha model inference call karenge

with tab2:
    st.header("Upload Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

with tab3:
    st.header("Dashboard")
    st.metric("Total Detections", "0")
    st.metric("Potholes Detected", "0")
    st.metric("Cracks Detected", "0")

st.divider()
st.caption("Built with YOLOv8 + Streamlit")