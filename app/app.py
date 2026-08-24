import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import torch
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="RoadVision AI",
    page_icon="",
    layout="wide"
)

# Title
st.title("RoadVision AI")
st.subheader("Intelligent Road Damage Detection System")

# Load model
@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")
    if model_path.exists():
        return YOLO(str(model_path))
    else:
        st.error("Model not found! Please train the model first.")
        return None

model = load_model()

# Sidebar
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.divider()
    
    st.header("Damage Severity")
    st.info("""
    - **High:** Pothole, Alligator Crack
    - **Medium:** Longitudinal Crack, Transverse Crack
    """)

# Main area
tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

with tab1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file is not None and model is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Detect Damage", type="primary"):
            with st.spinner("Analyzing image..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name
                
                # Run inference
                results = model.predict(
                    tmp_path,
                    conf=confidence,
                    save=False
                )
                
                # Process results
                if results and len(results[0].boxes) > 0:
                    # Get detections
                    boxes = results[0].boxes
                    
                    # Display results
                    st.success(f"Found {len(boxes)} damage(s)")
                    
                    # Show annotated image
                    annotated = results[0].plot()
                    st.image(annotated, caption="Detected Damage", use_container_width=True)
                    
                    # Show details
                    with st.expander("Detection Details"):
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls]
                            st.write(f"{i+1}. **{class_name}** - Confidence: {conf:.2%}")
                else:
                    st.info("✅ No damage detected")
                
                # Cleanup
                os.unlink(tmp_path)

with tab2:
    st.header("Upload Video")
    uploaded_video = st.file_uploader(
        "Choose a video...",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    if uploaded_video is not None and model is not None:
        st.video(uploaded_video)
        
        if st.button("🎬 Detect Damage in Video", type="primary"):
            st.info("Video detection coming soon...")

# Footer
st.divider()
st.caption("Built with YOLOv8 + Streamlit | RoadVision AI")