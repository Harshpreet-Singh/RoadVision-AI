import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="RoadVision AI",
    page_icon="🛣️",
    layout="wide"
)

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

# Title
st.title("RoadVision AI")
st.subheader("Intelligent Road Damage Detection System")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.divider()
    
    st.header("Severity Guide")
    st.info("""
    - 🟥 **High Priority:** Pothole, Alligator Crack  
    - 🟧 **Medium Priority:** Longitudinal Crack, Transverse Crack
    """)
    
    st.divider()
    st.caption("Model: YOLOv8s | Dataset: RDD2022")

# Tabs
tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

with tab1:
    st.header("Upload Image for Damage Detection")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            detect_btn = st.button("Detect Damage", type="primary", use_container_width=True)
        
        if detect_btn:
            with st.spinner("Analyzing image..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name
                
                results = model.predict(tmp_path, conf=confidence, save=False)
                
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    st.success(f"Found {len(boxes)} damage(s)")
                    
                    annotated = results[0].plot()
                    st.image(annotated, caption="Detection Results", use_container_width=True)
                    
                    with st.expander("Detection Details"):
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls]
                            
                            severity = "High" if class_name in ["Pothole", "Alligator Crack"] else "Medium"
                            emoji = "🔴" if severity == "High" else "🟠"
                            
                            st.write(f"{emoji} {i+1}. **{class_name}** - {conf:.1%} confidence - Severity: **{severity}**")
                else:
                    st.info("No damage detected - Road looks clear!")
                
                os.unlink(tmp_path)

with tab2:
    st.header("Upload Video for Damage Detection")
    uploaded_video = st.file_uploader(
        "Choose a video...",
        type=["mp4", "avi", "mov"]
    )
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🎬 Process Video", type="primary"):
            st.warning("Video detection feature coming soon!")

st.divider()
st.caption("Built with YOLOv8 + Streamlit | RoadVision AI")