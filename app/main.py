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
import base64
from io import BytesIO

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="RoadVision AI - Road Damage Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Import Google Font & Material Icons */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    /* Global */
    .stApp {
        background: #f8fafc;
    }
    
    /* Material Icons inline */
    .material-icons {
        font-family: 'Material Icons';
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
        vertical-align: middle;
    }
    
    .icon-sm { font-size: 18px; }
    .icon-md { font-size: 28px; }
    .icon-lg { font-size: 40px; }
    .icon-xl { font-size: 56px; }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Card */
    .card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid #e9edf4;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }
    
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Detection items */
    .detection-item {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        background: #f8fafc;
        border-radius: 10px;
        margin-bottom: 6px;
        border: 1px solid #e9edf4;
    }
    
    .confidence-bar {
        height: 6px;
        border-radius: 4px;
        background: #e9edf4;
        overflow: hidden;
        flex: 1;
        margin: 0 12px;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        transition: width 0.8s ease;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #e9edf4;
    }
    
    .sidebar-section h4 {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    /* Badges */
    .badge-high {
        background: #fef2f2;
        color: #dc2626;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-medium {
        background: #fffbeb;
        color: #d97706;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 24px 0 8px 0;
        color: #94a3b8;
        font-size: 13px;
        border-top: 1px solid #e9edf4;
        margin-top: 32px;
    }
    
    /* Image container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        background: #f1f5f9;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .image-container img {
        width: 100%;
        height: auto;
        display: block;
    }
    
    /* Icon with text */
    .icon-text {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .icon-text .material-icons {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")
    if model_path.exists():
        return YOLO(str(model_path))
    else:
        st.error("Model not found. Please train the model first.")
        return None

model = load_model()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="text-align: center; padding: 8px 0 16px 0;">
        <div style="font-size: 36px; font-weight: 800; font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            RoadVision
        </div>
        <div style="font-size: 12px; color: #64748b; font-weight: 500; letter-spacing: 0.5px; margin-top: -4px;">
            AI · Computer Vision
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Settings</h4>', unsafe_allow_html=True)
    
    confidence = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25, 0.05,
        help="Higher value = fewer but more accurate detections"
    )
    
    iou = st.slider(
        "IoU Threshold",
        0.0, 1.0, 0.45, 0.05,
        help="Controls overlap between bounding boxes"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Severity Guide
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Severity Guide</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 6px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span class="badge-high">High</span>
            <span style="font-size: 14px; color: #0f172a;">Pothole · Alligator Crack</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span class="badge-medium">Medium</span>
            <span style="font-size: 14px; color: #0f172a;">Longitudinal · Transverse</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Info
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Model Info</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 13px; color: #475569; line-height: 1.6;">
        <div><strong>Architecture:</strong> YOLOv8s</div>
        <div><strong>Dataset:</strong> RDD2022</div>
        <div><strong>Classes:</strong> 4</div>
        <div><strong>mAP50:</strong> 0.574</div>
        <div><strong>Inference:</strong> ~9.6ms/img</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption("v1.0 · Built with Streamlit")

# ============================================================================
# MAIN CONTENT
# ============================================================================
# Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
    <div>
        <h1 style="font-size: 26px; font-weight: 700; color: #0f172a; margin: 0;">
            <span class="material-icons" style="font-size: 28px; vertical-align: middle;">road</span>
            Road Damage Detection
        </h1>
        <p style="color: #64748b; font-size: 15px; margin: 4px 0 0 0;">
            Upload an image to detect potholes, cracks, and road defects
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs with icons
tab1, tab2, tab3 = st.tabs([
    "Image Detection",
    "Video Detection",
    "Dashboard"
])

# ============================================================================
# TAB 1: IMAGE DETECTION
# ============================================================================
with tab1:
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        img_width, img_height = image.size
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: 600; color: #0f172a; margin-bottom: 8px;">
                <span class="material-icons" style="font-size: 18px; vertical-align: middle;">image</span>
                Uploaded Image
                <span style="font-weight: 400; color: #94a3b8; font-size: 12px;">
                    ({img_width} x {img_height})
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; min-height: 200px; flex-direction: column; gap: 16px;">
            """, unsafe_allow_html=True)
            
            detect_btn = st.button(
                "Detect Damage",
                type="primary",
                use_container_width=True
            )
            
            st.markdown("""
                <p style="font-size: 13px; color: #94a3b8; text-align: center; margin: 0;">
                    Model will analyze the image and highlight detected damages
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if detect_btn:
            with st.spinner("Analyzing image..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name, format='JPEG', quality=95)
                    tmp_path = tmp.name
                
                results = model.predict(
                    tmp_path,
                    conf=confidence,
                    iou=iou,
                    save=False,
                    verbose=False
                )
                
                os.unlink(tmp_path)
                
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    num_detections = len(boxes)
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: #22c55e;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="metric-value">{num_detections}</div>
                                <div class="metric-label">Damages Detected</div>
                            </div>
                            <div style="font-size: 28px; color: #22c55e;">
                                <span class="material-icons">check_circle</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    annotated = results[0].plot()
                    st.image(annotated, caption="Detection Results", use_container_width=True)
                    
                    with st.expander("Detection Details", expanded=True):
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls]
                            
                            severity = "High" if class_name in ["Pothole", "Alligator Crack"] else "Medium"
                            badge_class = "badge-high" if severity == "High" else "badge-medium"
                            
                            conf_pct = conf * 100
                            
                            st.markdown(f"""
                            <div class="detection-item">
                                <div style="display: flex; align-items: center; gap: 12px; flex: 1;">
                                    <span style="font-weight: 600; font-size: 14px; min-width: 28px;">{i+1}</span>
                                    <span style="font-weight: 500; font-size: 14px; min-width: 160px;">{class_name}</span>
                                    <span class="{badge_class}" style="font-size: 11px;">{severity}</span>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: {conf_pct:.0f}%;"></div>
                                    </div>
                                    <span style="font-weight: 600; font-size: 14px; min-width: 56px; text-align: right; color: #0f172a;">{conf_pct:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: #3b82f6;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="metric-value">0</div>
                                <div class="metric-label">Damages Detected</div>
                            </div>
                            <div style="font-size: 28px; color: #22c55e;">
                                <span class="material-icons">check_circle</span>
                            </div>
                        </div>
                        <p style="color: #22c55e; font-weight: 500; margin-top: 8px;">No damage detected — Road looks clear</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: VIDEO DETECTION
# ============================================================================
with tab2:
    st.info("Video detection coming soon. Upload a video to detect damages frame by frame.")
    
    uploaded_video = st.file_uploader(
        "Choose a video...",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed"
    )
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("Process Video", type="primary"):
            st.warning("Video processing feature is under development.")

# ============================================================================
# TAB 3: DASHBOARD
# ============================================================================
with tab3:
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px;">
        <div class="metric-card" style="border-left-color: #3b82f6;">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value">0</div>
        </div>
        <div class="metric-card" style="border-left-color: #ef4444;">
            <div class="metric-label">High Priority</div>
            <div class="metric-value">0</div>
        </div>
        <div class="metric-card" style="border-left-color: #d97706;">
            <div class="metric-label">Medium Priority</div>
            <div class="metric-value">0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Dashboard will track detection history across sessions (coming soon).")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    Built with YOLOv8 · Streamlit · PyTorch &nbsp;|&nbsp; RoadVision AI v1.0
</div>
""", unsafe_allow_html=True)