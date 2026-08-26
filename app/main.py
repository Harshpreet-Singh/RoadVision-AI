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

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="RoadVision AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Base */
    .stApp {
        background: #f1f4f9;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Brand */
    .brand {
        font-size: 28px;
        font-weight: 700;
        color: #1a2634;
        letter-spacing: -0.3px;
        font-family: 'Inter', sans-serif;
    }
    
    .brand-sub {
        font-size: 11px;
        color: #7a8a9e;
        font-weight: 500;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    /* Cards */
    .card-light {
        background: #f7f9fc;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e4e9f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .card-white {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e4e9f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4a6fa5;
        border: 1px solid #e4e9f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a2634;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #7a8a9e;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    
    /* Detection Items */
    .detection-item {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        background: #f7f9fc;
        border-radius: 8px;
        margin-bottom: 6px;
        border: 1px solid #e8ecf3;
    }
    
    .confidence-bar {
        height: 5px;
        border-radius: 4px;
        background: #e4e9f0;
        overflow: hidden;
        flex: 1;
        margin: 0 12px;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: #4a6fa5;
    }
    
    /* Badges */
    .badge-high {
        background: #fce8e6;
        color: #b33c34;
        padding: 2px 12px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    .badge-medium {
        background: #fef3e0;
        color: #a06b2b;
        padding: 2px 12px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: #ffffff;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border: 1px solid #e4e9f0;
    }
    
    .sidebar-section h4 {
        font-size: 11px;
        font-weight: 600;
        color: #7a8a9e;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        background: #2d3e50 !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        transition: all 0.15s ease !important;
    }
    
    .stButton > button:hover {
        background: #1e2d3b !important;
        box-shadow: 0 2px 8px rgba(45, 62, 80, 0.25) !important;
    }
    
    /* Upload */
    .upload-area {
        border: 2px dashed #d0d7e2;
        border-radius: 12px;
        padding: 32px 16px;
        text-align: center;
        background: #f7f9fc;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px 0 6px 0;
        color: #9aabbe;
        font-size: 12px;
        border-top: 1px solid #e4e9f0;
        margin-top: 28px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #1a2634 !important;
    }
    
    /* Spacing */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1a2634;
        margin-bottom: 6px;
    }
    
    .section-sub {
        font-size: 14px;
        color: #7a8a9e;
        margin-bottom: 16px;
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
    return None

model = load_model()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 12px 0;">
        <div class="brand">RoadVision</div>
        <div class="brand-sub">AI · Computer Vision</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Settings</h4>', unsafe_allow_html=True)
    
    confidence = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25, 0.05,
        help="Higher value reduces false positives"
    )
    
    iou = st.slider(
        "IoU Threshold",
        0.0, 1.0, 0.45, 0.05,
        help="Controls overlap between detections"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Severity Guide
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Severity Guide</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 13px; line-height: 2.2; color: #1a2634;">
        <span class="badge-high">High</span> &nbsp; Pothole · Alligator Crack<br>
        <span class="badge-medium">Medium</span> &nbsp; Longitudinal · Transverse
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Info
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Model</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 13px; color: #3d5068; line-height: 1.8;">
        <span style="color: #7a8a9e;">Architecture</span>  YOLOv8s<br>
        <span style="color: #7a8a9e;">Dataset</span>  RDD2022<br>
        <span style="color: #7a8a9e;">Classes</span>  4<br>
        <span style="color: #7a8a9e;">mAP50</span>  0.574
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption("v1.0")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown("""
<div style="margin-bottom: 20px;">
    <div style="font-size: 22px; font-weight: 700; color: #1a2634; font-family: 'Inter', sans-serif;">
        Road Damage Detection
    </div>
    <div style="font-size: 14px; color: #7a8a9e; margin-top: 2px; font-family: 'Inter', sans-serif;">
        Upload an image to detect potholes, cracks, and road defects
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Dashboard"])

# ============================================================================
# TAB 1: IMAGE DETECTION
# ============================================================================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        img_width, img_height = image.size
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: 600; color: #1a2634; margin-bottom: 6px;">
                Uploaded Image
                <span style="font-weight: 400; color: #9aabbe; font-size: 12px;">
                    ({img_width} × {img_height})
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 180px; gap: 12px;">
            """, unsafe_allow_html=True)
            
            detect_btn = st.button("Detect Damage", use_container_width=True)
            
            st.markdown("""
                <div style="font-size: 13px; color: #9aabbe; text-align: center;">
                    The model will analyze the image and highlight detected damages
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if detect_btn:
            if model is None:
                st.error("Model not found. Please train first.")
            else:
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
                        <div class="metric-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div class="metric-value">{num_detections}</div>
                                    <div class="metric-label">Damages Detected</div>
                                </div>
                                <div style="font-size: 24px; color: #4a6fa5; font-weight: 300;">✓</div>
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
                                    <div style="display: flex; align-items: center; gap: 14px; flex: 1;">
                                        <span style="font-weight: 500; font-size: 13px; min-width: 28px; color: #7a8a9e;">{i+1}</span>
                                        <span style="font-weight: 500; font-size: 14px; min-width: 160px; color: #1a2634;">{class_name}</span>
                                        <span class="{badge_class}">{severity}</span>
                                        <div class="confidence-bar">
                                            <div class="confidence-fill" style="width: {conf_pct:.0f}%;"></div>
                                        </div>
                                        <span style="font-weight: 600; font-size: 14px; min-width: 52px; text-align: right; color: #1a2634;">{conf_pct:.1f}%</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card" style="border-left-color: #7a8a9e;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div class="metric-value">0</div>
                                    <div class="metric-label">Damages Detected</div>
                                </div>
                                <div style="font-size: 24px; color: #6b8a7e; font-weight: 300;">—</div>
                            </div>
                            <div style="color: #4f7a6a; font-weight: 500; margin-top: 8px; font-size: 14px;">
                                No damage detected — Road looks clear
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: VIDEO DETECTION
# ============================================================================
with tab2:
    st.markdown("""
    <div style="background: #f7f9fc; border-radius: 10px; padding: 24px; border: 1px solid #e4e9f0; margin-bottom: 16px;">
        <div style="font-size: 14px; color: #3d5068;">
            Video detection is under development. Upload a video to detect damages frame by frame.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Upload Video",
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #4a6fa5;">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #b33c34;">
            <div class="metric-label">High Priority</div>
            <div class="metric-value">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #a06b2b;">
            <div class="metric-label">Medium Priority</div>
            <div class="metric-value">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #f7f9fc; border-radius: 10px; padding: 20px; border: 1px solid #e4e9f0; margin-top: 12px;">
        <div style="font-size: 14px; color: #7a8a9e;">
            Dashboard will track detection history across sessions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    RoadVision AI · YOLOv8 · Streamlit
</div>
""", unsafe_allow_html=True)