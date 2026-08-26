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
# CUSTOM CSS - WARM THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Base - Warm background */
    .stApp {
        background: #f6efe8;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Brand */
    .brand {
        font-size: 28px;
        font-weight: 700;
        color: #3d2c1e;
        letter-spacing: -0.3px;
        font-family: 'Inter', sans-serif;
    }
    
    .brand-sub {
        font-size: 11px;
        color: #9a8776;
        font-weight: 500;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    /* Cards - Warm */
    .card-warm {
        background: #faf5ef;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e8ddd0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .card-white {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e8ddd0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    /* Metric Cards - Warm */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #b8956e;
        border: 1px solid #e8ddd0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #3d2c1e;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #9a8776;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    
    /* Detection Items */
    .detection-item {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        background: #faf5ef;
        border-radius: 8px;
        margin-bottom: 6px;
        border: 1px solid #e8ddd0;
    }
    
    .confidence-bar {
        height: 5px;
        border-radius: 4px;
        background: #e8ddd0;
        overflow: hidden;
        flex: 1;
        margin: 0 12px;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: #b8956e;
    }
    
    /* Badges - Warm */
    .badge-high {
        background: #f5e6e0;
        color: #a65a4a;
        padding: 2px 12px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    .badge-medium {
        background: #f5ede0;
        color: #9a7a4a;
        padding: 2px 12px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    /* Sidebar - Warm */
    .sidebar-section {
        background: #ffffff;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border: 1px solid #e8ddd0;
    }
    
    .sidebar-section h4 {
        font-size: 11px;
        font-weight: 600;
        color: #9a8776;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }
    
    /* Buttons - Warm */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        background: #b8956e !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: #a07d58 !important;
        box-shadow: 0 2px 8px rgba(184, 149, 110, 0.3) !important;
    }
    
    /* Upload */
    .upload-area {
        border: 2px dashed #d5c8b8;
        border-radius: 12px;
        padding: 32px 16px;
        text-align: center;
        background: #faf5ef;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px 0 6px 0;
        color: #b8a898;
        font-size: 12px;
        border-top: 1px solid #e8ddd0;
        margin-top: 28px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #3d2c1e !important;
    }
    
    /* Section */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #3d2c1e;
        margin-bottom: 6px;
    }
    
    .section-sub {
        font-size: 14px;
        color: #9a8776;
        margin-bottom: 16px;
    }
    
    /* Image container - fixed size */
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        background: #faf5ef;
        border: 1px solid #e8ddd0;
    }
    
    .image-container img {
        width: 100%;
        max-height: 450px;
        object-fit: contain;
        display: block;
    }
    
    /* Detection results image - same size */
    .result-container {
        border-radius: 10px;
        overflow: hidden;
        background: #faf5ef;
        border: 1px solid #e8ddd0;
    }
    
    .result-container img {
        width: 100%;
        max-height: 450px;
        object-fit: contain;
        display: block;
    }
    
    /* Uploaded image thumbnail */
    .thumb-container {
        border-radius: 10px;
        overflow: hidden;
        background: #faf5ef;
        border: 1px solid #e8ddd0;
    }
    
    .thumb-container img {
        width: 100%;
        max-height: 400px;
        object-fit: contain;
        display: block;
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
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Severity Guide</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 13px; line-height: 2.2; color: #3d2c1e;">
        <span class="badge-high">High</span> &nbsp; Pothole · Alligator Crack<br>
        <span class="badge-medium">Medium</span> &nbsp; Longitudinal · Transverse
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>Model</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 13px; color: #6a5a4a; line-height: 1.8;">
        <span style="color: #9a8776;">Architecture</span>  YOLOv8s<br>
        <span style="color: #9a8776;">Dataset</span>  RDD2022<br>
        <span style="color: #9a8776;">Classes</span>  4<br>
        <span style="color: #9a8776;">mAP50</span>  0.574
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption("v1.0")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown("""
<div style="margin-bottom: 20px;">
    <div style="font-size: 22px; font-weight: 700; color: #3d2c1e; font-family: 'Inter', sans-serif;">
        Road Damage Detection
    </div>
    <div style="font-size: 14px; color: #9a8776; margin-top: 2px; font-family: 'Inter', sans-serif;">
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
        
        # ============================================================
        # Desktop Layout: Left = Upload + Button, Right = Result
        # ============================================================
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            # Uploaded Image
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: 600; color: #3d2c1e; margin-bottom: 6px;">
                Uploaded Image
                <span style="font-weight: 400; color: #b8a898; font-size: 12px;">
                    ({img_width} × {img_height})
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Small image preview
            st.image(image, use_container_width=True)
            
            # Detect Button
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            detect_btn = st.button("Detect Damage", use_container_width=True)
            
            st.markdown("""
            <div style="font-size: 12px; color: #b8a898; text-align: center; margin-top: 6px;">
                The model will analyze the image and highlight detected damages
            </div>
            """, unsafe_allow_html=True)
        
        with col_right:
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
                            
                            # Metric
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div class="metric-value">{num_detections}</div>
                                        <div class="metric-label">Damages Detected</div>
                                    </div>
                                    <div style="font-size: 22px; color: #b8956e; font-weight: 300;">✓</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detected Image - Same size as uploaded
                            annotated = results[0].plot()
                            st.image(annotated, caption="Detection Results", use_container_width=True)
                            
                            # Details expander
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
                                            <span style="font-weight: 500; font-size: 13px; min-width: 28px; color: #9a8776;">{i+1}</span>
                                            <span style="font-weight: 500; font-size: 14px; min-width: 160px; color: #3d2c1e;">{class_name}</span>
                                            <span class="{badge_class}">{severity}</span>
                                            <div class="confidence-bar">
                                                <div class="confidence-fill" style="width: {conf_pct:.0f}%;"></div>
                                            </div>
                                            <span style="font-weight: 600; font-size: 14px; min-width: 52px; text-align: right; color: #3d2c1e;">{conf_pct:.1f}%</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="metric-card" style="border-left-color: #b8a898;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <div class="metric-value">0</div>
                                        <div class="metric-label">Damages Detected</div>
                                    </div>
                                    <div style="font-size: 22px; color: #8a7a6a; font-weight: 300;">—</div>
                                </div>
                                <div style="color: #7a9a7a; font-weight: 500; margin-top: 8px; font-size: 14px;">
                                    No damage detected — Road looks clear
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                # Placeholder when no detection yet
                st.markdown("""
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; min-height: 350px; flex-direction: column; gap: 12px; background: #faf5ef; border-radius: 10px; border: 1px dashed #d5c8b8;">
                    <div style="font-size: 48px; color: #d5c8b8; font-weight: 300; opacity: 0.5;">◻</div>
                    <div style="font-size: 16px; color: #b8a898; font-weight: 400;">
                        Click "Detect Damage" to see results
                    </div>
                    <div style="font-size: 13px; color: #d5c8b8;">
                        Detection results will appear here
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: VIDEO DETECTION
# ============================================================================
with tab2:
    st.markdown("""
    <div style="background: #faf5ef; border-radius: 10px; padding: 24px; border: 1px solid #e8ddd0; margin-bottom: 16px;">
        <div style="font-size: 14px; color: #6a5a4a;">
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
        <div class="metric-card" style="border-left-color: #b8956e;">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #a65a4a;">
            <div class="metric-label">High Priority</div>
            <div class="metric-value">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #9a7a4a;">
            <div class="metric-label">Medium Priority</div>
            <div class="metric-value">0</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #faf5ef; border-radius: 10px; padding: 20px; border: 1px solid #e8ddd0; margin-top: 12px;">
        <div style="font-size: 14px; color: #9a8776;">
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