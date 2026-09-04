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
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - WARM THEME & CLEAN UI
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Base - Warm background */
    .stApp {
        background: #f6efe8;
        font-family: 'Inter', sans-serif;
    }
    
    /* Brand */
    .brand {
        font-size: 26px;
        font-weight: 800;
        color: #3d2c1e;
        letter-spacing: -0.5px;
        font-family: 'Inter', sans-serif;
    }
    
    .brand-sub {
        font-size: 11px;
        color: #9a8776;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    /* UI Cards - Clean & Warm */
    .ui-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e8ddd0;
        box-shadow: 0 2px 8px rgba(61, 44, 30, 0.04);
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .card-header {
        font-size: 14px;
        font-weight: 700;
        color: #3d2c1e;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f0e6dc;
        padding-bottom: 10px;
    }
    
    .resolution-badge {
        font-size: 11px;
        font-weight: 500;
        color: #9a8776;
        background: #f6efe8;
        padding: 4px 10px;
        border-radius: 6px;
    }
    
    .card-body {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }
    
    /* Metric Card */
    .metric-card {
        background: #faf5ef;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #b8956e;
        border: 1px solid #e8ddd0;
        margin-bottom: 16px;
        width: 100%;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #3d2c1e;
        font-family: 'Inter', sans-serif;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        color: #9a8776;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }
    
    /* Detection Items */
    .detection-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background: #ffffff;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #e8ddd0;
        transition: transform 0.15s ease;
    }
    
    .detection-item:hover {
        transform: translateX(2px);
        border-color: #b8956e;
    }
    
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: #f0e6dc;
        overflow: hidden;
        flex: 1;
        margin: 0 16px;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #b8956e, #a07d58);
    }
    
    /* Badges */
    .badge-high {
        background: #f5e6e0;
        color: #a65a4a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
        white-space: nowrap;
    }
    
    .badge-medium {
        background: #f5ede0;
        color: #9a7a4a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
        white-space: nowrap;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #e8ddd0;
    }
    
    .sidebar-section h4 {
        font-size: 11px;
        font-weight: 700;
        color: #9a8776;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
        margin-top: 0;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border-radius: 10px !important;
        background: #3d2c1e !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.7rem 1.8rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        box-shadow: 0 2px 4px rgba(61, 44, 30, 0.1) !important;
    }
    
    .stButton > button:hover {
        background: #b8956e !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(184, 149, 110, 0.3) !important;
    }
    
    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 280px;
        background: #faf5ef;
        border-radius: 10px;
        border: 2px dashed #e8ddd0;
        padding: 24px;
        text-align: center;
        width: 100%;
    }
    
    .empty-state .icon {
        font-size: 40px;
        color: #d5c8b8;
        margin-bottom: 12px;
    }
    
    .empty-state .title {
        font-size: 15px;
        color: #3d2c1e;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .empty-state .sub {
        font-size: 13px;
        color: #9a8776;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #3d2c1e !important;
        font-size: 14px !important;
    }
    
    /* Image styling override for consistency */
    .stImage img {
        border-radius: 8px;
        border: 1px solid #f0e6dc;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 24px 0 12px 0;
        color: #b8a898;
        font-size: 12px;
        font-weight: 500;
        border-top: 1px solid #e8ddd0;
        margin-top: 40px;
        font-family: 'Inter', sans-serif;
    }

    /* Mobile Responsive Stacking Guarantee */
    @media (max-width: 768px) {
        .stColumn {
            width: 100% !important;
            max-width: 100% !important;
            flex: 0 0 100% !important;
            margin-bottom: 16px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "models" / "best.pt"
    
    if model_path.exists():
        try:
            return YOLO(str(model_path))
        except Exception as e:
            st.error(f"Model load error: {e}")
            return None
    else:
        st.error("Model not found at: " + str(model_path))
        return None

model = load_model()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div class="brand">RoadVision</div>
        <div class="brand-sub">AI · Computer Vision</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>⚙️ Settings</h4>', unsafe_allow_html=True)
    
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
    st.markdown('<h4>📊 Severity Guide</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 13px; line-height: 2.4; color: #3d2c1e;">
        <span class="badge-high">High</span> &nbsp; Pothole · Alligator Crack<br>
        <span class="badge-medium">Medium</span> &nbsp; Longitudinal · Transverse
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Info
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<h4>🧠 Model Info</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 13px; color: #6a5a4a; line-height: 1.8;">
        <span style="color: #9a8776; font-weight: 600;">Architecture</span> &nbsp; YOLOv8s<br>
        <span style="color: #9a8776; font-weight: 600;">Dataset</span> &nbsp; RDD2022<br>
        <span style="color: #9a8776; font-weight: 600;">Classes</span> &nbsp; 4<br>
        <span style="color: #9a8776; font-weight: 600;">mAP50</span> &nbsp; 0.574
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption("v1.0 · Built with Streamlit")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown("""
<div style="margin-bottom: 24px;">
    <div style="font-size: 26px; font-weight: 800; color: #3d2c1e; font-family: 'Inter', sans-serif; letter-spacing: -0.5px;">
        Road Damage Detection
    </div>
    <div style="font-size: 15px; color: #9a8776; margin-top: 6px; font-family: 'Inter', sans-serif; font-weight: 500;">
        Upload an image to detect potholes, cracks, and road defects instantly.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📊 Dashboard"])

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
        # --- LOGIC UNCHANGED ---
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        img_width, img_height = image.size
        
        # Side-by-side layout (natively stacks on mobile)
        col_left, col_right = st.columns(2, gap="large")
        
        with col_left:
            st.markdown(f"""
            <div class="ui-card">
                <div class="card-header">
                    <span>📷 Uploaded Image</span>
                    <span class="resolution-badge">{img_width} × {img_height}</span>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)
            
            st.image(image, use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            
            detect_btn = st.button("🔍 Detect Damage", type="primary", use_container_width=True)
            
            st.markdown("""
            <div style="font-size: 12px; color: #9a8776; text-align: center; margin-top: 10px; font-weight: 500;">
                Model analyzes the image and highlights detected damages
            </div>
            """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("""
            <div class="ui-card">
                <div class="card-header">
                    <span>🎯 Detection Results</span>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)
            
            # --- LOGIC UNCHANGED ---
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
                                    <div style="font-size: 24px; color: #b8956e; font-weight: 300;">✓</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('<div style="width: 100%; margin-bottom: 8px; font-size: 13px; font-weight: 600; color: #6a5a4a;">Annotated Output</div>', unsafe_allow_html=True)
                            annotated = results[0].plot()
                            st.image(annotated, use_container_width=True)
                            
                            st.markdown('</div></div>', unsafe_allow_html=True) # Close card-body and ui-card
                            
                            st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
                            
                            with st.expander("📋 Detection Details", expanded=True):
                                for i, box in enumerate(boxes):
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = model.names[cls]
                                    
                                    severity = "High" if class_name in ["Pothole", "Alligator Crack"] else "Medium"
                                    badge_class = "badge-high" if severity == "High" else "badge-medium"
                                    
                                    conf_pct = conf * 100
                                    
                                    st.markdown(f"""
                                    <div class="detection-item">
                                        <div style="display: flex; align-items: center; gap: 12px; flex: 1; width: 100%;">
                                            <span style="font-weight: 700; font-size: 13px; min-width: 24px; color: #b8956e;">#{i+1}</span>
                                            <span style="font-weight: 600; font-size: 14px; min-width: 140px; color: #3d2c1e;">{class_name}</span>
                                            <span class="{badge_class}">{severity}</span>
                                            <div class="confidence-bar">
                                                <div class="confidence-fill" style="width: {conf_pct:.0f}%;"></div>
                                            </div>
                                            <span style="font-weight: 700; font-size: 13px; min-width: 48px; text-align: right; color: #3d2c1e;">{conf_pct:.1f}%</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.markdown('</div>', unsafe_allow_html=True) # Close card-body
                            st.markdown("""
                            <div class="metric-card" style="border-left-color: #7a9a7a; text-align: center;">
                                <div style="font-size: 32px; margin-bottom: 8px;">🛣️</div>
                                <div class="metric-value" style="color: #7a9a7a;">0</div>
                                <div class="metric-label">Damages Detected</div>
                                <div style="color: #7a9a7a; font-weight: 600; margin-top: 12px; font-size: 14px;">
                                    No damage detected — Road looks clear
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True) # Close ui-card
            else:
                st.markdown('</div>', unsafe_allow_html=True) # Close card-body
                st.markdown("""
                <div class="empty-state">
                    <div class="icon">🔍</div>
                    <div class="title">Ready to Analyze</div>
                    <div class="sub">Click "Detect Damage" on the left to see results here</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True) # Close ui-card

    else:
        # Initial state when no file is uploaded
        st.markdown("""
        <div class="empty-state" style="min-height: 400px; background: #ffffff; border: 2px dashed #e8ddd0;">
            <div class="icon">📤</div>
            <div class="title" style="font-size: 18px; color: #3d2c1e;">Upload an Image to Begin</div>
            <div class="sub" style="max-width: 400px; margin-top: 8px;">
                Supported formats: JPG, JPEG, PNG, BMP, WEBP. 
                The AI will automatically scan for potholes and cracks.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: VIDEO DETECTION
# ============================================================================
with tab2:
    st.markdown("""
    <div class="ui-card" style="margin-bottom: 20px;">
        <div style="font-size: 15px; color: #6a5a4a; font-weight: 500; text-align: center; padding: 12px;">
            🚧 Video detection is under development. Upload a video to detect damages frame by frame.
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
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        if st.button("Process Video", type="primary", use_container_width=True):
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
    <div class="ui-card" style="margin-top: 12px;">
        <div style="font-size: 14px; color: #9a8776; text-align: center; padding: 20px; font-weight: 500;">
            📈 Dashboard will track detection history, geolocation data, and severity trends across sessions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    RoadVision AI · YOLOv8 · Streamlit · © 2026
</div>
""", unsafe_allow_html=True)