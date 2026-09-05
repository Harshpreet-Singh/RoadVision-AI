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
import psycopg2
from datetime import datetime
import time

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
    
    .stApp {
        background: #f6efe8;
        font-family: 'Inter', sans-serif;
    }
    
    .brand-title {
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
    
    .detection-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background: #ffffff;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #e8ddd0;
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
    
    .badge-high {
        background: #f5e6e0;
        color: #a65a4a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }
    
    .badge-medium {
        background: #f5ede0;
        color: #9a7a4a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }
    
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
    
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #3d2c1e !important;
        font-size: 14px !important;
    }
    
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
# DATABASE FUNCTIONS
# ============================================================================
DB_CONFIG = {
    "host": "localhost",
    "database": "roadvision_db",
    "user": "postgres",
    "password": "postgres"  # Apna password
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def save_detection_to_db(image_path, class_name, confidence, severity, lat, lon, device_id="phone"):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO damage_reports 
            (image_path, class_name, confidence, severity, latitude, longitude, device_id, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (image_path, class_name, confidence, severity, lat, lon, device_id, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def get_all_reports(limit=100):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, class_name, confidence, severity, latitude, longitude, timestamp, image_path 
            FROM damage_reports 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

def get_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM damage_reports")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM damage_reports WHERE severity = 'High'")
        high = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM damage_reports WHERE severity = 'Medium'")
        medium = cur.fetchone()[0]
        cur.close()
        conn.close()
        return total, high, medium
    except Exception as e:
        st.error(f"Database error: {e}")
        return 0, 0, 0

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
# SEVERITY HELPER
# ============================================================================
def get_severity(class_name):
    high_classes = ["Pothole", "Alligator Crack"]
    return "High" if class_name in high_classes else "Medium"

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div class="brand-title">RoadVision</div>
        <div class="brand-sub">AI · Computer Vision</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown('<div style="background:#ffffff; border-radius:10px; padding:16px; border:1px solid #e8ddd0; margin-bottom:12px;">', unsafe_allow_html=True)
    st.markdown('<h4 style="font-size:11px; font-weight:700; color:#9a8776; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:12px; margin-top:0;">Settings</h4>', unsafe_allow_html=True)
    
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="background:#ffffff; border-radius:10px; padding:16px; border:1px solid #e8ddd0; margin-bottom:12px;">', unsafe_allow_html=True)
    st.markdown('<h4 style="font-size:11px; font-weight:700; color:#9a8776; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:12px; margin-top:0;">Severity Guide</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size:13px; line-height:2.4; color:#3d2c1e;">
        <span class="badge-high">High</span> Pothole · Alligator Crack<br>
        <span class="badge-medium">Medium</span> Longitudinal · Transverse
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="background:#ffffff; border-radius:10px; padding:16px; border:1px solid #e8ddd0; margin-bottom:12px;">', unsafe_allow_html=True)
    st.markdown('<h4 style="font-size:11px; font-weight:700; color:#9a8776; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:12px; margin-top:0;">Model Info</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size:13px; color:#6a5a4a; line-height:1.8;">
        <span style="color:#9a8776; font-weight:600;">Architecture</span> YOLOv8s<br>
        <span style="color:#9a8776; font-weight:600;">Dataset</span> RDD2022<br>
        <span style="color:#9a8776; font-weight:600;">Classes</span> 4<br>
        <span style="color:#9a8776; font-weight:600;">mAP50</span> 0.574
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption("v2.0 · PostgreSQL + Camera")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown("""
<div style="margin-bottom:24px;">
    <div style="font-size:26px; font-weight:800; color:#3d2c1e; font-family:'Inter', sans-serif; letter-spacing:-0.5px;">
        Road Damage Detection
    </div>
    <div style="font-size:15px; color:#9a8776; margin-top:6px; font-family:'Inter', sans-serif; font-weight:500;">
        Upload image or capture from phone camera. Detections saved to database.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Upload Image", "Phone Camera", "Dashboard", "Reports"])

# ============================================================================
# TAB 1: UPLOAD IMAGE
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
        
        col_left, col_right = st.columns(2, gap="large")
        
        with col_left:
            st.markdown(f"""
            <div class="ui-card">
                <div class="card-header">
                    <span>Uploaded Image</span>
                    <span class="resolution-badge">{img_width} × {img_height}</span>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)
            
            st.image(image, use_container_width=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            
            detect_btn = st.button("Detect Damage", type="primary", use_container_width=True)
        
        with col_right:
            st.markdown("""
            <div class="ui-card">
                <div class="card-header">
                    <span>Detection Results</span>
                </div>
                <div class="card-body">
            """, unsafe_allow_html=True)
            
            if detect_btn:
                if model is None:
                    st.error("Model not found.")
                else:
                    with st.spinner("Analyzing..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            image.save(tmp.name, format='JPEG', quality=95)
                            tmp_path = tmp.name
                        
                        results = model.predict(tmp_path, conf=confidence, iou=iou, save=False, verbose=False)
                        os.unlink(tmp_path)
                        
                        if results and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            num_detections = len(boxes)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div>
                                        <div class="metric-value">{num_detections}</div>
                                        <div class="metric-label">Damages Detected</div>
                                    </div>
                                    <div style="font-size:24px; color:#b8956e; font-weight:300;">✓</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('<div style="width:100%; margin-bottom:8px; font-size:13px; font-weight:600; color:#6a5a4a;">Annotated Output</div>', unsafe_allow_html=True)
                            annotated = results[0].plot()
                            st.image(annotated, use_container_width=True)
                            
                            st.markdown('</div></div>', unsafe_allow_html=True)
                            st.markdown('<div style="margin-top:16px;"></div>', unsafe_allow_html=True)
                            
                            with st.expander("Detection Details", expanded=True):
                                for i, box in enumerate(boxes):
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    class_name = model.names[cls]
                                    severity = get_severity(class_name)
                                    badge_class = "badge-high" if severity == "High" else "badge-medium"
                                    conf_pct = conf * 100
                                    
                                    st.markdown(f"""
                                    <div class="detection-item">
                                        <div style="display:flex; align-items:center; gap:12px; flex:1; width:100%;">
                                            <span style="font-weight:700; font-size:13px; min-width:24px; color:#b8956e;">#{i+1}</span>
                                            <span style="font-weight:600; font-size:14px; min-width:140px; color:#3d2c1e;">{class_name}</span>
                                            <span class="{badge_class}">{severity}</span>
                                            <div class="confidence-bar">
                                                <div class="confidence-fill" style="width:{conf_pct:.0f}%;"></div>
                                            </div>
                                            <span style="font-weight:700; font-size:13px; min-width:48px; text-align:right; color:#3d2c1e;">{conf_pct:.1f}%</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown("""
                            <div class="metric-card" style="border-left-color:#7a9a7a; text-align:center;">
                                <div style="font-size:32px; margin-bottom:8px;">🛣️</div>
                                <div class="metric-value" style="color:#7a9a7a;">0</div>
                                <div class="metric-label">Damages Detected</div>
                                <div style="color:#7a9a7a; font-weight:600; margin-top:12px; font-size:14px;">
                                    No damage detected — Road looks clear
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="empty-state">
                    <div class="icon">🔍</div>
                    <div class="title">Ready to Analyze</div>
                    <div class="sub">Click "Detect Damage" on the left to see results</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state" style="min-height:400px; background:#ffffff; border:2px dashed #e8ddd0;">
            <div class="icon">📤</div>
            <div class="title" style="font-size:18px; color:#3d2c1e;">Upload an Image to Begin</div>
            <div class="sub" style="max-width:400px; margin-top:8px;">
                Supported: JPG, JPEG, PNG, BMP, WEBP. The AI will scan for damage.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: WEBCAM CAPTURE (OpenCV)
# ============================================================================
with tab2:
    st.markdown("""
    <div style="font-size:15px; font-weight:600; color:#3d2c1e; margin-bottom:12px;">
        Capture from Webcam
    </div>
    """, unsafe_allow_html=True)
    
    # Location Input
    st.markdown("""
    <div style="background:#faf5ef; border-radius:10px; padding:16px; border:1px solid #e8ddd0; margin-bottom:16px;">
        <div style="font-size:13px; color:#9a8776; font-weight:500;">
            📍 Location Input
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat = st.number_input("Latitude", value=0.0, format="%.6f", step=0.0001)
    with col_lon:
        lon = st.number_input("Longitude", value=0.0, format="%.6f", step=0.0001)
    
    # Webcam Selection
    st.markdown('<div style="font-size:13px; font-weight:600; color:#3d2c1e; margin-bottom:4px;">Select Webcam</div>', unsafe_allow_html=True)
    camera_index = st.selectbox("", [0, 1, 2, 3], format_func=lambda x: f"Camera {x}", label_visibility="collapsed")
    
    # Capture button
    if st.button("📸 Capture from Webcam", type="primary"):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error(f"Camera {camera_index} not available. Try another index.")
        else:
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                st.image(img, caption=f"Captured from Camera {camera_index}", use_container_width=True)
                
                # Save and process
                with st.spinner("Analyzing and saving..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        img.save(tmp.name, format='JPEG', quality=95)
                        tmp_path = tmp.name
                    
                    results = model.predict(tmp_path, conf=confidence, iou=iou, save=False, verbose=False)
                    
                    if results and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            conf_val = float(box.conf[0])
                            class_name = model.names[cls]
                            severity = get_severity(class_name)
                            
                            saved = save_detection_to_db(
                                tmp_path, class_name, conf_val, severity, lat, lon, device_id="webcam"
                            )
                        
                        if saved:
                            st.success(f"✅ {len(results[0].boxes)} damage(s) saved to database!")
                            st.image(results[0].plot(), caption="Detection Results", use_container_width=True)
                    else:
                        st.info("No damage detected. Nothing saved to database.")
                    
                    os.unlink(tmp_path)
            else:
                st.error(f"Failed to capture from Camera {camera_index}")

# ============================================================================
# TAB 3: DASHBOARD
# ============================================================================
with tab3:
    total, high, medium = get_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#b8956e;">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#a65a4a;">
            <div class="metric-label">High Priority</div>
            <div class="metric-value">{high}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#9a7a4a;">
            <div class="metric-label">Medium Priority</div>
            <div class="metric-value">{medium}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#ffffff; border-radius:12px; border:1px solid #e8ddd0; padding:20px; margin-top:12px;">
        <div style="font-size:14px; font-weight:600; color:#3d2c1e; margin-bottom:12px;">
            Recent Detections
        </div>
    """, unsafe_allow_html=True)
    
    rows = get_all_reports(limit=20)
    if rows:
        for row in rows:
            st.markdown(f"""
            <div style="background:#faf5ef; padding:10px 14px; border-radius:8px; margin-bottom:6px; border-left:4px solid #b8956e; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong>{row[1]}</strong> 
                    <span style="font-size:12px; color:#9a8776;">| {row[3]}</span>
                    <span style="font-size:12px; color:#9a8776; margin-left:8px;">Conf: {row[2]:.1%}</span>
                </div>
                <div style="font-size:11px; color:#b8a898;">
                    📍 {row[4]:.6f}, {row[5]:.6f} 
                    🕐 {row[6].strftime('%d %b %H:%M') if row[6] else 'N/A'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#9a8776; font-size:14px; text-align:center; padding:20px;">
            No detections in database yet. Capture images from Phone Camera tab.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 4: REPORTS
# ============================================================================
with tab4:
    st.markdown("""
    <div style="background:#faf5ef; border-radius:10px; padding:20px; border:1px solid #e8ddd0; margin-bottom:16px;">
        <div style="font-size:14px; color:#6a5a4a; font-weight:500;">
            📄 Report generation coming soon. Export PDF with all detections from database.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rows = get_all_reports(limit=50)
    if rows:
        st.download_button(
            "Download CSV Report",
            data="\n".join([",".join([str(row[0]), row[1], str(row[2]), row[3], str(row[4]), str(row[5]), str(row[6])]) for row in rows]),
            file_name="damage_report.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    RoadVision AI · YOLOv8 · PostgreSQL · Streamlit v2.0
</div>
""", unsafe_allow_html=True)