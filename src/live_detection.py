import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import psycopg2
from datetime import datetime

# Load model
model_path = Path(__file__).resolve().parent.parent / "models" / "best.pt"
model = YOLO(str(model_path))

# Database config
DB_CONFIG = {
    "host": "localhost",
    "database": "roadvision_db",
    "user": "postgres",
    "password": "postgres"
}

def save_to_db(image_path, class_name, confidence, severity, lat, lon):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO damage_reports 
            (image_path, class_name, confidence, severity, latitude, longitude, device_id, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (image_path, class_name, confidence, severity, lat, lon, "cctv", datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"DB error: {e}")
        return False

def get_severity(class_name):
    high = ["Pothole", "Alligator Crack"]
    return "High" if class_name in high else "Medium"

# Open webcam (0 = default, 1 = OBS Virtual Camera)
cap = cv2.VideoCapture(0)  # Try 0, 1, 2

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Press 's' to save detection, 'q' to quit")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    results = model.predict(frame, conf=0.25, save=False, verbose=False)
    annotated = results[0].plot()
    
    # Show detections
    cv2.imshow("RoadVision AI - Live CCTV", annotated)
    
    # If damage detected
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            severity = get_severity(class_name)
            
            print(f"Detected: {class_name} ({conf:.2%})")
            
            # Save on 's' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                img_path = f"captures/cctv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(img_path, frame)
                save_to_db(img_path, class_name, conf, severity, 0.0, 0.0)
                print(f"Saved to DB: {class_name}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()