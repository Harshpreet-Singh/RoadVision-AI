# RoadVision AI

Intelligent Road Damage Detection and Maintenance Prioritization Using Computer Vision

## Overview

RoadVision AI is an end-to-end system that detects road damage (potholes, cracks, and defects) from images, videos, and live camera feeds using YOLOv8. It saves detections to a PostgreSQL database, provides a real-time dashboard, and generates professional PDF inspection reports for maintenance prioritization.

## Features

- Real-time road damage detection (images, webcam, CCTV)
- YOLOv8-based object detection with 4 damage classes
- Confidence scoring and severity estimation (High/Medium)
- Live webcam detection with instant database save
- Phone camera integration with location capture
- PostgreSQL database for persistent storage
- Dashboard with filters, charts, and analytics
- PDF report generation with charts and tables
- CSV export for selected detections
- Warm, professional UI with responsive layout

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Deep Learning | PyTorch, YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Frontend | Streamlit |
| Database | PostgreSQL |
| PDF Generation | ReportLab |
| Data Handling | NumPy, Pandas, PIL |

## Dataset

**RDD2022** — Road Damage Detection Dataset (47,420 images, 7 countries)

- Classes: Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole
- Split: 32,628 train / 5,757 val / 9,035 test
- Format: YOLO txt labels

## Model Performance

| Metric | Value |
|--------|-------|
| Model | YOLOv8s |
| mAP50 | 0.574 |
| mAP50-95 | 0.293 |
| Precision | 0.623 |
| Recall | 0.538 |
| Inference Speed | ~9.6ms/image |

### Class-wise Performance

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| Alligator Crack | 0.675 | 0.358 |
| Transverse Crack | 0.576 | 0.285 |
| Longitudinal Crack | 0.569 | 0.314 |
| Pothole | 0.476 | 0.214 |

## Project Structure

```text
RoadVision-AI/
├── app/
│   └── app.py                  # Streamlit application
├── data/                       # Dataset (ignored by git)
│   └── data.yaml               # Dataset config
├── models/
│   └── best.pt                 # Trained model weights (ignored)
├── reports/
│   └── generator.py            # PDF report generator
├── src/
│   ├── config.py               # Configuration
│   ├── data_loader.py          # Data loading & validation
│   ├── model.py                # YOLO model class
│   └── live_detection.py       # Real-time webcam detection
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Harshpreet-Singh/RoadVision-AI.git
cd RoadVision-AI
```

### 2. Create virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download dataset

Download RDD2022 YOLO format from:

https://www.kaggle.com/datasets/sreekaraditya/rdd2022-yolo-crackscan-v2

Extract to the `data/` folder:

```text
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 5. Setup PostgreSQL database

```sql
CREATE DATABASE roadvision_db;

CREATE TABLE damage_reports (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(500),
    class_name VARCHAR(50),
    confidence FLOAT,
    severity VARCHAR(20),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    location_name VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    device_id VARCHAR(50),
    cycle_id INTEGER,
    confirmed BOOLEAN DEFAULT TRUE
);
```

### 6. Configure database credentials

Edit `app/app.py` and `src/live_detection.py`:

```python
DB_CONFIG = {
    "host": "localhost",
    "database": "roadvision_db",
    "user": "postgres",
    "password": "your_password"
}
```

## Usage

### Training the Model

```bash
python -c "from src.model import RoadDamageDetector; d = RoadDamageDetector(); d.train(epochs=30)"
```

### Validation

```bash
yolo val model=models/best.pt data=data/data.yaml
```

### Streamlit App

```bash
streamlit run app/app.py
```

### Live Webcam Detection

```bash
python src/live_detection.py
```

- Press `s` to save detection to the database
- Press `q` to quit

### Mobile Access

```bash
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Access from your phone via:

```text
http://<your-ip>:8501
```

## Application Tabs

| Tab | Purpose |
|-----|---------|
| **Upload Image** | Upload and detect damages from image files |
| **Phone Camera** | Capture from webcam/phone with location input |
| **Dashboard** | View all detections with filters, charts, and checkboxes |
| **Reports** | Generate PDF/CSV from selected detections |

## Report Generation

The PDF report includes:

- Summary statistics (Total, High, Medium, Types)
- Detection table with all selected rows
- Class distribution bar chart
- Severity distribution pie chart
- Recommendations section
- Professional header and footer

## Sample Detection

| Image | Detections |
|-------|------------|
| `China_MotorBike_002004.jpg` | 3 Longitudinal Cracks (71.7%, 40.3%, 29.1%) |

## License

This project is licensed under the MIT License.

## Author

**Harshpreet Singh**

## Commit

```bash
git add README.md
git commit -m "Update README with complete project documentation"
git push
```
