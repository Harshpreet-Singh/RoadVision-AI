# RoadVision AI

Intelligent Road Damage Detection and Maintenance Prioritization Using Computer Vision

## Overview
RoadVision AI is an automated system that detects road damage (potholes, cracks, and defects) from images and videos using YOLOv8 object detection. It provides confidence scores, severity estimation, and generates inspection reports for infrastructure maintenance prioritization.

## Features
- Real-time pothole and crack detection
- Confidence scoring for each detection
- Damage severity estimation (Low/Medium/High)
- Image and video upload support
- Inspection report generation
- Streamlit-based dashboard

## Tech Stack
- Python 3.11
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy/Pandas

## Dataset
RDD2022 - Road Damage Detection Dataset (47,420 images from Japan, India, Czech Republic, Norway)

## Project Structure

```text
RoadVision-AI/
├── data/ # Dataset (ignored by git)
├── notebooks/ # Jupyter notebooks
├── src/ # Source code
│   ├── config.py # Configuration
│   ├── data_loader.py # Data loading
│   └── model.py # YOLO model
├── models/ # Trained weights (ignored)
├── app/ # Streamlit application
│   └── app.py
├── utils/ # Helper functions
├── requirements.txt # Dependencies
└── README.md
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Harshpreet-Singh/RoadVision-AI.git
cd RoadVision-AI
```
### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download dataset

Download RDD2022 YOLO format from:
https://www.kaggle.com/datasets/sreekaraditya/rdd2022-yolo-crackscan-v2

Place extracted files in data/ folder.

5. Run the application
```bash
streamlit run app/app.py
```

### Training the Model
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

## License
This project is licensed under the MIT License.

## Author
Harshpreet Singh