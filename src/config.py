"""
Configuration file for RoadVision AI project.
All paths, hyperparameters, and constants are defined here.
"""

import os
from pathlib import Path

# ============================================================================
# BASE PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # Root directory of project
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"
UTILS_DIR = BASE_DIR / "utils"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, APP_DIR, UTILS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# RDD2022 dataset paths (YOLO format)

DATASET_PATH = DATA_DIR / "rdd2022-yolo"  # Extract dataset here

# Dataset splits
TRAIN_IMAGES = DATASET_PATH / "train" / "images"
TRAIN_LABELS = DATASET_PATH / "train" / "labels"
VAL_IMAGES = DATASET_PATH / "val" / "images"
VAL_LABELS = DATASET_PATH / "val" / "labels"
TEST_IMAGES = DATASET_PATH / "test" / "images"
TEST_LABELS = DATASET_PATH / "test" / "labels"

# Dataset YAML file (contains class names and paths)
DATA_YAML = DATASET_PATH / "data.yaml"

# ============================================================================
# CLASS NAMES (RDD2022)
# ============================================================================

CLASS_NAMES = {
    0: "Longitudinal Crack",   # D00
    1: "Transverse Crack",     # D10
    2: "Alligator Crack",      # D20
    3: "Pothole",              # D40
}

CLASS_NAMES_LIST = list(CLASS_NAMES.values())
NUM_CLASSES = len(CLASS_NAMES)

# Severity mapping for maintenance prioritization
SEVERITY_MAP = {
    "Longitudinal Crack": "Medium",
    "Transverse Crack": "Medium",
    "Alligator Crack": "High",
    "Pothole": "High",
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# YOLO model variants: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'

YOLO_MODEL = "yolov8s.pt"  # Recommended for your GPU (6GB VRAM)

# Input image size for training/inference
IMAGE_SIZE = 640

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 16  # Adjust based on GPU memory (6GB VRAM -> 16 is safe)
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# Augmentation parameters
AUGMENTATION = {
    "hsv_h": 0.015,  # Hue augmentation
    "hsv_s": 0.7,    # Saturation augmentation
    "hsv_v": 0.4,    # Value augmentation
    "degrees": 0.0,  # Rotation
    "translate": 0.1, # Translation
    "scale": 0.5,    # Scaling
    "shear": 0.0,    # Shear
    "perspective": 0.0,  # Perspective
    "flipud": 0.0,   # Flip up-down
    "fliplr": 0.5,   # Flip left-right
    "mosaic": 1.0,   # Mosaic augmentation
    "mixup": 0.0,    # Mixup augmentation
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
SEED = 42
WORKERS = 4  # Number of dataloader workers
PATIENCE = 50  # Early stopping patience
SAVE_PERIOD = 10  # Save checkpoint every N epochs

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100

# ============================================================================
# MODEL SAVING
# ============================================================================
BEST_MODEL_PATH = MODELS_DIR / "best.pt"
LAST_MODEL_PATH = MODELS_DIR / "last.pt"
RESULTS_DIR = MODELS_DIR / "results"

# ============================================================================
# LOGGING
# ============================================================================
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================
def validate_paths():
    """Check if all required paths exist. Prints warnings if not."""
    if not DATASET_PATH.exists():
        print(f"WARNING: Dataset not found at {DATASET_PATH}")
        print("Please download RDD2022 YOLO dataset from:")
        print("https://www.kaggle.com/datasets/sreekaraditya/rdd2022-yolo-crackscan-v2")
    else:
        print(f"Dataset found at {DATASET_PATH}")
    
    if not DATA_YAML.exists():
        print(f"WARNING: data.yaml not found at {DATA_YAML}")

if __name__ == "__main__":
    validate_paths()
    print(f"\nConfiguration loaded successfully!")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {CLASS_NAMES_LIST}")
    print(f"YOLO Model: {YOLO_MODEL}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")