"""
YOLO model definition for road damage detection.
Handles training, validation, and inference.
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
from datetime import datetime

from .config import (
    YOLO_MODEL,
    DATA_YAML,
    EPOCHS,
    BATCH_SIZE,
    IMAGE_SIZE,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    PATIENCE,
    WORKERS,
    SEED,
    SAVE_PERIOD,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    RESULTS_DIR,
    LOG_DIR
)


class RoadDamageDetector:
    """
    Main class for road damage detection using YOLO.
    """
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the detector.
        
        Args:
            model_name: YOLO model name (e.g., 'yolov8s.pt')
            device: 'cpu' or 'cuda'. If None, auto-detects.
        """
        self.model_name = model_name or YOLO_MODEL
        self.data_yaml = DATA_YAML
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Model: {self.model_name}")
        
        # Load model
        self.model = YOLO(self.model_name)
        
        # Training results
        self.training_results = None
    
    def train(self, epochs=None, batch_size=None, imgsz=None, resume=False):
        """
        Train the model on RDD2022 dataset.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Image size
            resume: Resume training from last checkpoint
        """
        epochs = epochs or EPOCHS
        batch_size = batch_size or BATCH_SIZE
        imgsz = imgsz or IMAGE_SIZE
        
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {imgsz}")
        print(f"Device: {self.device}")
        print(f"Dataset: {self.data_yaml}")
        print(f"{'='*60}\n")
        
        # Training arguments
        args = {
            'data': str(self.data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': self.device,
            'workers': WORKERS,
            'seed': SEED,
            'patience': PATIENCE,
            'save_period': SAVE_PERIOD,
            'project': str(RESULTS_DIR),
            'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'lr0': LEARNING_RATE,
            'momentum': MOMENTUM,
            'weight_decay': WEIGHT_DECAY,
            'verbose': True,
            'plots': True,
        }
        
        # Train the model
        self.training_results = self.model.train(**args)
        
        # Save best model path
        best_path = RESULTS_DIR / self.training_results.save_dir / 'weights' / 'best.pt'
        last_path = RESULTS_DIR / self.training_results.save_dir / 'weights' / 'last.pt'
        
        # Copy best model to models directory
        if best_path.exists():
            import shutil
            shutil.copy2(best_path, BEST_MODEL_PATH)
            shutil.copy2(last_path, LAST_MODEL_PATH)
            print(f"\nBest model saved to: {BEST_MODEL_PATH}")
        
        return self.training_results
    
    def validate(self, model_path=None):
        model_path = model_path or BEST_MODEL_PATH
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}")
            return None
        
        print(f"\nValidating model: {model_path}")
        
        # Load model with data.yaml
        model = YOLO(str(model_path))
        results = model.val(
            data=str(self.data_yaml),
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=self.device,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            plots=True,
            save_json=True
        )
        return results
    
    def predict(self, image_path, save=True):
        if not Path(image_path).exists():
            print(f"File not found: {image_path}")
            return None
        
        # Load trained model
        if BEST_MODEL_PATH.exists():
            model = YOLO(str(BEST_MODEL_PATH))
        else:
            print(f"Model not found at {BEST_MODEL_PATH}")
            return None
        
        results = model.predict(
            source=image_path,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=self.device,
            save=save,
            save_txt=True,
            save_conf=True,
            project=str(RESULTS_DIR / 'predictions'),
            name='inference'
        )
        
        return results
    
    def export(self, format='onnx'):
        """
        Export model to different formats.
        
        Args:
            format: 'onnx', 'tensorrt', 'torchscript', etc.
        """
        if not BEST_MODEL_PATH.exists():
            print(f"Model not found at {BEST_MODEL_PATH}")
            return None
        
        print(f"Exporting model to {format} format...")
        export_path = self.model.export(
            format=format,
            imgsz=IMAGE_SIZE,
            device=self.device
        )
        print(f"Exported to: {export_path}")
        return export_path
    
    def get_summary(self):
        """
        Print model summary and training metrics.
        """
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        
        if self.training_results:
            results = self.training_results.results_dict
            print(f"Training completed at: {self.training_results.save_dir}")
            print(f"Best mAP@0.5: {results.get('metrics/mAP_0.5', 'N/A'):.4f}")
            print(f"Best mAP@0.5:0.95: {results.get('metrics/mAP_0.5:0.95', 'N/A'):.4f}")
            print(f"Precision: {results.get('metrics/precision', 'N/A'):.4f}")
            print(f"Recall: {results.get('metrics/recall', 'N/A'):.4f}")
        else:
            print("No training results found. Train the model first.")
        
        print("="*60 + "\n")


def main():
    """
    Quick test function to verify model setup.
    """
    detector = RoadDamageDetector()
    print(f"Model loaded successfully!")
    print(f"Model layers: {len(detector.model.model.model)}")
    print(f"Data YAML exists: {DATA_YAML.exists()}")
    
    # Show model summary
    detector.get_summary()


if __name__ == "__main__":
    main()