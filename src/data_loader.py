"""
Data loader for RDD2022 dataset.
Handles loading, validation, and preprocessing of YOLO format data.
"""

import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATASET_PATH, DATA_YAML, CLASS_NAMES, CLASS_NAMES_LIST,
    TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS,
    TEST_IMAGES, TEST_LABELS, IMAGE_SIZE
)


class RDD2022DataLoader:
    """
    Data loader class for RDD2022 dataset in YOLO format.
    Provides methods to load, validate, and explore dataset.
    """
    
    def __init__(self, dataset_path=None):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to dataset root. If None, uses config.DATASET_PATH
        """
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET_PATH
        self.class_names = CLASS_NAMES
        self.class_names_list = CLASS_NAMES_LIST
        self.num_classes = len(CLASS_NAMES)
        self.img_size = IMAGE_SIZE
        
        # Data splits
        self.splits = {
            'train': {
                'images': self.dataset_path / 'train' / 'images',
                'labels': self.dataset_path / 'train' / 'labels'
            },
            'val': {
                'images': self.dataset_path / 'val' / 'images',
                'labels': self.dataset_path / 'val' / 'labels'
            },
            'test': {
                'images': self.dataset_path / 'test' / 'images',
                'labels': self.dataset_path / 'test' / 'labels'
            }
        }
        
        # Load data.yaml configuration
        self.data_config = None
        if DATA_YAML.exists():
            with open(DATA_YAML, 'r') as f:
                self.data_config = yaml.safe_load(f)
        
        # Cache for dataset stats
        self._stats = None
    
    def validate_dataset(self):
        """
        Validate dataset structure and check for missing files.
        
        Returns:
            dict: Validation results with status and issues
        """
        results = {
            'valid': True,
            'issues': [],
            'stats': {}
        }
        
        for split_name, split_paths in self.splits.items():
            img_path = split_paths['images']
            label_path = split_paths['labels']
            
            # Check if directories exist
            if not img_path.exists():
                results['valid'] = False
                results['issues'].append(f"{split_name}/images missing at {img_path}")
                continue
            
            if not label_path.exists():
                results['valid'] = False
                results['issues'].append(f"{split_name}/labels missing at {label_path}")
                continue
            
            # Count files
            img_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png')) + list(img_path.glob('*.jpeg'))
            label_files = list(label_path.glob('*.txt'))
            
            # Check if image has corresponding label
            img_names = {f.stem for f in img_files}
            label_names = {f.stem for f in label_files}
            
            missing_labels = img_names - label_names
            missing_images = label_names - img_names
            
            if missing_labels:
                results['issues'].append(f"{split_name}: {len(missing_labels)} images missing labels")
            
            if missing_images:
                results['issues'].append(f"{split_name}: {len(missing_images)} labels missing images")
            
            results['stats'][split_name] = {
                'images': len(img_files),
                'labels': len(label_files),
                'valid_pairs': len(img_names.intersection(label_names))
            }
        
        if results['issues']:
            results['valid'] = False
        
        return results
    
    def get_dataset_stats(self):
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            dict: Dataset statistics including class distribution
        """
        if self._stats:
            return self._stats
        
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': Counter(),
            'avg_boxes_per_image': 0,
            'splits': {}
        }
        
        for split_name, split_paths in self.splits.items():
            label_path = split_paths['labels']
            if not label_path.exists():
                continue
            
            label_files = list(label_path.glob('*.txt'))
            split_stats = {
                'images': len(label_files),
                'annotations': 0,
                'class_counts': Counter(),
                'boxes_per_image': []
            }
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        num_boxes = len(lines)
                        split_stats['annotations'] += num_boxes
                        split_stats['boxes_per_image'].append(num_boxes)
                        
                        # Count class occurrences
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                split_stats['class_counts'][class_id] += 1
                except Exception as e:
                    print(f"Error reading {label_file}: {e}")
            
            stats['splits'][split_name] = {
                'images': split_stats['images'],
                'annotations': split_stats['annotations'],
                'class_distribution': dict(split_stats['class_counts']),
                'avg_boxes': np.mean(split_stats['boxes_per_image']) if split_stats['boxes_per_image'] else 0
            }
            
            stats['total_images'] += split_stats['images']
            stats['total_annotations'] += split_stats['annotations']
            stats['class_distribution'] += split_stats['class_counts']
        
        # Calculate overall average
        total_boxes = []
        for split_name, split_stats in stats['splits'].items():
            # Recalculate from scratch for accuracy
            label_path = self.splits[split_name]['labels']
            if label_path.exists():
                for label_file in label_path.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        total_boxes.append(len(f.readlines()))
        
        stats['avg_boxes_per_image'] = np.mean(total_boxes) if total_boxes else 0
        
        self._stats = stats
        return stats
    
    def visualize_sample(self, split='train', num_samples=4):
        """
        Visualize sample images with their bounding boxes.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            num_samples: Number of samples to visualize
        """
        img_path = self.splits[split]['images']
        label_path = self.splits[split]['labels']
        
        if not img_path.exists() or not label_path.exists():
            print(f"Split '{split}' not found")
            return
        
        img_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png')) + list(img_path.glob('*.jpeg'))
        
        # Select random samples
        import random
        selected = random.sample(img_files, min(num_samples, len(img_files)))
        
        fig, axes = plt.subplots(1, len(selected), figsize=(15, 5))
        if len(selected) == 1:
            axes = [axes]
        
        for idx, img_file in enumerate(selected):
            # Load image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            # Load corresponding label
            label_file = label_path / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * width
                            y_center = float(parts[2]) * height
                            box_width = float(parts[3]) * width
                            box_height = float(parts[4]) * height
                            
                            # Convert to x1, y1, x2, y2
                            x1 = int(x_center - box_width / 2)
                            y1 = int(y_center - box_height / 2)
                            x2 = int(x_center + box_width / 2)
                            y2 = int(y_center + box_height / 2)
                            
                            # Draw bounding box
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            class_name = self.class_names.get(class_id, f"Class {class_id}")
                            cv2.putText(img, class_name, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_file.name}\n{img_file.stem}")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """
        Print a summary of the dataset.
        """
        print("\n" + "="*60)
        print("RDD2022 DATASET SUMMARY")
        print("="*60)
        
        # Validate first
        validation = self.validate_dataset()
        if validation['valid']:
            print("Status: VALID")
        else:
            print("Status: ISSUES FOUND")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        print("\nDataset Statistics:")
        stats = self.get_dataset_stats()
        
        print(f"Total Images: {stats['total_images']:,}")
        print(f"Total Annotations: {stats['total_annotations']:,}")
        print(f"Average Boxes per Image: {stats['avg_boxes_per_image']:.2f}")
        
        print("\nClass Distribution:")
        for class_id, count in sorted(stats['class_distribution'].items()):
            class_name = self.class_names.get(class_id, f"Unknown-{class_id}")
            percentage = (count / stats['total_annotations']) * 100 if stats['total_annotations'] > 0 else 0
            print(f"  {class_id}: {class_name:20s} {count:6,} ({percentage:.1f}%)")
        
        print("\nSplit-wise Statistics:")
        for split_name, split_stats in stats['splits'].items():
            print(f"  {split_name}:")
            print(f"    Images: {split_stats['images']:,}")
            print(f"    Annotations: {split_stats['annotations']:,}")
            print(f"    Avg Boxes per Image: {split_stats['avg_boxes']:.2f}")
        
        print("="*60 + "\n")


def main():
    """
    Quick test function to run when this file is executed directly.
    """
    loader = RDD2022DataLoader()
    
    # Validate dataset
    print("Validating dataset...")
    validation = loader.validate_dataset()
    print(f"Valid: {validation['valid']}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    # Print summary
    loader.print_summary()
    
    # Visualize some samples
    print("Visualizing samples...")
    loader.visualize_sample('train', num_samples=4)


if __name__ == "__main__":
    main()