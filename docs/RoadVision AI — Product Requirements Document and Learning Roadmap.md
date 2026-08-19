# RoadVision AI

## Product Requirements Document (PRD) + Learning and Implementation Roadmap

**Project Type:** Academic Major/Subject Project\
**Domain:** Artificial Intelligence + Deep Learning + Computer Vision + Full-Stack Development\
**Timeline:** August–December 2026\
**Recommended Team Size:** 4 Students\
**Project Status:** Planning Phase\
**Version:** 1.0

---

# 1. Project Overview

## 1.1 Project Name

**RoadVision AI: An AI-Assisted Road Damage Detection and Maintenance Prioritization System**

---

## 1.2 Problem Statement

Road infrastructure requires regular inspection to identify potholes, cracks, and other surface defects. Traditional inspection can be time-consuming, manual, inconsistent, and difficult to scale.

Existing computer vision demonstrations often stop after detecting a pothole and drawing a bounding box around it.

RoadVision AI aims to go beyond simple detection.

The system will analyze road images and videos to identify visible road damage and provide structured information that can assist with road inspection and maintenance prioritization.

The long-term product concept is:

```text
Road Image / Video
        ↓
AI Damage Detection
        ↓
Damage Classification
        ↓
Confidence + Location of Damage
        ↓
Damage Severity Assessment
        ↓
Maintenance Priority Estimation
        ↓
Inspection Record
        ↓
Dashboard / Report / Analytics
```

---

# 2. Why This Project

This project is selected as a deliberate next step after completing a classical machine learning project.

Previous ML experience provides familiarity with concepts such as:

- Data cleaning
- Feature engineering
- Data preprocessing
- Training machine learning models
- Comparing models
- Hyperparameter tuning
- Model evaluation
- Prediction pipelines
- Connecting an ML model with a frontend application

The next learning gap is:

```text
Classical Machine Learning
        ↓
Deep Learning
        ↓
Neural Networks
        ↓
Computer Vision
        ↓
Convolutional Neural Networks
        ↓
PyTorch
        ↓
Object Detection
        ↓
YOLO
        ↓
Complete AI Product
```

Therefore, RoadVision AI is not intended to be only a "pothole detector."

It will serve two purposes:

1. **Academic/Product Goal:** Build a complete road damage detection and prioritization system.
2. **Learning Goal:** Use the project to systematically transition from classical ML into Deep Learning and Computer Vision.

---

# 3. Primary Objectives

The project should be able to:

1. Accept a road image as input.
2. Detect visible road damage.
3. Identify the damage category where supported by the selected dataset.
4. Display bounding boxes and confidence scores.
5. Process videos in the advanced phase.
6. Store inspection results.
7. Estimate or assign damage severity using a clearly defined methodology.
8. Generate a maintenance priority level.
9. Display historical inspection data and analytics.
10. Generate an inspection report.

---

# 4. Core Product Vision

## Basic Student Project Version

```text
Upload Image
      ↓
YOLO Model
      ↓
Pothole / Crack Detection
      ↓
Bounding Boxes
      ↓
Confidence Score
```

This alone is NOT the final intended vision.

---

## RoadVision AI Version

```text
Inspection Input
(Image / Video)
        ↓
Computer Vision Model
        ↓
Road Damage Detection
        ↓
Damage Classification
        ↓
Damage Analysis
        ↓
Severity Estimation
        ↓
Maintenance Priority
        ↓
Store Inspection
        ↓
Dashboard + History + Report
```

The AI model is therefore one important component of a larger system.

---

# 5. Target Users

## Primary Academic Users

- Project evaluators
- Faculty
- Students
- Demonstration audience

## Potential Real-World Users

- Municipal road inspection teams
- Highway maintenance teams
- Civil infrastructure departments
- Private road maintenance companies
- Smart city monitoring systems

The first implementation will be a prototype and must not claim to replace professional road inspection.

---

# 6. Project Scope

## Phase 1 Scope — Mandatory MVP

The Minimum Viable Product must include:

- Image upload
- AI road damage detection
- Bounding boxes
- Damage class
- Confidence score
- Processed image output
- Basic result summary
- Model evaluation
- Proper documentation

---

## Phase 2 Scope — Strong Project Version

Add:

- Multiple damage categories where dataset support exists
- Video processing
- Detection history
- Database storage
- Severity estimation
- Maintenance priority
- Dashboard analytics
- Downloadable inspection report

---

## Phase 3 Scope — Stretch Goals

Only attempted after the core project works reliably:

- GPS/location integration
- Road-wise damage mapping
- Mobile camera integration
- Real-time webcam detection
- Edge deployment
- Model optimization
- Advanced severity estimation
- Multi-model comparison
- Road maintenance recommendation logic

---

# 7. Functional Requirements

## FR-01: Image Input

The user shall be able to upload a road image.

Supported input should initially include common image formats such as:

- JPG
- JPEG
- PNG

The system shall validate the uploaded file.

---

## FR-02: Road Damage Detection

The AI model shall process the uploaded image and identify supported damage classes.

Initial target classes depend on the final dataset.

Potential classes may include:

- Pothole
- Crack
- Other road surface defects

**Important:** Final supported classes will be determined only after dataset selection and analysis.

The project must not promise classes that the selected training dataset cannot reliably support.

---

## FR-03: Detection Visualization

The system shall display:

- Original image
- Processed image
- Bounding boxes
- Detected class
- Confidence score

---

## FR-04: Detection Summary

For every inspection, the system shall provide a summary containing:

- Number of detected defects
- Damage categories detected
- Detection confidence
- Timestamp
- Inspection identifier

---

## FR-05: Video Processing

In the advanced phase, the user shall be able to provide a video.

The system shall:

1. Read frames.
2. Run damage detection.
3. Annotate detections.
4. Produce processed visual output.

Real-time performance is not mandatory for the first version.

---

## FR-06: Severity Estimation

RoadVision AI should not initially claim that a YOLO confidence score equals damage severity.

These are different concepts.

For example:

```text
Detection Confidence:
How confident is the model that an object is a pothole?

Damage Severity:
How serious is the detected road damage?
```

A severity methodology must therefore be explicitly designed.

Initial prototype approaches may investigate features such as:

- Number of detected damages
- Relative bounding-box area
- Damage type
- Number of severe-looking defects within an image

Example prototype:

```text
Low Severity
    ↓
Small / limited detected damage

Medium Severity
    ↓
Moderate damage indicators

High Severity
    ↓
Large or multiple damage indicators
```

The exact logic must be documented and presented as a **prototype prioritization methodology**, not as a certified civil engineering measurement.

---

## FR-07: Maintenance Priority

Based on the severity methodology, the system should assign a priority such as:

- Low — Monitor
- Medium — Repair Soon
- High — Immediate Inspection Recommended

This feature is intended to make the project more useful than simple object detection.

---

## FR-08: Detection History

Advanced versions should store inspection records.

Example fields:

- Inspection ID
- Timestamp
- Image reference
- Damage types
- Number of detections
- Severity
- Maintenance priority

---

## FR-09: Dashboard

The system should provide analytics such as:

- Total inspections
- Total detected damages
- Damage type distribution
- Severity distribution
- High-priority inspections
- Recent inspections

---

## FR-10: Inspection Report

The advanced system should generate a structured inspection report containing:

- Inspection details
- Original or processed image reference
- Detected damage summary
- Confidence information
- Severity result
- Maintenance priority
- Disclaimer regarding prototype limitations

---

# 8. Non-Functional Requirements

## Performance

The system should provide inference within a reasonable time on the available hardware.

Exact performance targets will be finalized after model and hardware testing.

---

## Reliability

The system should:

- Handle invalid uploads
- Handle images with no detected damage
- Avoid application crashes
- Clearly communicate prediction limitations

---

## Usability

The interface should be simple enough for a non-ML user to:

1. Upload an image.
2. Run inspection.
3. Understand the result.
4. Review the priority.

---

## Maintainability

The repository should use a modular structure.

Example:

```text
roadvision-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset_docs/
│
├── notebooks/
│   └── optional_experiments/
│
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── severity/
│   └── utils/
│
├── models/
│
├── experiments/
│
├── app/
│
├── reports/
│
├── docs/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

The final structure may change as the architecture becomes clearer.

---

# 9. Technology Stack

## Deep Learning

**PyTorch**

Purpose:

- Understand tensors
- Build Deep Learning knowledge
- Train and evaluate models
- Understand model workflows

---

## Computer Vision

**OpenCV**

Purpose:

- Image reading
- Image preprocessing
- Video processing
- Frame handling
- Image visualization

---

## Object Detection

**YOLO**

Purpose:

- Detect multiple road damages
- Identify locations using bounding boxes
- Generate confidence scores

The exact YOLO version should be selected during implementation rather than permanently fixed in the planning phase.

---

## Application Layer

Initial recommendation:

**Streamlit**

Reason:

- Faster AI prototype development
- Focus remains on ML/DL learning
- Easy image and video demonstrations

Possible later architecture:

```text
React Frontend
        ↓
FastAPI Backend
        ↓
Model Inference Service
        ↓
Database
```

This migration is optional and should only happen if the MVP is already complete.

The project must not sacrifice Deep Learning work simply to build a more complicated frontend.

---

## Database

Possible initial choice:

- SQLite

Possible later upgrade:

- PostgreSQL

Database implementation should begin only after the detection pipeline is functional.

---

# 10. Learning Roadmap

This section is a core part of the project.

The goal is not to blindly use YOLO.

The goal is to understand enough of the underlying concepts to explain the project confidently.

---

# Stage 0 — Prepare and Understand the Problem

## Learn

- What is Computer Vision?
- Difference between image classification, object detection, and segmentation
- What is a dataset?
- Train/validation/test split
- Image labels and annotations
- Bounding boxes

## Implement

- Explore sample road damage datasets
- Read dataset documentation
- Visualize images
- Visualize annotations
- Inspect class distribution

## Deliverable

**Dataset Exploration Report**

Questions to answer:

- What classes exist?
- How many images exist?
- Is the dataset balanced?
- What annotation format is used?
- What are the limitations?

---

# Stage 1 — Neural Network Fundamentals

## Learn

- What is a neural network?
- Neurons and layers
- Weights and biases
- Activation functions
- Forward propagation
- Loss function
- Gradient descent
- Backpropagation
- Epoch
- Batch size
- Learning rate
- Overfitting and underfitting

## Implementation

Build at least one small neural network exercise.

The goal is understanding, not production quality.

Possible example:

```text
Input
  ↓
Hidden Layer
  ↓
Hidden Layer
  ↓
Output
```

## Deliverable

**Neural Network Learning Notes + Small Implementation**

You should be able to explain:

> How does a neural network learn?

---

# Stage 2 — PyTorch Fundamentals

## Learn

- Tensors
- Tensor shapes
- CPU vs GPU
- Dataset and DataLoader
- Model class
- Forward pass
- Optimizer
- Loss calculation
- Training loop
- Validation loop

## Implement

Create a small PyTorch model before starting YOLO.

Possible task:

- MNIST digit classification
- CIFAR image classification

The exact starter dataset can be selected later.

## Deliverable

**First PyTorch Training Project**

This stage ensures YOLO is not the first thing you ever train.

---

# Stage 3 — CNN Fundamentals

## Learn

- Why dense neural networks are not ideal for images
- Convolution
- Filters/kernels
- Feature maps
- Padding
- Stride
- Pooling
- Channels
- ReLU
- Flattening
- Fully connected layers

Conceptual flow:

```text
Image
  ↓
Convolution
  ↓
Feature Maps
  ↓
Activation
  ↓
Pooling
  ↓
More Features
  ↓
Prediction
```

## Implement

Train or study a small CNN classification example.

## Deliverable

**CNN Learning Notes + Experiment**

You should understand:

> Why CNNs can learn edges, patterns, textures, and higher-level visual features.

---

# Stage 4 — Computer Vision Fundamentals

## Learn

- Image pixels
- RGB channels
- Image dimensions
- Resizing
- Normalization
- Data augmentation
- OpenCV basics

## Implement

Create small scripts for:

- Reading images
- Resizing
- Converting color formats
- Drawing rectangles
- Processing video frames

## Deliverable

**Computer Vision Utility Module**

---

# Stage 5 — Object Detection Fundamentals

Before using YOLO, learn:

- Classification vs detection
- Bounding boxes
- Ground truth
- Predicted boxes
- IoU
- Non-Maximum Suppression
- Precision
- Recall
- False positives
- False negatives
- mAP

Important learning objective:

Do not report only "accuracy."

Object detection requires more appropriate metrics.

## Deliverable

**Object Detection Concepts Documentation**

---

# Stage 6 — YOLO

## Learn

- What problem YOLO solves
- One-stage object detection
- YOLO dataset format
- Training pipeline
- Pretrained weights
- Transfer learning
- Fine-tuning
- Inference
- Validation

## Implement

First:

- Run inference using pretrained weights.

Then:

- Train/fine-tune on the selected road damage dataset.

## Deliverable

**Baseline Road Damage Detection Model**

---

# Stage 7 — Dataset Preparation

This is one of the most important implementation stages.

## Tasks

- Select dataset
- Verify licenses and intended use where applicable
- Inspect annotation quality
- Remove unusable data where justified
- Check class distribution
- Check duplicate images where possible
- Create train/validation/test splits
- Document all preprocessing

Potential augmentation experiments:

- Brightness changes
- Contrast changes
- Blur
- Rotation where appropriate
- Scaling

Augmentation must be justified rather than added randomly.

## Deliverable

**Versioned Training Dataset + Dataset Documentation**

---

# Stage 8 — Baseline Model

Train the first baseline.

Do not spend too much time tuning immediately.

Record:

- Model
- Parameters
- Dataset version
- Image size
- Epochs
- Training time
- Validation results
- Observations

## Deliverable

**Baseline Experiment Report**

---

# Stage 9 — Model Experimentation

This is where the project begins to move beyond a simple tutorial.

Possible experiments:

### Experiment A — Model Size

Compare appropriate model variants.

Example questions:

- Does a larger model improve mAP?
- Is the improvement worth the extra inference time?

### Experiment B — Image Size

Compare selected input sizes.

Example:

```text
Smaller Input
    ↓
Faster Training / Inference
    ↓
Potentially Less Detail

Larger Input
    ↓
More Detail
    ↓
Higher Compute Cost
```

### Experiment C — Data Augmentation

Measure whether selected augmentation improves generalization.

### Experiment D — Hyperparameters

Only after a baseline exists.

Potential variables:

- Learning rate
- Batch size
- Epochs
- Image size

Do not randomly tune everything.

Every experiment should answer a question.

---

# 11. Experiment Tracking

Create an experiment log.

Every experiment should record:

| Experiment | Change        | Reason               | Result | Decision |
| ---------- | ------------- | -------------------- | ------ | -------- |
| Baseline   | Initial model | Establish benchmark  | TBD    | Keep     |
| Exp-01     | Model variant | Compare capacity     | TBD    | TBD      |
| Exp-02     | Image size    | Test detail vs speed | TBD    | TBD      |
| Exp-03     | Augmentation  | Improve robustness   | TBD    | TBD      |

The final project should include both successful and unsuccessful experiments.

Failed experiments are useful if they explain a technical decision.

---

# 12. Evaluation Strategy

The model should be evaluated using object detection metrics such as:

- Precision
- Recall
- mAP
- IoU-based evaluation where applicable
- Confusion matrix
- Per-class performance where supported

Also perform qualitative analysis.

Test examples such as:

- Clear road damage
- Small damage
- Multiple damages
- Difficult lighting
- Shadows
- Blurry images
- Road textures that may resemble damage

Create a section called:

## Failure Analysis

Example questions:

- Where does the model fail?
- What causes false positives?
- What causes missed detections?
- Are some classes harder than others?
- Does lighting affect predictions?

This section is essential for demonstrating genuine understanding.

---

# 13. Severity and Prioritization Research Stage

This feature must be developed separately from basic object detection.

## Step 1

Clearly define what severity means for the prototype.

## Step 2

Investigate what information can reasonably be inferred from a 2D image.

## Step 3

Develop a transparent prototype scoring methodology.

Potential conceptual inputs:

```text
Damage Type
        +
Detected Area Proxy
        +
Number of Damages
        +
Optional Confidence Validation
        ↓
Prototype Severity Score
        ↓
Maintenance Priority
```

The exact formula or ML approach should not be finalized before dataset analysis.

The project should document limitations such as:

- Bounding-box size is not equal to physical depth.
- A 2D image cannot reliably measure all structural properties.
- Perspective can affect apparent damage size.

This honesty will strengthen the project.

---

# 14. Application Development Stages

## App Version 1 — Model Demo

Features:

- Upload image
- Run prediction
- Show processed image
- Show detections

Goal:

Verify that the model can be used outside the training environment.

---

## App Version 2 — Inspection Experience

Add:

- Detection summary
- Better result visualization
- Clear severity display
- Maintenance priority

---

## App Version 3 — Data and History

Add:

- Save inspections
- View history
- Filter results
- Dashboard analytics

---

## App Version 4 — Reports

Add:

- Structured inspection report
- Download functionality
- Result summary

---

# 15. Recommended Team Responsibilities

## Member 1 — AI/Deep Learning Lead

Primary work:

- PyTorch
- CNN learning
- YOLO
- Training
- Experiments
- Evaluation

Important:

This member should document model decisions.

---

## Member 2 — Dataset and Research Lead

Primary work:

- Dataset search and evaluation
- Annotation inspection
- Dataset analysis
- Literature review
- Experiment documentation

---

## Member 3 — Application/UI Lead

Primary work:

- Streamlit or frontend
- Upload workflow
- Results visualization
- Dashboard
- User experience

---

## Member 4 — Integration and Product Lead

Primary work:

- Backend integration
- Database
- Detection history
- Reports
- Testing
- Repository management

---

## Team Rule

Although members specialize, every team member should understand:

```text
Input
  ↓
Dataset
  ↓
Training
  ↓
Evaluation
  ↓
Inference
  ↓
Application
```

No member should be unable to explain the complete system during a viva.

---

# 16. August–December Roadmap

# August — Deep Learning Foundation

### Week 1

- Project setup
- Git repository
- Python environment
- Dataset research
- Neural network fundamentals

### Week 2

- PyTorch fundamentals
- First PyTorch model
- Training loop understanding

### Week 3

- CNN fundamentals
- Small image classification experiment

### Week 4

- OpenCV
- Object detection concepts
- IoU, Precision, Recall, mAP
- Run pretrained YOLO

### August Success Condition

> Every core AI team member can explain a neural network, CNN, basic PyTorch workflow, and object detection fundamentals.

---

# September — Road Damage AI Development

### Week 1

- Final dataset selection
- Dataset analysis
- Annotation verification

### Week 2

- Dataset preparation
- Baseline YOLO training

### Week 3

- Evaluation
- Error analysis

### Week 4

- Controlled experiments
- Model comparison or optimization

### September Success Condition

> A trained road damage detection model produces documented evaluation results.

---

# October — Product Implementation

### Week 1

- Model inference pipeline
- Image upload application

### Week 2

- Results visualization
- Detection summary

### Week 3

- Video support if feasible

### Week 4

- Severity and prioritization prototype

### October Success Condition

> A complete user can upload an image and receive an understandable inspection result.

---

# November — RoadVision Intelligence Layer

Focus:

- Detection history
- Database
- Dashboard
- Analytics
- Severity methodology refinement
- Maintenance prioritization
- Report generation
- Testing

### November Success Condition

> The project functions as an inspection system rather than only a model demo.

---

# December — Finalization

Focus:

- Code cleanup
- README
- Architecture documentation
- Final experiment report
- Presentation
- Viva preparation
- Demo testing
- Known limitations
- Future work

### December Success Condition

> Every feature is stable, documented, demonstrable, and explainable.

---

# 17. Project Success Metrics

Success will not be defined only by a high model score.

The project will be evaluated across multiple dimensions.

## Learning Success

Can the team explain:

- Neural networks?
- CNNs?
- PyTorch?
- Transfer learning?
- Object detection?
- IoU?
- Precision?
- Recall?
- mAP?
- Why the final model was selected?

---

## ML Success

- Reproducible training
- Documented dataset
- Baseline model
- Controlled experiments
- Proper evaluation
- Error analysis

---

## Engineering Success

- Modular code
- Working inference pipeline
- Functional application
- Error handling
- Clean repository

---

## Product Success

A user should be able to understand:

- What damage was detected
- Where it was detected
- How confident the model is
- What the prototype severity level means
- What maintenance priority is suggested

---

# 18. What Will Make This Project Stand Out

The project should NOT claim originality merely because it uses YOLO.

The differentiation should come from the complete engineering and analysis process.

RoadVision AI should aim to include:

### 1. Learning-First Development

The team learns core Deep Learning concepts before treating YOLO as a black box.

### 2. Experiment-Driven Model Selection

The final model is selected using evidence rather than simply choosing the latest or largest model.

### 3. Failure Analysis

The project openly shows where the model works and where it fails.

### 4. Detection to Decision

The system goes beyond:

```text
Pothole Detected: 92%
```

toward:

```text
Detected Road Damage
        ↓
Analyze Inspection
        ↓
Prototype Severity
        ↓
Maintenance Priority
        ↓
Stored Inspection Record
```

### 5. Complete Documentation

The repository documents:

- Dataset decisions
- Model experiments
- Results
- Failures
- Architecture
- Limitations

### 6. Honest Technical Claims

The system must not claim that:

- Confidence equals severity
- Bounding-box size equals physical damage depth
- The prototype can replace professional road inspection

Instead, limitations will be explicitly documented.

---

# 19. Risks

## Risk 1 — Trying to Learn Everything Before Building

Mitigation:

Learn in stages and immediately apply knowledge.

---

## Risk 2 — Using YOLO as a Black Box

Mitigation:

Complete the Neural Network → PyTorch → CNN → Object Detection learning stages first.

---

## Risk 3 — Poor Dataset Quality

Mitigation:

Perform dataset analysis before training.

---

## Risk 4 — Too Much Scope

Mitigation:

Follow this priority:

```text
1. Learning Foundation
2. Dataset
3. Working Detection Model
4. Evaluation
5. Application
6. Severity/Priority
7. History/Dashboard
8. Reports
9. Stretch Features
```

Never start feature 9 before feature 1–8 are stable.

---

## Risk 5 — Team Members Working in Isolation

Mitigation:

Weekly integration and knowledge-sharing sessions.

---

## Risk 6 — Building UI Before the Model Works

Mitigation:

Model pipeline first.

A beautiful UI cannot compensate for an unreliable AI pipeline.

---

# 20. Definition of Done

RoadVision AI will be considered complete when:

- [ ] Dataset is documented.
- [ ] Deep Learning fundamentals have been studied and documented.
- [ ] A small PyTorch project has been completed.
- [ ] CNN fundamentals are understood.
- [ ] Object detection metrics are understood.
- [ ] A baseline road damage model exists.
- [ ] Controlled experiments were performed.
- [ ] Final model selection is justified.
- [ ] Evaluation results are documented.
- [ ] Failure analysis is included.
- [ ] Image inference works in the application.
- [ ] Detection results are clearly visualized.
- [ ] Severity methodology is documented if implemented.
- [ ] Maintenance prioritization is clearly explained if implemented.
- [ ] Application is tested.
- [ ] README is complete.
- [ ] Architecture is documented.
- [ ] Presentation is prepared.
- [ ] Every team member can explain the complete pipeline.
- [ ] Known limitations are documented.

---

# 21. Final Development Philosophy

The purpose of this project is not:

> "Use the newest YOLO model and get the highest number possible."

The purpose is:

> **Learn Deep Learning properly by building a complete Computer Vision system and making technically justified decisions at every stage.**

The final journey should look like:

```text
Classical ML Knowledge
        ↓
Understand Neural Networks
        ↓
Learn PyTorch
        ↓
Understand CNNs
        ↓
Learn Computer Vision
        ↓
Understand Object Detection
        ↓
Train Road Damage Model
        ↓
Evaluate and Analyze Failures
        ↓
Experiment and Improve
        ↓
Build RoadVision AI
        ↓
Severity + Prioritization
        ↓
Dashboard + Reports
        ↓
Complete AI Product
```

---

# Final Recommendation

**Start small. Learn deeply. Build incrementally.**

The first milestone is not:

> "Finish RoadVision AI."

The first milestone is:

> **"Train and understand my first neural network."**

Then:

> **"Train and understand my first CNN."**

Then:

> **"Understand how object detection works."**

Then:

> **"Build RoadVision AI."**

This ensures that the project becomes both a successful academic submission and a genuine transition from Classical Machine Learning into Deep Learning and Computer Vision.
