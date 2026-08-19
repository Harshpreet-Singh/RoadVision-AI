# Project Planning Document

## RoadVision AI: Intelligent Road Damage Detection and Maintenance Prioritization Using Computer Vision

**Team Size:** 4 Students
**Year:** 2nd Year
**Timeline:** August–December (Approx. 4 months)
**Domain:** Artificial Intelligence + Computer Vision

---

# 1. Why We Selected This Project

Among multiple AI/IoT/Blockchain ideas, Road Damage Detection was selected because it has:

* Strong industry relevance
* Real-world impact
* Good resume value
* No mandatory hardware requirement
* Availability of public datasets
* Scope for future improvements

The project is challenging enough to stand out but achievable for second-year students.

---

# 2. Project Objective

Build an AI-based system that automatically detects road damage from images/videos and helps authorities prioritize maintenance.

The system should identify:

* Potholes
* Cracks
* Road defects

and provide:

* Detection result
* Confidence score
* Damage severity
* Inspection report

---

# 3. Final Project Name Ideas

Recommended:

**RoadVision AI: Intelligent Road Damage Detection and Maintenance Prioritization System**

Other options:

* Smart Road Inspection System Using Computer Vision
* AI-Based Automated Road Condition Monitoring System

---

# 4. Technology Stack

## Programming Language

### Python 3.11

Recommended because:

* Stable
* Compatible with AI libraries
* Good support for PyTorch and OpenCV

---

# 5. Technologies Explained

## OpenCV

Purpose:
Computer vision library.

Used for:

* Reading images
* Processing videos
* Webcam access
* Drawing detection boxes
* Image manipulation

Flow:

Image → OpenCV → AI Model → Output Image

---

## PyTorch

Purpose:
Deep learning framework.

Used for:

* Building neural networks
* Training AI models
* Loading trained models
* Making predictions

PyTorch is the "brain" that allows AI learning.

---

## CNN (Convolutional Neural Network)

Purpose:
A deep learning architecture used for image understanding.

CNN learns:

Pixels
↓
Edges
↓
Shapes
↓
Patterns
↓
Objects

Used in:

* Image recognition
* Medical imaging
* Self-driving cars
* Road damage detection

---

## YOLO

Full form:

You Only Look Once

Purpose:
Object detection model.

Difference:

Image Classification:

Image → "Pothole"

Object Detection:

Image → Pothole + Location + Confidence

YOLO can:

* Detect multiple objects
* Draw bounding boxes
* Give confidence scores

Example:

Pothole detected

Confidence: 96%

---

## Streamlit

Purpose:
Create a simple web interface.

Example:

User uploads image

↓

AI detects damage

↓

Website displays result

Used for:

* Demo
* Dashboard
* Presentation

Recommended for this project.

---

## FastAPI

Purpose:
Create APIs.

Used when:

* Mobile apps need AI
* Backend services are required
* Production deployment is needed

Not necessary initially.

---

# 6. Project Architecture

Complete flow:

Road Image/Video

↓

OpenCV

↓

YOLO Model

↓

PyTorch

↓

Detection Result

↓

Streamlit Dashboard

↓

User Output

---

# 7. Hardware Requirements

## Required

None.

Only:

* Laptop
* Internet
* Webcam (optional)

## Optional Future Hardware

* Mobile camera
* Raspberry Pi
* NVIDIA Jetson Nano
* GPS module

Start software-first.

---

# 8. Team Division (4 Members)

## Member 1: AI/ML Lead

Responsibilities:

* Learn YOLO
* Train model
* Fine tune model
* Evaluate accuracy

Deliverables:

* Trained model
* Model weights
* Accuracy report

Skills:

* Python
* PyTorch
* YOLO
* Deep Learning

---

## Member 2: Dataset & Research Lead

Responsibilities:

* Find datasets
* Clean data
* Data preprocessing
* Research papers
* Documentation

Deliverables:

* Dataset
* Literature review
* Dataset analysis

Skills:

* OpenCV
* Data handling
* Research

---

## Member 3: Frontend/UI Lead

Responsibilities:

Build Streamlit application.

Features:

* Upload image
* Upload video
* Show detected damage
* Dashboard
* Reports

Skills:

* Streamlit
* Python
* UI design

---

## Member 4: Backend & Integration Lead

Responsibilities:

* Connect model with UI
* Database
* Project structure
* Testing
* GitHub management

Skills:

* Python
* SQLite
* Git

---

# 9. Should Everyone Follow Same Roadmap?

No.

Everyone should learn the basics together first.

First 2–3 weeks:

Everyone learns:

* Python
* OpenCV basics
* CNN basics
* YOLO basics
* Git

After that:

Members specialize.

Everyone should still understand the complete project for viva.

---

# 10. Timeline

## August: Foundation

Goals:

* Python revision
* OpenCV basics
* CNN understanding
* Learn YOLO
* Run pretrained YOLO model

---

## September: AI Development

Goals:

* Select dataset
* Train YOLO model
* Evaluate results
* Improve accuracy

---

## October: Application Development

Goals:

* Build Streamlit app
* Image upload
* Video detection
* Display results

---

## November: Improvement Phase

Add:

* Severity estimation
* Detection history
* Reports
* Better UI
* Testing

---

## December: Final Phase

Complete:

* Documentation
* Presentation
* Demo
* Viva preparation

---

# 11. Features

## Basic Features

✓ Upload image
✓ Upload video
✓ Detect potholes
✓ Detect cracks
✓ Confidence score

## Advanced Features

✓ Damage severity estimation

Example:

Low → Monitor

Medium → Repair Soon

High → Immediate Action

✓ Inspection reports

✓ Detection history

✓ Dashboard analytics

✓ GPS mapping (future)

---

# 12. Learning Roadmap

Current Level:

Random Forest + Basic ML

Next:

Python
↓
NumPy/Pandas
↓
OpenCV
↓
Neural Networks
↓
CNN
↓
PyTorch
↓
YOLO
↓
Computer Vision Project
↓
Deployment

---

# 13. Important Advice

Do not:

* Spend weeks only learning theory
* Try to master everything before starting
* Add unnecessary hardware early

Do:

* Build small versions quickly
* Learn while building
* Maintain GitHub
* Document progress

---

# Final Mentor Recommendation

For a second-year team wanting to stand out:

Recommended Project:

**RoadVision AI: Intelligent Road Damage Detection and Maintenance Prioritization Using Computer Vision**

Reason:

* Good difficulty level
* Strong AI learning
* Industry relevance
* Good portfolio value
* Expandable for future projects

The goal is not just to make a working model.

The goal is to build a complete AI product.
