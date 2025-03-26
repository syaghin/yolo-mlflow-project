# YOLO-MLflow Project  

This repository contains two main scripts for **Object Detection** and **Image Classification** using [Ultralytics YOLO](https://docs.ultralytics.com/) and [MLflow](https://mlflow.org/). Key features include:

- **Resume Training**: Easily continue training from your last checkpoint.  
- **MLflow Logging**: Automatically track metrics and parameters in MLflow.  
- **Custom Datasets**: Supports both detection and classification datasets.

## 1. Overview

1. **Detection**  
   - Script example: `detection-mlflow.py`  
   - Uses YOLO for object detection with bounding boxes.  
2. **Classification**  
   - Script example: `classification-mlflow.py`  
   - Uses YOLO for multi-class image classification.

Both scripts allow you to resume training from a previous run, saving and loading **run IDs** as well as **model checkpoints**.

## 2. Setup

1. **Install Dependencies**  
   ```bash
   pip install ultralytics mlflow opencv-python scikit-learn
2. **Configure MLflow**
   - Update MLFLOW_TRACKING_URI to your MLflow server URI.
   - Set EXPERIMENT_NAME to an appropriate experiment name.
3. **Prepare Datasets**
   - Detection: Follow the YOLO directory structure (images + labels) and update your .yaml data file.
   - Classification: Organize images in subfolders per class (train/val/test) or use split=... for auto-splitting.
  
