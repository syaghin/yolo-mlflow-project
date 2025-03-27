# YOLO-MLflow Project

This repository contains two main scripts for **Object Detection** and **Image Classification** using [Ultralytics YOLO](https://docs.ultralytics.com/) and [MLflow](https://mlflow.org/). This project aims to train YOLO models and evaluate metrics results of model through clear graphical representations.

---

## 1. Overview

- **Detection**  
  - **Script**: `mlflow_detection.py`  
  - **Purpose**: Uses YOLO for object detection with bounding boxes.

- **Classification**  
  - **Script**: `mlflow_classify.py`  
  - **Purpose**: Uses YOLO for multi-class image classification.

Both scripts support resuming training from a previous run by saving and loading **run IDs** and **model checkpoints**.

---

## 2. Setup

### 2.1. Install Dependencies

Ensure you have Python 3.7+ installed. Then, install the required packages:
```bash
pip install ultralytics mlflow opencv-python scikit-learn
```

### 2.2. Configure MLflow

In both scripts, update the following variables at the top of the file:

- **`MLFLOW_TRACKING_URI`**  
  Set this to MLflow tracking server URL. For example:
    ```python
    MLFLOW_TRACKING_URI = "http://xx.xx.xxx.xx:xxxx"

- **`EXPERIMENT_NAME`**  
  Choose experiment name which is already in MLflow server. For example:
  ```python
  EXPERIMENT_NAME = "Plate-Detection"

- **`RUN_NAME`**  
  Choose a meaningful run name that reflects the task. For example:
  ```python
  EXPERIMENT_NAME = "Best_model_1"

### 2.3. Adjust Models & Dataset Paths

Each script contains configurable variables for model and dataset paths.

- **For Object Detection (in `mlflow_detection.py`)**
  Model Configuration
  ```python
  MODEL_NAME = "yolo11n.pt"  # Path to your YOLO detection model
  ```
  Dataset Path
  ```python
  DATA_YAML = "/path/to/dataset_detection/data.yaml"
  # Ensure your dataset follows the YOLO structure (separate folders for images and labels).
  ```

- **For Image Classification (in `mlflow_classify.py`)**
  Model Configuration
  ```python
  MODEL_ARCH = "yolo11n-cls.pt"  # Path to your YOLO classification model
  ```
  Dataset Paths
  ```python
  DATA_YAML = "/path/to/dataset_classification"         # Main folder for classification dataset
  TEST_DATASET = "/path/to/dataset_classification/test" # Folder for testing images
  ```

### Resume Training
Both scripts use a boolean variable, typically named resume_training, to decide whether to start a new training run or resume from a checkpoint. Adjust it as needed.

---

## 3. MLflow Tracking

When training and evaluating your models, MLflow automatically logs key metrics:

- **Object Detection**   
  - `test/box_mAP`: Mean Average Precision over multiple IoU thresholds (0.5:0.95)  
  - `test/box_mAP50`: Mean Average Precision at IoU = 0.5 
  - `test/box_mAP75`: Mean Average Precision at IoU = 0.75

- **Image Classification**  
  - `test/accuracy`: Overall model accuracy  
  - `test/macro/f1_score` & `test/micro/f1_score`: F1 scores (macro and micro)  
  - `test/macro/precision` & `test/micro/precision`: Precision (macro and micro)  
  - `test/macro/recall` & `test/micro/recall`: Recall (macro and micro)

---

### Example usage:
1. Start training your classification model.
2. After training, metrics will be automatically logged.
3. Launch MLflow UI to compare or visualize your runs
