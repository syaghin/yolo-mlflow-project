import os
import glob
import cv2
import mlflow
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ------------------- CONFIGURATION -------------------
# Set up MLflow tracking URI and experiment for classification
MLFLOW_TRACKING_URI = "http://xx.xx.xx.xx:xxxx"
EXPERIMENT_NAME = "Yolo Classification"

# Model configuration for classification
MODEL_ARCH = "yolo11n-cls.pt"  
RUN_NAME = "Classification-xx"

# Data paths
DATA_YAML = "../dataset_classify"
TEST_DATASET = "../dataset_classify/test" 

# Set up MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Change to True to continue from previous checkpoint/run.
resume_training = False

# ------------------- HELPER FUNCTIONS -------------------
def train_yolo(data_yaml, model_arch, run_name, resume=False):
    """
    Train the YOLO classification model and log training parameters to MLflow.
    
    Returns:
      - run_id: The MLflow run ID.
      - best_model_path: Path to the best model weights.
    """
    # End any active MLflow run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    # Set directory and file to save run_id
    run_id_dir = os.path.join("runs", "classify", run_name)
    os.makedirs(run_id_dir, exist_ok=True)
    run_id_file = os.path.join(run_id_dir, f"{run_name}_run_id.txt")
    
    # If resume=True, make sure run_id file is exist
    if resume:
        if os.path.isfile(run_id_file):
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
            print(f"Resuming run with run id: {run_id}")
            run_context = mlflow.start_run(run_id=run_id)
        else:
            raise FileNotFoundError(
                f"Run id file '{run_id_file}' tidak ditemukan. "
                "Pastikan training sebelumnya sudah dijalankan dengan resume=False."
            )
    else:
        # Start new training process and save new run id
        run_context = mlflow.start_run(run_name=run_name)
        run_id = run_context.info.run_id
        with open(run_id_file, "w") as f:
            f.write(run_id)
        print(f"Run id disimpan di: {run_id_file}")

    with run_context as run:
        # If resume, check the last checkpoint training
        if resume:
            checkpoint_path = os.path.join("runs", "classify", run_name, "weights", "last.pt")
            if os.path.isfile(checkpoint_path):
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                model = YOLO(checkpoint_path)
            else:
                print("No checkpoint found; starting training from scratch.")
                model = YOLO(model_arch)
        else:
            model = YOLO(model_arch)

        # Start training
        results = model.train(
            data=data_yaml,
            task="classify",
            name=run_name,
            resume=resume,
            exist_ok=True,
        )

        # Decides best model path
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if not best_model_path.exists():
            best_model_path = Path(results.save_dir) / "weights" / "last.pt"
        
        # Log param to MLflow
        mlflow.log_param("best_model_path", str(best_model_path))

        print(f"Training completed. Run ID: {run_id}")
    
    mlflow.end_run()
    return run_id, str(best_model_path)

def predict_yolo(model, root_dataset, run_id):
    """
    Perform classification predictions using the trained YOLO model.
    The function computes and logs performance metrics (accuracy, F1 score, precision, recall) to MLflow.
    
    Returns:
      - y_true: List of true class indices.
      - y_pred: List of predicted class indices (top-1 prediction).
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_id=run_id):
        # Collect class names from dataset subdirectories (alphabetically sorted)
        class_names = sorted(d.name for d in os.scandir(root_dataset) if d.is_dir())
        
        # Assume model.names is a dictionary mapping index -> class name
        idx_to_class = model.names  
        # Create a mapping from class name to index for ground truth extraction
        class_to_idx = {name: idx for idx, name in idx_to_class.items()}
        
        y_true, y_pred = [], []
        
        # Loop through each class folder in the dataset
        for class_name in class_names:
            class_folder = os.path.join(root_dataset, class_name)
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPEG"]:
                image_paths.extend(glob.glob(os.path.join(class_folder, ext)))
            
            for img_path in image_paths:
                # Derive ground truth index from folder name
                true_class_idx = class_to_idx.get(class_name, -1)
                y_true.append(true_class_idx)
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Gagal membaca {img_path}")
                    y_pred.append(-1)
                    continue
                
                # Get prediction from the model
                results = model.predict(img)
                if not results:
                    print(f"Tidak ada hasil prediksi untuk {img_path}")
                    y_pred.append(-1)
                    continue
                
                res = results[0]
                # Extract top-1 prediction index from the prediction probabilities
                pred_class_idx = int(res.probs.top1)
                y_pred.append(pred_class_idx)
                
                # Compute and log cumulative classification metrics after each prediction
                metrics = {
                    "test/accuracy": accuracy_score(y_true, y_pred),
                    "test/macro/f1_score": f1_score(y_true, y_pred, average="macro"),
                    "test/micro/f1_score": f1_score(y_true, y_pred, average="micro"),
                    "test/macro/precision": precision_score(y_true, y_pred, average="macro"),
                    "test/micro/precision": precision_score(y_true, y_pred, average="micro"),
                    "test/macro/recall": recall_score(y_true, y_pred, average="macro"),
                    "test/micro/recall": recall_score(y_true, y_pred, average="micro")
                }
                mlflow.log_metrics(metrics)
       
        print("Prediksi dan evaluasi selesai. Metrik telah di-log.")
    mlflow.end_run()

    return y_true, y_pred

if __name__ == "__main__":

    # 1. Train the classification model
    new_run_id, best_model_path = train_yolo(DATA_YAML, MODEL_ARCH, RUN_NAME, resume=resume_training)
    
    # 2. Load the best trained model
    best_model = YOLO(best_model_path)
    
    # 3. Perform predictions on the test dataset and log metrics
    predict_yolo(best_model, TEST_DATASET, new_run_id)