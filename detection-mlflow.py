import os
import mlflow
from ultralytics import YOLO

# ------------------- CONFIGURATION -------------------
# Set up MLflow tracking URI and experiment for detection
MLFLOW_TRACKING_URI = "http://xx.xx.xx.xx:xxxx"
EXPERIMENT_NAME = "Yolo Detection"

# Model configuration for detection
MODEL_NAME = "yolo11n.pt"
RUN_NAME = "Detection-xx"

# Data paths
DATA_YAML = "../data.yaml"

# Change to True to continue from previous checkpoint/run.
resume_training = True

# ------------------- MLflow SETUP -------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ------------------- HELPER FUNCTIONS -------------------
def train_yolo(data_yaml, model_name, run_name, resume=False):
    """
    Train the YOLO model and log training metrics to MLflow.
    """
    # End any active MLflow run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    # Set directory and file to save run_id
    run_id_dir = os.path.join("runs", "detect", run_name)
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
            checkpoint_path = os.path.join("runs", "detect", run_name, "weights", "last.pt")
            if os.path.isfile(checkpoint_path):
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                model = YOLO(checkpoint_path)
            else:
                print("No checkpoint found; starting training from scratch.")
                model = YOLO(model_name)
        else:
            model = YOLO(model_name)
        
        # Start training
        results = model.train(
            data=data_yaml,
            name=run_name,
            resume=resume,
            exist_ok=True
        )

        mlflow.log_metrics({"train/box_mAP50": float(results.box.map50)})
        print(f"Training completed. Run ID: {run_id}")

    mlflow.end_run()
    return run_id

def validate_yolo(data_yaml, run_name, run_id):
    """
    Validate the trained YOLO model and log validation metrics to MLflow.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_id=run_id):
        model_path = os.path.join("runs", "detect", run_name, "weights", "best.pt")
        model = YOLO(model_path)
        validation_results = model.val(data=data_yaml)
        metrics = {
            "test/box_mAP": float(validation_results.box.map),
            "test/box_mAP50": float(validation_results.box.map50),
            "test/box_mAP75": float(validation_results.box.map75)
        }
        mlflow.log_metrics(metrics)
        print("Validation completed. Metrics logged.")

    mlflow.end_run()

if __name__ == "__main__":

    new_run_id = train_yolo(DATA_YAML, MODEL_NAME, RUN_NAME, resume=resume_training)
    validate_yolo(DATA_YAML, RUN_NAME, new_run_id)